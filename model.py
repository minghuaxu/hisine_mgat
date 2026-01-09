#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py (Fixed with Task Decoupling)
=====================================
Motif-Guided SINE分类器 - 适配多任务解耦

修复核心：
1. 引入 Adapter 层解耦全局分类和序列分割任务，防止分割任务的梯度噪音干扰分类任务。
2. 优化了 Head 结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifConfidenceModule(nn.Module):
    """
    裁判员模块：判断这个 Motif 到底是不是真的
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 0~1，表示"这是真Motif的概率"
        )
    
    def forward(self, hidden_states, motif_mask):
        # 1. 提取 Motif 区域的特征
        mask_expanded = motif_mask.unsqueeze(-1)
        motif_region_features = (hidden_states * mask_expanded).sum(dim=1)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)
        motif_region_embedding = motif_region_features / mask_sum
        
        # 2. 裁判打分
        confidence = self.net(motif_region_embedding)
        return confidence

class MotifAwareAttention(nn.Module):
    """
    Motif感知的注意力层
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        motif_mask: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # 1. 自注意力
        attn_output, _ = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False  # 添加这个参数，节省显存
        )
        attn_output = self.dropout(attn_output)
        
        # 2. Motif加权
        assert motif_mask.size(1) == hidden_states.size(1), \
            f"Motif mask长度({motif_mask.size(1)})与hidden states({hidden_states.size(1)})不匹配"
        
        motif_weights = motif_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        scaled_output = attn_output * motif_weights
        
        # 3. 残差连接 + LayerNorm
        output = self.layer_norm(hidden_states + scaled_output)
        
        return output


class MotifGuidedSINEClassifier(nn.Module):
    """
    端到端的SINE分类器 (含 Task Decoupling & Segmentation Head)
    """
    
    def __init__(
        self,
        backbone,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_token_labels: int = 5,  # 0=Bg, 1=TSD, 2=Body, 3=PolyA, 4=Pad/-100
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.backbone = backbone
        self.backbone_dim = backbone.config.hidden_size
        self.num_classes = num_classes
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Motif感知注意力 (共享特征提取)
        self.motif_attention = MotifAwareAttention(
            hidden_dim=self.backbone_dim,
            num_heads=8,
            dropout=dropout
        )
        # 置信度模块
        self.confidence_module = MotifConfidenceModule(self.backbone_dim)
        
        # --- [新增] 任务适配层 (Task Adapters) ---
        # 关键修改：将两个任务的特征空间解耦，防止梯度冲突
        
        # 适配层 1: 用于全局分类
        self.cls_adapter = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 适配层 2: 用于序列分割 (边界预测)
        self.seg_adapter = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # --- 任务头 (Task Heads) ---
        
        # 1. 全局分类头 (接在 cls_adapter 后)
        self.classifier = nn.Sequential(
            # 输入已经是 hidden_dim
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 2. 序列标注头 (接在 seg_adapter 后)
        self.token_classifier = nn.Sequential(
            # 输入已经是 hidden_dim，直接映射到 token labels
            nn.Linear(hidden_dim, num_token_labels) 
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: torch.Tensor
    ):
        """
        前向传播返回两个 logits
        """
        # 1. Backbone编码
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
       

        # Motif 置信度
        confidence_score = self.confidence_module(hidden_states, motif_mask)
        refined_motif_mask = motif_mask * confidence_score
        
        # 2. Motif感知注意力 (增强特征)
        enhanced_states = self.motif_attention(
            hidden_states, 
            refined_motif_mask, 
            attention_mask=attention_mask
        )
        
        # --- 分支 1: 全局分类 (走 cls_adapter) ---
        # 先经过适配层，隔离特征
        cls_features = self.cls_adapter(enhanced_states)
        
        # 加权池化 (使用 cls 特征)
        weights = motif_mask.unsqueeze(-1) * attention_mask.unsqueeze(-1)
        weighted_sum = (cls_features * weights).sum(dim=1)
        sum_weights = weights.sum(dim=1).clamp(min=1e-9)
        sequence_representation = weighted_sum / sum_weights
        
        global_logits = self.classifier(sequence_representation)

        # --- 分支 2: 序列标注 (走 seg_adapter) ---
        # 先经过适配层，隔离特征
        seg_features = self.seg_adapter(enhanced_states)
        
        # 对每个 token 进行分类
        token_logits = self.token_classifier(seg_features) # (B, L, Num_Labels)
        
        return global_logits, token_logits
    
    def predict_proba(self, input_ids, attention_mask, motif_mask):
        """仅用于推理全局概率"""
        global_logits, _ = self.forward(input_ids, attention_mask, motif_mask)
        return torch.softmax(global_logits, dim=1)
    
    def get_attention_weights(self, input_ids, attention_mask, motif_mask):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            return outputs.attentions[-1][:, :, 0, :].mean(dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        targets_float = targets.float()
        alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss