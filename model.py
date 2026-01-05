#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py
========
Motif-Guided SINE分类器 - 适配offset_mapping对齐

关键改进:
1. 修复了 num_token_labels 未定义的 Bug
2. 移除了错误的reshape逻辑
3. 直接使用对齐好的token级motif_mask
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
            key_padding_mask=key_padding_mask 
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
    端到端的SINE分类器 (含 Segmentation Head)
    """
    
    def __init__(
        self,
        backbone,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_token_labels: int = 5,  # [修复] 必须添加这个参数，默认值为 5
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
        
        # Motif感知注意力
        self.motif_attention = MotifAwareAttention(
            hidden_dim=self.backbone_dim,
            num_heads=8,
            dropout=dropout
        )
        # 置信度模块
        self.confidence_module = MotifConfidenceModule(self.backbone_dim)
        
        # 1. 全局分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 2. 序列标注头 (Segmentation Head)
        # 这里使用了传入的 num_token_labels 参数
        self.token_classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
        # 形状验证
        batch_size, seq_len = input_ids.shape
        
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
        
        # 2. Motif感知注意力
        enhanced_states = self.motif_attention(
            hidden_states, 
            refined_motif_mask, 
            attention_mask=attention_mask
        )
        
        # --- 分支 1: 全局分类 ---
        # 加权池化
        weights = motif_mask.unsqueeze(-1) * attention_mask.unsqueeze(-1)
        weighted_sum = (enhanced_states * weights).sum(dim=1)
        sum_weights = weights.sum(dim=1).clamp(min=1e-9)
        sequence_representation = weighted_sum / sum_weights
        
        global_logits = self.classifier(sequence_representation)

        # --- 分支 2: 序列标注 ---
        # 对每个 token 进行分类
        token_logits = self.token_classifier(enhanced_states) # (B, L, Num_Labels)
        
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