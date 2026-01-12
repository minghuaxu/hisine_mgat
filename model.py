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
from torchcrf import CRF

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
        
        # motif_weights = motif_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        # scaled_output = attn_output * motif_weights
        motif_weights = motif_mask.unsqueeze(-1).clamp(min=0.5)  # 最低 0.5，保证基本注意力
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
        
        # --- [修正点 1] 朋友建议：共享投影层 ---
        # 两个任务共用这个层。
        # 好处：负样本进来时，分类 Loss 会更新这个层，防止它变成“只懂正样本的瞎子”。
        self.shared_adapter = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # --- [修正点 2] 任务头 ---
        self.token_classifier = nn.Linear(hidden_dim, num_token_labels)
        
        # 分类头：输入依然是 hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # batch_first=True 表示输入维度是 (Batch, Seq_Len, Tags)
        self.crf = CRF(num_token_labels, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        token_labels: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        # 1. Backbone编码
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
       
        # Motif 置信度
        confidence_score = self.confidence_module(hidden_states, motif_mask)
        # refined_motif_mask = motif_mask * confidence_score
        refined_motif_mask = motif_mask * (0.5 + 0.5 * confidence_score)
        
        # 2. Motif感知注意力
        enhanced_states = self.motif_attention(
            hidden_states, 
            refined_motif_mask, 
            attention_mask=attention_mask
        )
        
        # 2. 共享特征提取
        # 无论是分类还是分割，都基于这个 shared_features
        shared_features = self.shared_adapter(enhanced_states)

        # 3. 序列标注输出 (Emissions)
        emissions = self.token_classifier(shared_features)

        # ---  硬核耦合 (Segmentation-Guided Pooling) ---
        # 我们不再只用 motif_mask 做池化，而是利用模型自己预测的“前景概率”来做池化。
        # 逻辑：如果模型认为这块是 Body/TSD/PolyA，那么这块特征对分类就更重要。
        
        with torch.no_grad():
            # 计算前景概率 (Foreground Probability)
            # 假设 index 0 是 Background (O)
            probs = torch.softmax(emissions, dim=-1)
            # p_foreground = 1 - p_background
            foreground_prob = 1.0 - probs[:, :, 0] 
            
            # 结合原本的 attention_mask
            pooling_weights = foreground_prob * attention_mask

        # 加权平均池化 (Weighted Mean Pooling)
        # 形状: (Batch, Seq, 1) * (Batch, Seq, Hidden) -> Sum -> (Batch, Hidden)
        weighted_sum = (shared_features * pooling_weights.unsqueeze(-1)).sum(dim=1)
        sum_weights = pooling_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        seg_guided_repr = weighted_sum / sum_weights
        
        # 为了稳健，我们还是加上全局平均池化 (Global Mean Pooling) 做残差
        # 这样即使一开始分割预测很烂，分类也能靠全局特征撑住
        global_mean_repr = (shared_features * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        final_cls_repr = seg_guided_repr + global_mean_repr
        
        global_logits = self.classifier(final_cls_repr)

        # 4. CRF Loss 计算 (保留我的“切片逻辑”，这依然是必要的！)
        # 虽然层共享了，但 CRF Loss 的数值如果太大还是会干扰。
        # 且我们依然不需要对负样本计算复杂的 CRF 路径。
        crf_loss = torch.tensor(0.0, device=input_ids.device)
        
        if token_labels is not None and labels is not None:
            valid_mask = (attention_mask.bool() & (token_labels != -100))
            pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0:
                sub_emissions = emissions[pos_indices]
                sub_labels = token_labels[pos_indices]
                sub_mask = valid_mask[pos_indices]
                
                safe_sub_labels = sub_labels.clone().masked_fill(~sub_mask, 0)
                
                with torch.cuda.amp.autocast(enabled=False):
                    log_likelihood = self.crf(sub_emissions.float(), safe_sub_labels, mask=sub_mask, reduction='sum')
                    num_valid_tokens = sub_mask.sum().float().clamp(min=1.0)
                    crf_loss = -log_likelihood / num_valid_tokens
            
        return global_logits, emissions, crf_loss

    def decode(self, emissions, attention_mask):
        """
        [新增] 使用 Viterbi 算法解码最佳路径
        返回: List[List[int]] (Batch 中每个样本的 Tag 序列)
        """
        # 同样，确保 decode 时的 mask 第一位也是 True
        mask = attention_mask.bool()
        mask[:, 0] = True 
        return self.crf.decode(emissions, mask=mask)
    
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