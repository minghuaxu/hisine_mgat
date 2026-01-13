#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py (Fixed with Task Decoupling & Gated Fusion)
====================================================
本模块实现了 Motif-Guided SINE Classifier 的核心架构。

核心技术亮点 (Key Technologies):
1. Task Decoupling (任务解耦): 
   - 使用 Shared Adapter 分离特征空间，防止分类与分割任务直接冲突。
   - 实现了"辅助任务(分割)倒逼主任务(分类)"的梯度回传机制。

2. Motif-Aware Attention (Motif 感知注意力): 
   - 结合先验的 Motif Mask (A-box/B-box) 增强关键区域的特征提取。

3. Gated Fusion (门控融合): 
   - 引入可学习的 Gate Layer，动态融合"全局保底特征"与"结构引导特征"。
   - 机制：当分割边界清晰时 (SINE特征明显)，权重偏向结构特征；当边界模糊时 (模拟数据/背景)，自动回退到全局特征，保证分类鲁棒性。

4. CRF (条件随机场): 
   - 用于精确的序列边界解码 (TSD/Body/PolyA)，保证生物学结构的完整性。

5. Selective CRF Loss (选择性 CRF 损失): 
   - 仅在正样本 (SINE) 上计算 CRF Loss，避免负样本的无意义边界干扰分割头学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

class MotifConfidenceModule(nn.Module):
    """
    [模块 1] Motif置信度
    功能：判断输入的 Motif Mask 是否包含有效信号。
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
        
        # 2. 生成置信度分数
        confidence = self.net(motif_region_embedding)
        return confidence

class MotifAwareAttention(nn.Module):
    """
    [模块 2] Motif 感知注意力层
    功能：在 Self-Attention 中融入 Motif 的位置信息。
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,  hidden_states, motif_mask, attention_mask):
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # 1. 标准自注意力
        attn_output, _ = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False  # 添加这个参数，节省显存
        )
        attn_output = self.dropout(attn_output)
        
        # 2. Motif 加权 (Hard Attention Guidance)
        # clamp(min=0.5) 保证非 Motif 区域也能保留至少 50% 的原始注意力流
        motif_weights = motif_mask.unsqueeze(-1).clamp(min=0.5)
        scaled_output = attn_output * motif_weights
        
        # 3. 残差连接 + LayerNorm
        output = self.layer_norm(hidden_states + scaled_output)
        
        return output


class MotifGuidedSINEClassifier(nn.Module):
    """
    [主模型] 端到端 SINE 分类器
    集成了：Task Decoupling, Gated Fusion, CRF
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
        
        # 1. 特征增强模块
        self.motif_attention = MotifAwareAttention(
            hidden_dim=self.backbone_dim, num_heads=8, dropout=dropout
        )
        # 置信度模块
        self.confidence_module = MotifConfidenceModule(self.backbone_dim)

        # 2. 门控融合层 (Gated Fusion)
        # 输入维度是 hidden_dim * 2 (Global + Structural)，输出 1 个 alpha 值
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1), 
            nn.Sigmoid()
        )
        
        # 3. 共享适配层 (Shared Adapter)
        # 分类与分割任务共用此层，实现特征空间的初步解耦
        self.shared_adapter = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. 任务头 (Task Heads)
        # 序列分割头
        self.token_classifier = nn.Linear(hidden_dim, num_token_labels)
        
        # 全局分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), 
            nn.GELU(),
            nn.Dropout(dropout),         
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 5. CRF 层
        self.crf = CRF(num_token_labels, batch_first=True) # batch_first=True 表示输入维度是 (Batch, Seq_Len, Tags)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        token_labels: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        # ---------------------------------------------------
        # Step 1: 基础特征提取 (Backbone + Motif Attention)
        # ---------------------------------------------------
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
       
        # 动态调整 Motif Mask
        confidence_score = self.confidence_module(hidden_states, motif_mask)
        refined_motif_mask = motif_mask * (0.5 + 0.5 * confidence_score)
        
        # Motif感知注意力
        enhanced_states = self.motif_attention(
            hidden_states, 
            refined_motif_mask, 
            attention_mask=attention_mask
        )
        
        # 经过共享适配层
        shared_features = self.shared_adapter(enhanced_states)

        # 3. 序列标注输出 (Emissions)
        emissions = self.token_classifier(shared_features)

        # ---------------------------------------------------
        # Step 2: 双路特征聚合 (Dual-Path Aggregation)
        # ---------------------------------------------------

        # Path A: 全局通路 (Global Path) - "保底视力"
        # 传统的 Mean Pooling，保证在边界不清晰时模型依然能工作
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        global_repr = (shared_features * input_mask_expanded).sum(dim=1) / input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # Path B: 结构通路 (Structural Path) - "精准视力"
        # 利用分割头的预测结果 (Foreground Probability) 来加权特征
        emissions = self.token_classifier(shared_features)  # (Batch, Seq, NumLabels)

        with torch.no_grad():
            # 计算前景概率 (Foreground Probability)
            # 假设 index 0 是 Background (O), 我们取 1 - P(Bg)
            probs = torch.softmax(emissions, dim=-1)
            foreground_prob = 1.0 - probs[:, :, 0]  # 0 is Background
            
        # 注意: 这里没有 detach()，允许分类 Loss 对分割头进行梯度反馈 (Auxiliary Learning)
        pooling_weights = foreground_prob * attention_mask

        # 结构化加权池化
        structural_repr = (shared_features * pooling_weights.unsqueeze(-1)).sum(dim=1)
        sum_weights = pooling_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        structural_repr = structural_repr / sum_weights

        # ---------------------------------------------------
        # Step 3: 门控融合 (Gated Fusion)
        # ---------------------------------------------------
        # alpha 决定了模型多大程度上依赖"结构特征"
        # alpha 越接近 1，表示边界越清晰，模型越自信
        gate_input = torch.cat([global_repr, structural_repr], dim=-1)
        alpha = self.gate_layer(gate_input)

        # 动态融合 [CRITICAL STEP]
        final_cls_repr = alpha * structural_repr + (1 - alpha) * global_repr

        # ---------------------------------------------------
        # Step 4: 输出与 Loss 计算
        # ---------------------------------------------------
        global_logits = self.classifier(final_cls_repr)

        # 计算 CRF Loss (仅在正样本上)
        crf_loss = torch.tensor(0.0, device=input_ids.device)
        
        if token_labels is not None and labels is not None:
            # 筛选有效区域
            valid_mask = (attention_mask.bool() & (token_labels != -100))
            # 筛选正样本 (SINE)
            pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0:
                sub_emissions = emissions[pos_indices]
                sub_labels = token_labels[pos_indices]
                sub_mask = valid_mask[pos_indices]
                
                # 安全处理：Mask 掉的地方 label 设为 0 (虽不参与计算但 CRF 需要合法输入)
                safe_sub_labels = sub_labels.clone().masked_fill(~sub_mask, 0)
                
                # 关闭混合精度以保证 CRF 数值稳定
                with torch.amp.autocast('cuda', enabled=False):
                    log_likelihood = self.crf(sub_emissions.float(), safe_sub_labels, mask=sub_mask, reduction='sum')
                    num_valid_tokens = sub_mask.sum().float().clamp(min=1.0)
                    crf_loss = -log_likelihood / num_valid_tokens
            
        return global_logits, emissions, crf_loss

    def decode(self, emissions, attention_mask):
        """
        [新增] 使用 Viterbi 算法解码最佳路径
        返回: List[List[int]] (Batch 中每个样本的 Tag 序列)
        """
        mask = attention_mask.bool()
        # 强制 mask 第一位为 True (防止全 0 mask 导致 crash)
        mask[:, 0] = True 
        return self.crf.decode(emissions, mask=mask)
    
    def predict_proba(self, input_ids, attention_mask, motif_mask):
        """仅用于推理全局概率"""
        # 注意：预测时不传 labels
        global_logits, _, _ = self.forward(input_ids, attention_mask, motif_mask)
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
    """
    处理类别不平衡的损失函数
    """
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