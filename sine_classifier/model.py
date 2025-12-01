#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py
========
Motif-Guided SINE分类器 - 适配offset_mapping对齐

关键改进:
1. 移除了错误的reshape逻辑
2. 直接使用对齐好的token级motif_mask
3. 更清晰的注意力机制实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifAwareAttention(nn.Module):
    """
    Motif感知的注意力层
    
    核心思想:
    - 使用motif_mask直接调制注意力输出
    - motif_mask已经通过offset_mapping精确对齐到每个token
    - 增强motif区域，抑制非motif区域
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
        """
        前向传播
        
        参数:
            hidden_states: (batch_size, seq_len, hidden_dim)
            motif_mask: (batch_size, seq_len) - 已对齐的token级权重
        
        返回:
            output: (batch_size, seq_len, hidden_dim)

        PyTorch MultiheadAttention key_padding_mask: True 表示要忽略 (padding)
        我的 attention_mask: 1 表示有效, 0 表示 padding
        所以需要取反: key_padding_mask = (attention_mask == 0)
        """
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
        # 关键: motif_mask已经与hidden_states的seq_len维度完全对齐
        # 形状检查
        assert motif_mask.size(1) == hidden_states.size(1), \
            f"Motif mask长度({motif_mask.size(1)})与hidden states({hidden_states.size(1)})不匹配"
        
        motif_weights = motif_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        scaled_output = attn_output * motif_weights
        
        # 3. 残差连接 + LayerNorm
        output = self.layer_norm(hidden_states + scaled_output)
        
        return output


class MotifGuidedSINEClassifier(nn.Module):
    """
    端到端的SINE分类器
    
    改进点:
    - 移除了复杂的mask对齐逻辑
    - 直接使用dataset提供的对齐mask
    - 更简洁、更易维护
    """
    
    def __init__(
        self,
        backbone,
        hidden_dim: int = 256,
        num_classes: int = 2,
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
            print("[INFO] Backbone参数已冻结")
        
        # Motif感知注意力
        self.motif_attention = MotifAwareAttention(
            hidden_dim=self.backbone_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # 分类头
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            input_ids: (batch_size, seq_len) - tokenizer输出
            attention_mask: (batch_size, seq_len) - padding mask
            motif_mask: (batch_size, seq_len) - 对齐的motif权重
        
        返回:
            logits: (batch_size, num_classes)
        """
        # 形状验证
        batch_size, seq_len = input_ids.shape
        assert motif_mask.shape == (batch_size, seq_len), \
            f"Motif mask形状不匹配: 期望{(batch_size, seq_len)}, 实际{motif_mask.shape}"
        
        # 1. Backbone编码
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        
        # 2. Motif感知注意力
        enhanced_states = self.motif_attention(
            hidden_states, 
            motif_mask, 
            attention_mask=attention_mask
        )
        
        # 使用加权池化代替 CLS
        # 确保 mask 维度匹配 (batch, seq_len, 1)
        weights = motif_mask.unsqueeze(-1)
        
        # 将 padding 位置的权重设为 0
        weights = weights * attention_mask.unsqueeze(-1)
        
        # 加权求和: Sum(H_i * W_i)
        weighted_sum = (enhanced_states * weights).sum(dim=1)
        
        # 权重之和 (避免除以0)
        sum_weights = weights.sum(dim=1).clamp(min=1e-9)
        
        # 得到加权平均的序列表示
        sequence_representation = weighted_sum / sum_weights

        logits = self.classifier(sequence_representation)
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: torch.Tensor
    ) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(input_ids, attention_mask, motif_mask)
        return torch.softmax(logits, dim=1)
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motif_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        获取[CLS]位置的注意力权重用于可视化
        
        返回:
            attention_weights: (batch_size, seq_len)
        """
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            
            # 最后一层的注意力
            last_attn = outputs.attentions[-1]
            
            # [CLS]位置(0)的注意力，在所有头上平均
            cls_attention = last_attn[:, :, 0, :].mean(dim=1)
            
            return cls_attention


class FocalLoss(nn.Module):
    """
    Focal Loss用于类别不平衡
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== 可视化工具 ====================

def visualize_model_attention(
    model,
    input_ids,
    attention_mask,
    motif_mask,
    tokenizer,
    save_path=None
):
    """
    可视化模型的注意力分布与motif mask
    
    用于验证mask对齐的正确性
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    with torch.no_grad():
        # 获取模型注意力
        attn_weights = model.get_attention_weights(
            input_ids, attention_mask, motif_mask
        )
    
    # 只可视化第一个样本
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
    attn = attn_weights[0].cpu().numpy()
    motif = motif_mask[0].cpu().numpy()
    
    # 只显示非padding部分
    valid_len = attention_mask[0].sum().item()
    tokens = tokens[:valid_len]
    attn = attn[:valid_len]
    motif = motif[:valid_len]
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    
    x = np.arange(len(tokens))
    
    # 注意力权重
    ax1.bar(x, attn, alpha=0.7, color='blue', label='Model Attention')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Model Attention Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Motif mask
    colors = ['red' if m > 1.5 else 'orange' if m > 0.9 else 'gray' for m in motif]
    ax2.bar(x, motif, alpha=0.7, color=colors, label='Motif Mask')
    ax2.set_ylabel('Motif Weight')
    ax2.set_xlabel('Token Position')
    ax2.set_title('Motif Mask Distribution (Red: A/B-box, Orange: TSD/PolyA, Gray: Background)')
    ax2.set_xticks(x[::5])
    ax2.set_xticklabels(tokens[::5], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化保存到: {save_path}")
    else:
        plt.show()


# ==================== 辅助函数 ====================

def count_parameters(model: nn.Module) -> dict:
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_pct': 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_model_summary(model: nn.Module):
    """打印模型摘要"""
    params = count_parameters(model)
    
    print("\n" + "="*60)
    print("模型摘要")
    print("="*60)
    print(f"总参数数量:       {params['total']:,}")
    print(f"可训练参数:       {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
    print(f"冻结参数:         {params['frozen']:,}")
    print("="*60 + "\n")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """模块测试"""
    print("测试Motif-Guided SINE分类器 (offset对齐版本)...")
    
    from transformers import AutoConfig, AutoModelForMaskedLM
    
    # 1. 创建backbone
    config = AutoConfig.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        trust_remote_code=True
    )
    backbone = AutoModelForMaskedLM.from_config(config)
    
    # 2. 创建分类器
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=256,
        num_classes=2,
        dropout=0.1
    )
    
    print_model_summary(model)
    
    # 3. 测试前向传播
    batch_size = 4
    seq_len = 50
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    motif_mask = torch.rand(batch_size, seq_len)
    
    print(f"输入形状:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  motif_mask: {motif_mask.shape}")
    
    # 前向传播
    try:
        logits = model(input_ids, attention_mask, motif_mask)
        probs = model.predict_proba(input_ids, attention_mask, motif_mask)
        
        print(f"\n输出形状:")
        print(f"  logits: {logits.shape}")
        print(f"  probs: {probs.shape}")
        
        # 验证形状
        assert logits.shape == (batch_size, 2), f"Logits形状错误: {logits.shape}"
        assert probs.shape == (batch_size, 2), f"Probs形状错误: {probs.shape}"
        
        print(f"\n✅ 形状验证通过")
    except AssertionError as e:
        print(f"\n❌ 形状验证失败: {e}")
        raise
    
    # 4. 测试损失函数
    targets = torch.randint(0, 2, (batch_size,))
    
    ce_loss = F.cross_entropy(logits, targets)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(logits, targets)
    
    print(f"\n损失函数:")
    print(f"  CrossEntropy Loss: {ce_loss.item():.4f}")
    print(f"  Focal Loss: {focal_loss.item():.4f}")
    
    print("\n✅ 所有测试通过!")