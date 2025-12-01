#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py
=======
端到端SINE Dataset - 100%对齐官方tokenizer版本（已修复对齐问题）

关键改进：
1. 完全使用官方 tokenizer 构建 token-base 映射（decode single token）
2. 自动处理超长序列，保持 3' 端（左截断 token 级别）
3. CLS token 使用 mean（更合理）
4. padding 位置强制为 0.0
5. 去除了所有手动 k-mer 逻辑，杜绝对齐误差
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

def gaussian_kernel_smooth(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    对一维数组应用高斯平滑
    模拟 Motif 信号的生物学渐变特征
    """
    if sigma <= 0:
        return mask
    
    # 创建高斯核 (核大小 = 6*sigma + 1，保证覆盖绝大部分概率)
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # 归一化
    
    # 卷积平滑 (mode='same' 保持长度不变)
    smoothed = np.convolve(mask, kernel, mode='same')
    return smoothed
    
class SINEDatasetE2E(Dataset):
    def __init__(
        self,
        sequences_with_ids: List[Tuple[str, str]],
        labels: Optional[List[int]],
        motif_df: pd.DataFrame,
        tokenizer,
        max_token_length: int = 1024
    ):
        self.sequences_with_ids = sequences_with_ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.is_training = labels is not None
        
        self.motif_data = motif_df.set_index('unique_id', drop=False)
        
        if self.is_training:
            assert len(sequences_with_ids) == len(labels)

        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
        # [配置] 动态权重的范围 (min, max)
        # 训练时在这个范围内随机采样，预测时取均值
        self.weight_ranges = {
            'A_box': (1.8, 2.4),    # 核心特征，权重最高
            'B_box': (1.8, 2.4),
            'polyA': (1.4, 1.8),    # 次要特征
            'left_TSD': (0.8, 1.2), # 辅助特征
            'right_TSD': (0.8, 1.2),
            'background': (0.2, 0.4) # 背景提升到 0.3 左右，避免过度抑制
        }
        
        print(f"[INFO] SINE Dataset 加载完成 (Mask Dropout=0.7, Smoothing=ON)")

    def __len__(self):
        return len(self.sequences_with_ids)

    def __getitem__(self, idx):
        unique_id, raw_sequence = self.sequences_with_ids[idx]
        label = self.labels[idx] if self.is_training else None
        
        sequence = raw_sequence.upper().replace('U', 'T')
        seq_len = len(sequence)
        
        # 1. 序列截断处理 (保持不变) ...
        conservative_base_len = (self.max_token_length - 1) * 6 + 512
        if seq_len > conservative_base_len:
            sequence = sequence[-conservative_base_len:]
            seq_len = len(sequence)
        
        # ==================== 2. 创建 base-level mask (核心修改) ====================
        try:
            motif_coords = self.motif_data.loc[unique_id]
            # 传入 is_training 标志以启用动态权重
            base_mask = self._create_base_level_mask(seq_len, motif_coords)
        except KeyError:
            # 默认背景值也稍微动态化
            bg_val = 0.3
            if self.is_training:
                bg_val = np.random.uniform(*self.weight_ranges['background'])
            base_mask = np.full(seq_len, bg_val, dtype=np.float32)
        
        # 3. 编码 (保持不变) ...
        content_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        if isinstance(content_ids, torch.Tensor):
            content_ids = content_ids.tolist()
        
        max_content_tokens = self.max_token_length - 1
        if len(content_ids) > max_content_tokens:
            content_ids = content_ids[-max_content_tokens:]
        
        # 4. Token 映射 (保持不变) ...
        mapping = []
        pos = 0
        for tid in content_ids:
            tok_str = self.tokenizer.decode([tid])
            tok_len = len(tok_str)
            if tok_str == '<unk>': tok_len = 1
            mapping.append((pos, pos + tok_len))
            pos += tok_len
        
        skipped_bases = seq_len - pos
        if skipped_bases > 0:
            base_mask = base_mask[skipped_bases:]
        
        if len(base_mask) != pos:
            if len(base_mask) > pos:
                base_mask = base_mask[-pos:]
            else:
                pad = np.full(pos - len(base_mask), 0.1, dtype=np.float32)
                base_mask = np.concatenate([pad, base_mask])
        
        # 5. 构建 input_ids (保持不变) ...
        input_ids = torch.full((self.max_token_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_token_length, dtype=torch.long)
        input_ids[0] = self.cls_token_id
        attention_mask[0] = 1
        content_len = len(content_ids)
        input_ids[1:1+content_len] = torch.tensor(content_ids, dtype=torch.long)
        attention_mask[1:1+content_len] = 1
        
        # 6. 构建 token_mask (使用 max pooling 从 base 映射到 token)
        token_mask = torch.full((self.max_token_length,), 0.0, dtype=torch.float32)
        token_mask[0] = float(base_mask.mean()) if len(base_mask) > 0 else 0.3
        
        for i, (start, end) in enumerate(mapping):
            token_idx = i + 1
            segment = base_mask[start:end]
            if len(segment) > 0:
                token_mask[token_idx] = float(np.max(segment))
            else:
                token_mask[token_idx] = 0.3

        # ==================== 7. Mask Dropout (修改为 0.7) ====================
        if self.is_training:
            rand_val = np.random.rand()
            
            # [修改点] 将概率阈值提高到 0.7
            # 70% 的概率抹平 Mask，迫使模型看序列
            if rand_val < 0.7:
                # 使用当前的背景值范围均值作为“平坦”值
                flat_val = sum(self.weight_ranges['background']) / 2
                
                # 获取非 padding 区域
                is_padding = (input_ids == self.pad_token_id)
                token_mask.fill_(flat_val)
                token_mask[is_padding] = 0.0
                
            # 另外 30% 的概率，保留 Mask (但已经在 _create_base_level_mask 里加入了抖动和平滑)
            # 这里可以额外再加一点点高斯噪声，防止过拟合精确的浮点数
            else:
                noise = torch.randn_like(token_mask) * 0.1
                token_mask = (token_mask + noise).clamp(min=0.1, max=3.0)
                is_padding = (input_ids == self.pad_token_id)
                token_mask[is_padding] = 0.0

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motif_mask': token_mask,
            'unique_id': unique_id
        }
        
        if self.is_training:
            result['label'] = torch.tensor(label, dtype=torch.long)
        
        return result

    def _create_base_level_mask(self, seq_len: int, motif_coords: pd.Series) -> np.ndarray:
        """
        创建带有动态权重和高斯平滑的 Mask
        """
        # 1. 确定背景值
        if self.is_training:
            bg_val = np.random.uniform(*self.weight_ranges['background'])
        else:
            bg_val = sum(self.weight_ranges['background']) / 2
            
        mask = np.full(seq_len, bg_val, dtype=np.float32)
        
        # 2. 确定 Motif 权重
        original_sine_start = int(motif_coords.get('original_sine_start_rel', 0))
        
        feature_keys = ['A_box', 'B_box', 'polyA', 'left_TSD', 'right_TSD']
        
        for feature in feature_keys:
            start = motif_coords.get(f'{feature}_start', -1)
            end = motif_coords.get(f'{feature}_end', -1)
            
            if pd.isna(start) or start == -1:
                continue
                
            start, end = int(start), int(end)
            
            # 获取权重
            if self.is_training:
                weight = np.random.uniform(*self.weight_ranges[feature])
            else:
                weight = sum(self.weight_ranges[feature]) / 2
            
            # 映射坐标
            rel_start = max(0, start - original_sine_start)
            rel_end = min(seq_len, end - original_sine_start)
            
            if rel_start < rel_end:
                # 使用 maximum 叠加权重 (比如 TSD 和 A-box 重叠时，取大值)
                mask[rel_start:rel_end] = np.maximum(mask[rel_start:rel_end], weight)
        
        # 3. 应用高斯平滑 (Soft Masking)
        # Sigma 值也可以微调：训练时稍微大一点增加模糊度，预测时标准一点
        sigma = 3.0 if self.is_training else 2.0
        mask = gaussian_kernel_smooth(mask, sigma=sigma)
        
        return mask


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    motif_mask = torch.stack([item['motif_mask'] for item in batch])
    unique_ids = [item['unique_id'] for item in batch]
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'motif_mask': motif_mask,
        'unique_ids': unique_ids
    }
    
    if 'label' in batch[0]:
        result['label'] = torch.stack([item['label'] for item in batch])
    
    return result



def print_batch_info(batch):
    """打印batch信息用于调试"""
    print("\nBatch信息:")
    print(f"  Batch size: {batch['input_ids'].size(0)}")
    print(f"  Sequence length: {batch['input_ids'].size(1)}")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  motif_mask shape: {batch['motif_mask'].shape}")
    
    if 'label' in batch:
        print(f"  labels: {batch['label']}")
        print(f"  SINE比例: {batch['label'].float().mean().item():.2%}")


def visualize_token_alignment(dataset, idx=0):
    """
    可视化token与motif mask的对齐情况
    """
    sample = dataset[idx]
    tokenizer = dataset.tokenizer
    
    input_ids = sample['input_ids']
    motif_mask = sample['motif_mask']
    
    print("\n" + "="*80)
    print("Token-Motif Alignment Visualization")
    print("="*80)
    
    print(f"\nSample ID: {sample.get('unique_id', 'N/A')}")
    print(f"Total tokens: {len(input_ids)}")
    print(f"High-weight tokens (>1.5): {(motif_mask > 1.5).sum()}")
    print(f"Mid-weight tokens (0.8-1.5): {((motif_mask > 0.8) & (motif_mask <= 1.5)).sum()}")
    
    print("\nToken details (前30个):")
    print(f"{'Pos':<5} {'Token':<15} {'Mask':<12} {'Type'}")
    print("-" * 80)
    
    for i, (token_id, mask_val) in enumerate(zip(input_ids, motif_mask)):
        token_str = tokenizer.decode([token_id])
        
        # 分类权重
        if mask_val > 1.8:
            comment = "🔴 A/B-box"
        elif mask_val > 1.3:
            comment = "🟡 PolyA"
        elif mask_val > 0.8:
            comment = "🟢 TSD"
        elif mask_val > 0.5:
            comment = "⚪ Background"
        else:
            comment = "⚫ Padding"
        
        print(f"{i:<5} {token_str:<15} {mask_val:<12.2f} {comment}")
        
        if i >= 29:
            print("... (truncated)")
            break
    
    print("="*80 + "\n")

