#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py (OBIE Positional Labeling)
==================================
基于固定侧翼长度 (Positional Prior) 生成 OBIE 标签：
0: Outside (O)
1: Begin (B)   -> Core Start
2: Inside (I)  -> Core Body
3: End (E)     -> Core End
-100: Ignore
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def gaussian_kernel_smooth(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if sigma <= 0: return mask
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return np.convolve(mask, kernel, mode='same')

class SINEDatasetE2E(Dataset):
    def __init__(
        self,
        samples: list,            # 改名: List[Dict] {'uid', 'seq', 'label', 'core_start', 'core_end'}
        mask_path: str,           
        tokenizer,
        max_token_length: int = 1024,
        is_training: bool = False
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.is_training = is_training
        
        # 仍然加载 Mask 用于 Attention 引导 (可选)
        self.masks_dict = {}
        if mask_path and os.path.exists(mask_path):
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[INFO] Loading float masks from {mask_path}...")
            self.masks_dict = torch.load(mask_path, weights_only=False)
        
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_dropratio = 0.3 if is_training else 0.0

    def __len__(self):
        return len(self.samples)
        
    def _create_token_labels(self, seq_len, core_start, core_end, is_positive):
        """
        生成 OBIE 标签 (4分类)
        """
        labels = np.zeros(seq_len, dtype=np.int64) # Default 0 (Outside)
        
        # 只有正样本才打 B/I/E 标签
        # 负样本 (is_positive=False) 保持全 0 (Outside)
        if is_positive:
            # 确保坐标在合理范围内
            s = max(0, core_start)
            e = min(seq_len, core_end)
            
            if s < e:
                # 1. 填充 Inside (2)
                labels[s:e] = 2
                
                # 2. 标记 Begin (1) - SINE 的第一个碱基
                if s < seq_len:
                    labels[s] = 1
                
                # 3. 标记 End (3) - SINE 的最后一个碱基 (e-1)
                if e-1 >= 0 and e-1 < seq_len:
                    labels[e-1] = 3
                    
        return labels

    def __getitem__(self, idx):
        item = self.samples[idx]
        unique_id = item['uid']
        sequence = item['seq']
        label = item['label']
        
        # 获取预先计算好的核心区坐标
        core_start = item.get('core_start', -1)
        core_end = item.get('core_end', -1)
        is_positive = (label == 1)

        # 1. 获取 Mask (Float)
        if unique_id in self.masks_dict:
            base_mask = self.masks_dict[unique_id]
        else:
            base_mask = np.full(len(sequence), 0.3, dtype=np.float32)

        # 2. 生成 OBIE Labels (Int)
        base_labels = self._create_token_labels(len(sequence), core_start, core_end, is_positive)

        # 3. 长度截断
        min_len = min(len(base_mask), len(sequence), len(base_labels))
        base_mask = base_mask[:min_len]
        base_labels = base_labels[:min_len]
        sequence = sequence[:min_len]

        # 4. 增强
        if self.is_training:
            noise = np.random.normal(0, 0.05, size=base_mask.shape)
            base_mask = np.clip(base_mask + noise, 0.1, 3.0)
            base_mask = gaussian_kernel_smooth(base_mask, sigma=2.0)

        # 5. Tokenization
        content_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        max_content_tokens = self.max_token_length - 1
        if len(content_ids) > max_content_tokens:
            content_ids = content_ids[-max_content_tokens:]
            
        # 6. Alignment
        mapping = []
        pos = 0
        for tid in content_ids:
            tok_str = self.tokenizer.decode([tid])
            tok_len = max(1, len(tok_str))
            mapping.append((pos, pos + tok_len))
            pos += tok_len
        
        if pos < len(base_mask):
            base_mask = base_mask[:pos]
            base_labels = base_labels[:pos]
        elif pos > len(base_mask):
            pad_len = pos - len(base_mask)
            base_mask = np.concatenate([base_mask, np.full(pad_len, 0.3, dtype=np.float32)])
            base_labels = np.concatenate([base_labels, np.full(pad_len, -100, dtype=np.int64)])

        # 7. Tensors
        input_ids = torch.full((self.max_token_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_token_length, dtype=torch.long)
        token_mask = torch.zeros(self.max_token_length, dtype=torch.float32)
        token_labels = torch.full((self.max_token_length,), -100, dtype=torch.long)
        
        input_ids[0] = self.cls_token_id
        attention_mask[0] = 1
        token_mask[0] = float(base_mask.mean()) if len(base_mask) > 0 else 0.3
        token_labels[0] = -100
        
        valid_len = len(content_ids)
        input_ids[1 : 1+valid_len] = torch.tensor(content_ids, dtype=torch.long)
        attention_mask[1 : 1+valid_len] = 1
        
        for i, (start, end) in enumerate(mapping):
            token_idx = i + 1
            seg_mask = base_mask[start:end]
            token_mask[token_idx] = float(np.max(seg_mask)) if len(seg_mask) > 0 else 0.3
            
            # Label Mapping Strategy for OBIE
            seg_label = base_labels[start:end]
            if len(seg_label) > 0:
                if np.any(seg_label == -100):
                    token_labels[token_idx] = -100
                else:
                    # 优先级：B(1) > E(3) > I(2) > O(0)
                    # 我们希望尽可能捕捉边界
                    if 1 in seg_label: token_labels[token_idx] = 1
                    elif 3 in seg_label: token_labels[token_idx] = 3
                    elif 2 in seg_label: token_labels[token_idx] = 2
                    else: token_labels[token_idx] = 0
            else:
                token_labels[token_idx] = -100

        if self.is_training and np.random.rand() < self.mask_dropratio:
            is_padding = (input_ids == self.pad_token_id)
            token_mask.fill_(0.3)
            token_mask[is_padding] = 0.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motif_mask': token_mask,
            'token_labels': token_labels,
            'unique_id': unique_id,
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    motif_mask = torch.stack([item['motif_mask'] for item in batch])
    token_labels = torch.stack([item['token_labels'] for item in batch])
    unique_ids = [item['unique_id'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'motif_mask': motif_mask,
        'token_labels': token_labels,
        'unique_ids': unique_ids,
        'label': labels
    }