#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py (Segmentation Ready)
============================
适配序列标注任务：
1. 同时加载 masks.pt (用于 Attention 引导) 和 motif_pos.tsv (用于生成分割标签)
2. 输出 token_labels: (Batch, Seq_Len) 
   0=Background, 1=TSD, 2=Body, 3=PolyA
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
        sequences_with_ids: list, 
        labels: list,             
        mask_path: str,           
        motif_tsv_path: str,      # [新增] 需要传入 TSV 路径以获取精确坐标
        tokenizer,
        max_token_length: int = 1024,
        is_training: bool = False
    ):
        self.sequences_with_ids = sequences_with_ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.is_training = is_training
        
        # 1. 加载 Mask 字典 (Float, 用于 Attention 引导)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[INFO] Loading float masks from {mask_path}...")
        self.masks_dict = torch.load(mask_path, weights_only=False)
        
        # 2. [新增] 加载 Motif 坐标数据 (用于生成分割标签)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[INFO] Loading coordinates from {motif_tsv_path}...")
        
        # 读取 TSV 并设索引，方便快速查找
        df = pd.read_csv(motif_tsv_path, sep='\t')
        # 确保 unique_id 存在且唯一
        if 'unique_id' not in df.columns:
             # 如果 TSV 里没 unique_id，尝试构建 (需与之前脚本逻辑一致)
             # 这里假设你的 TSV 已经是标准格式
             pass 
        self.motif_df = df.set_index('unique_id')

        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_dropratio = 0.3 if is_training else 0.0

    def __len__(self):
        return len(self.sequences_with_ids)
        
    def _create_token_labels(self, seq_len, motif_coords):
        """
        生成 Base 级标签: 0=Bg, 1=TSD, 2=Body, 3=PolyA
        """
        # 默认全为 0 (Background)
        # 使用 int64 以兼容 CrossEntropyLoss
        labels = np.zeros(seq_len, dtype=np.int64) 
        
        # 如果坐标不存在，直接返回全0
        if motif_coords is None:
            return labels

        def fill(key_start, key_end, val):
            s = motif_coords.get(key_start, -1)
            e = motif_coords.get(key_end, -1)
            
            # 检查无效值 (NaN 或 -1)
            if pd.isna(s) or pd.isna(e) or s == -1 or e == -1:
                return

            s, e = int(s), int(e)
            
            # 注意：04_build_masks.py 生成的 masks.pt 是基于拼接序列的
            # 这里的 motif_coords 如果来自 03_detect_motifs.py，
            # 其中的 'start'/'end' 通常已经是相对于拼接序列的 (如果使用了 new_sine_start 等)
            # 或者如果是原始 TSV，可能需要 offsets。
            # 假设你使用的是 03_detect_motifs_parallel.py 输出的最终 TSV，
            # 它的 left_TSD_start 等字段已经是拼接序列上的绝对坐标。
            # 所以直接使用，不需要减 offset。
            
            s, e = max(0, s), min(seq_len, e)
            if s < e:
                labels[s:e] = val

        # 填充优先级：先画大的，再画小的覆盖
        # 1. 填充 Body (从 A-box Start 到 PolyA Start)
        #    如果没有具体 Body 坐标，可以用 A_box_start 到 PolyA_start 近似
        #    或者简单的：Body = 2
        
        # 这里我们假设 Body 区域覆盖了关键 Motif 之间的区域
        # 为了简单，先填 Motif
        
        # 定义：Body (2)
        # 如果你有 explicit body coordinates 最好，如果没有，
        # 可以粗略认为从 Left TSD End 到 Right TSD Start 都是 Body
        body_start = motif_coords.get('left_TSD_end', -1)
        body_end = motif_coords.get('right_TSD_start', -1)
        if body_start != -1 and body_end != -1:
            fill('left_TSD_end', 'right_TSD_start', 2) # SINE Body
        else:
            # 兜底：如果没 TSD，用 A-box 到 PolyA
            fill('A_box_start', 'polyA_start', 2)

        # 2. 填充具体 Motif (覆盖 Body)
        fill('polyA_start', 'polyA_end', 3) # PolyA (优先级高)
        fill('left_TSD_start', 'left_TSD_end', 1) # TSD
        fill('right_TSD_start', 'right_TSD_end', 1) # TSD
                
        return labels

    def __getitem__(self, idx):
        unique_id, sequence = self.sequences_with_ids[idx]
        
        # 1. 获取 Base Mask (Float)
        if unique_id in self.masks_dict:
            base_mask = self.masks_dict[unique_id]
        else:
            base_mask = np.full(len(sequence), 0.3, dtype=np.float32)

        # 2. [新增] 获取 Base Labels (Int)
        try:
            motif_coords = self.motif_df.loc[unique_id]
        except KeyError:
            motif_coords = None
        
        base_labels = self._create_token_labels(len(sequence), motif_coords)

        # 长度对齐截断
        min_len = min(len(base_mask), len(sequence), len(base_labels))
        base_mask = base_mask[:min_len]
        base_labels = base_labels[:min_len]
        sequence = sequence[:min_len]

        # 3. 动态增强 (仅训练时，只针对 Float Mask，不改 Label)
        if self.is_training:
            noise = np.random.normal(0, 0.05, size=base_mask.shape)
            base_mask = np.clip(base_mask + noise, 0.1, 3.0)
            base_mask = gaussian_kernel_smooth(base_mask, sigma=2.0)

        # 4. Tokenization
        content_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        max_content_tokens = self.max_token_length - 1
        if len(content_ids) > max_content_tokens:
            content_ids = content_ids[-max_content_tokens:]
            
        # 5. Token 对齐
        mapping = []
        pos = 0
        for tid in content_ids:
            tok_str = self.tokenizer.decode([tid])
            tok_len = max(1, len(tok_str))
            mapping.append((pos, pos + tok_len))
            pos += tok_len
        
        # 截断 Mask 和 Labels
        if pos < len(base_mask):
            base_mask = base_mask[:pos]
            base_labels = base_labels[:pos]
        elif pos > len(base_mask):
            pad_len = pos - len(base_mask)
            base_mask = np.concatenate([base_mask, np.full(pad_len, 0.3, dtype=np.float32)])
            base_labels = np.concatenate([base_labels, np.zeros(pad_len, dtype=np.int64)])

        # 6. 构建 Tensors
        input_ids = torch.full((self.max_token_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_token_length, dtype=torch.long)
        
        token_mask = torch.zeros(self.max_token_length, dtype=torch.float32)
        # [新增] Token Labels, 默认填充 4 (Padding/Ignore)
        token_labels = torch.full((self.max_token_length,), 4, dtype=torch.long)
        
        # CLS
        input_ids[0] = self.cls_token_id
        attention_mask[0] = 1
        token_mask[0] = float(base_mask.mean()) if len(base_mask) > 0 else 0.3
        token_labels[0] = 4 # CLS 不参与计算，或设为 0(Bg)
        
        # Content
        valid_len = len(content_ids)
        input_ids[1 : 1+valid_len] = torch.tensor(content_ids, dtype=torch.long)
        attention_mask[1 : 1+valid_len] = 1
        
        # Max Pooling 对齐
        for i, (start, end) in enumerate(mapping):
            token_idx = i + 1
            
            # Float Mask (Feature) -> Max
            seg_mask = base_mask[start:end]
            token_mask[token_idx] = float(np.max(seg_mask)) if len(seg_mask) > 0 else 0.3
            
            # Int Label (Target) -> Max (优先保留高等级标签 PolyA/Body)
            # 0=Bg, 1=TSD, 2=Body, 3=PolyA
            # Max Pooling 会让包含部分 PolyA 的 token 被标记为 PolyA，这对于边界检测是合理的
            seg_label = base_labels[start:end]
            if len(seg_label) > 0:
                token_labels[token_idx] = int(np.max(seg_label))
            else:
                token_labels[token_idx] = 0

        # Mask Dropout
        if self.is_training and np.random.rand() < self.mask_dropratio:
            is_padding = (input_ids == self.pad_token_id)
            token_mask.fill_(0.3)
            token_mask[is_padding] = 0.0

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motif_mask': token_mask,
            'token_labels': token_labels, # [新增] 用于计算 Segmentation Loss
            'unique_id': unique_id
        }
        
        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return result

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    motif_mask = torch.stack([item['motif_mask'] for item in batch])
    token_labels = torch.stack([item['token_labels'] for item in batch]) # [新增]
    unique_ids = [item['unique_id'] for item in batch]
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'motif_mask': motif_mask,
        'token_labels': token_labels,
        'unique_ids': unique_ids
    }
    
    if 'label' in batch[0]:
        result['label'] = torch.stack([item['label'] for item in batch])
    
    return result