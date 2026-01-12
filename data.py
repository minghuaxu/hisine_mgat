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
        
        # 1. 原始坐标
        core_start = item.get('core_start', -1)
        core_end = item.get('core_end', -1)
        is_positive = (label == 1)
        
        # -------------------------------------------------------------------------
        # [Step 1] 随机裁剪 (安全版)
        # -------------------------------------------------------------------------
        target_char_len = self.max_token_length - 5 
        seq_len = len(sequence)
        
        start_idx = 0
        end_idx = seq_len

        if self.is_training and seq_len > target_char_len:
            if is_positive:
                # 必须包含 SINE 的一部分
                safe_min = max(0, core_end - target_char_len + 10)
                safe_max = max(0, min(seq_len - target_char_len, core_start - 10))
                
                if safe_min < safe_max:
                    start_idx = np.random.randint(safe_min, safe_max)
                else:
                    start_idx = np.random.randint(0, seq_len - target_char_len)
            else:
                start_idx = np.random.randint(0, seq_len - target_char_len)
            
            end_idx = start_idx + target_char_len

        # 执行裁剪
        sequence_cropped = sequence[start_idx:end_idx]
        
        # Mask 裁剪
        base_mask_cropped = None
        if unique_id in self.masks_dict:
            full_mask = self.masks_dict[unique_id]
            if len(full_mask) >= end_idx:
                base_mask_cropped = full_mask[start_idx:end_idx]
            else:
                # 补齐长度不足的 mask
                pad_len = seq_len - len(full_mask)
                if pad_len > 0:
                    full_mask = np.concatenate([full_mask, np.full(pad_len, 0.3, dtype=np.float32)])
                base_mask_cropped = full_mask[start_idx:end_idx]
        
        if base_mask_cropped is None:
            base_mask_cropped = np.full(len(sequence_cropped), 0.3, dtype=np.float32)

        # 更新相对坐标
        new_core_start = (core_start - start_idx) if is_positive else -1
        new_core_end = (core_end - start_idx) if is_positive else -1

        # 生成字符级标签
        base_labels_cropped = self._create_token_labels(
            len(sequence_cropped), new_core_start, new_core_end, is_positive
        )

        # -------------------------------------------------------------------------
        # [Step 2] Tokenization (不请求 offsets_mapping)
        # -------------------------------------------------------------------------
        encoding = self.tokenizer(
            sequence_cropped,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        # -------------------------------------------------------------------------
        # [Step 3] 手动计算 Offsets (适配 NucleotidesKmersTokenizer)
        # -------------------------------------------------------------------------
        # 获取特殊 token 的 ID 集合 (用于跳过计算)
        special_token_ids = {
            self.tokenizer.bos_token_id, 
            self.tokenizer.eos_token_id, 
            self.tokenizer.cls_token_id, 
            self.tokenizer.pad_token_id, 
            self.tokenizer.mask_token_id,
            self.tokenizer.unk_token_id
        }
        # 如果 tokenizer 还有其他 special tokens，也加入
        if hasattr(self.tokenizer, 'all_special_ids'):
            special_token_ids.update(self.tokenizer.all_special_ids)

        # 将 tensor 转回 token 列表
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        manual_offsets = []
        current_pos = 0
        
        for i, token in enumerate(tokens):
            # 1. 跳过 Padding (通过 attention_mask 判断最准)
            if attention_mask[i] == 0:
                manual_offsets.append((0, 0))
                continue

            # 2. 处理特殊 Token (CLS, BOS, EOS 等)
            # 它们的字符长度不计入 sequence 进度
            token_id = input_ids[i].item()
            if token_id in special_token_ids:
                manual_offsets.append((0, 0))
                continue
            
            # 3. 处理普通 K-mer Token
            # NucleotidesKmersTokenizer 的 token 就是原始序列的子串 (例如 "ATC", "G", "N")
            # 不会有 WordPiece 的 "##" 或 BPE 的 "Ġ" 前缀，所以直接取 len 即可
            token_len = len(token)
            
            # 记录区间 [start, end)
            start = current_pos
            end = current_pos + token_len
            
            # 简单的边界保护 (防止 token 长度溢出裁剪后的序列)
            if end > len(sequence_cropped):
                end = len(sequence_cropped)
                
            manual_offsets.append((start, end))
            
            # 推进光标
            current_pos += token_len

        # -------------------------------------------------------------------------
        # [Step 4] 映射 Labels 和 Mask (使用手动计算的 offsets)
        # -------------------------------------------------------------------------
        motif_mask = torch.zeros(self.max_token_length, dtype=torch.float32)
        token_labels = torch.full((self.max_token_length,), -100, dtype=torch.long)
        
        # 将所有“有效 Token 位置”预设为 0
        # 这样可以保证只要 attention_mask[i]==1，token_labels[i] 就不是 -100
        # 同时也解决了 CLS (index 0) 的问题
        valid_indices = (attention_mask == 1)
        token_labels[valid_indices] = 0

        actual_seq_len = len(sequence_cropped)
        
        for idx, (s, e) in enumerate(manual_offsets):
            # 忽略 (0,0) 的 Offset (特殊 token 或 padding)
            if s == e:
                continue

            if s >= actual_seq_len: 
                continue
            
            # 映射 Mask
            seg_mask = base_mask_cropped[s:e]
            motif_mask[idx] = float(np.max(seg_mask)) if len(seg_mask) > 0 else 0.3
            
            # 映射 Labels
            seg_label = base_labels_cropped[s:e]
            if len(seg_label) > 0:
                if 1 in seg_label: token_labels[idx] = 1   # B
                elif 3 in seg_label: token_labels[idx] = 3 # E
                elif 2 in seg_label: token_labels[idx] = 2 # I
                else: token_labels[idx] = 0                # O
            else:
                token_labels[idx] = 0 # Outside

        # -------------------------------------------------------------------------
        # [Step 5] Mask 增强
        # -------------------------------------------------------------------------
        if self.is_training:
            if np.random.rand() < self.mask_dropratio:
                valid_pos = (attention_mask == 1)
                motif_mask[valid_pos] = 0.3
            else:
                # 可选：加一点高斯噪声
                pass

        if self.is_training and item['label'] == 1 and np.random.rand() < 0.3:  # 30% 正样本模拟退化
            # 随机 drop 部分 motif 区域
            motif_mask = motif_mask * torch.bernoulli(torch.full_like(motif_mask, 0.5)).float()
            # 或直接降级
            motif_mask = motif_mask.clamp(max=1.0)  # 模拟模糊 motif

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motif_mask': motif_mask,
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