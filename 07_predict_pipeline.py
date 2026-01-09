#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_predict_pipeline.py (Robust OBIE Segmentation)
================================================
适配训练代码 train_e2e_classifier.py
1. 适配 OBIE 标签体系 (0=Outside, 1=Begin, 2=Inside, 3=End)。
2. 引入【鲁棒解码算法】：
   - 前景概率聚合 (Sum Prob of B, I, E)
   - 概率平滑 (Gaussian Smoothing)
   - 空洞填充 (Gap Filling)
   - 长度过滤 (Min Length Filtering)
"""

import argparse
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf, edgeitems=10)

import torch
import subprocess
import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path

# 引入项目模块 (确保路径正确，指向包含 sine_classifier 的父目录)
sys.path.insert(0, str(Path(__file__).parent))
try:
    # 优先尝试训练代码的导入路径
    from sine_classifier.model import MotifGuidedSINEClassifier
except ImportError:
    # 兼容旧路径
    from model import MotifGuidedSINEClassifier

# ================= 配置 =================
FLANK_LEN = 150 
BATCH_SIZE = 16 # 推理时稍微调大 Batch Size 以加速

def gaussian_smooth(x, window_size=5):
    """简单的一维高斯平滑"""
    if window_size < 2: return x
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode='same')

def fill_binary_gaps(mask, max_gap=10):
    """
    填充二值掩码中的小空洞。
    例如: 1 1 0 0 1 1 -> 1 1 1 1 1 1 (如果空洞长度 <= max_gap)
    """
    n = len(mask)
    if n == 0: return mask
    
    indices = np.where(mask)[0]
    if len(indices) < 2: return mask
    
    new_mask = mask.copy()
    for i in range(len(indices) - 1):
        curr = indices[i]
        next_ = indices[i+1]
        gap = next_ - curr - 1
        if 0 < gap <= max_gap:
            new_mask[curr+1 : next_] = True
            
    return new_mask

def build_mask_for_sample(seq_len, motif_row):
    WEIGHTS = {'background': 0.3, 'A_box': 2.0, 'B_box': 2.0, 'polyA': 1.6, 'TSD': 1.0}
    mask = np.full(seq_len, WEIGHTS['background'], dtype=np.float32)
    def fill(k_start, k_end, w):
        s, e = motif_row.get(k_start, -1), motif_row.get(k_end, -1)
        if s != -1 and e != -1:
            s, e = int(s), int(e)
            s, e = max(0, s), min(seq_len, e)
            if s < e: mask[s:e] = np.maximum(mask[s:e], w)
    fill('left_TSD_start', 'left_TSD_end', WEIGHTS['TSD'])
    fill('right_TSD_start', 'right_TSD_end', WEIGHTS['TSD'])
    fill('polyA_start', 'polyA_end', WEIGHTS['polyA'])
    fill('A_box_start', 'A_box_end', WEIGHTS['A_box'])
    fill('B_box_start', 'B_box_end', WEIGHTS['B_box'])
    return mask.astype(np.float32)

class InferenceDataset(Dataset):
    def __init__(self, data_list, motif_df, tokenizer, max_len=256): 
        self.data = data_list
        self.motif_df = motif_df.set_index('unique_id')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls_id = tokenizer.cls_token_id
        self.pad_id = tokenizer.pad_token_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        uid, seq = self.data[idx]
        if uid in self.motif_df.index:
            row = self.motif_df.loc[uid]
            base_mask = build_mask_for_sample(len(seq), row)
        else:
            base_mask = np.full(len(seq), 0.3, dtype=np.float32)
            
        content_ids = self.tokenizer.encode(seq, add_special_tokens=False)
        if len(content_ids) > self.max_len - 1:
            content_ids = content_ids[:self.max_len - 1]
            
        mapping = []
        pos = 0
        for tid in content_ids:
            tok_len = max(1, len(self.tokenizer.decode([tid])))
            mapping.append((pos, pos + tok_len))
            pos += tok_len
            
        if pos < len(base_mask): base_mask = base_mask[:pos]
        elif pos > len(base_mask): 
            base_mask = np.concatenate([base_mask, np.full(pos-len(base_mask), 0.3)])
            
        input_ids = torch.full((self.max_len,), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        token_mask = torch.zeros(self.max_len, dtype=torch.float32)
        padded_mapping = torch.zeros((self.max_len, 2), dtype=torch.long)
        
        input_ids[0] = self.cls_id
        attention_mask[0] = 1
        token_mask[0] = float(base_mask.mean()) if len(base_mask)>0 else 0.3
        
        valid_len = len(content_ids)
        input_ids[1 : 1+valid_len] = torch.tensor(content_ids, dtype=torch.long)
        attention_mask[1 : 1+valid_len] = 1
        
        for i, (s, e) in enumerate(mapping):
            token_idx = i + 1
            seg = base_mask[s:e]
            token_mask[token_idx] = float(np.max(seg)) if len(seg)>0 else 0.3
            padded_mapping[token_idx] = torch.tensor([s, e])
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motif_mask': token_mask,
            'offset_mapping': padded_mapping,
            'raw_sequence': seq,
            'unique_id': uid
        }

def load_genome(fasta_path):
    print(f"Loading genome: {fasta_path} ...")
    return SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

def extract_sequences(bed_path, genome_dict, temp_csv_path):
    print("Extracting sequences...")
    data = []
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            label = parts[3] if len(parts) > 3 else 'Unknown'
            strand = parts[5] if len(parts) > 5 else '+'
            
            if chrom not in genome_dict: continue
            full_chrom_seq = genome_dict[chrom].seq
            
            # Extract with flank
            l_start = max(0, start - FLANK_LEN)
            r_end = min(len(full_chrom_seq), end + FLANK_LEN)
            
            core_seq = str(full_chrom_seq[start:end])
            left_seq = str(full_chrom_seq[l_start:start])
            right_seq = str(full_chrom_seq[end:r_end])
            
            data.append({
                'chrom': chrom, 'start': start, 'end': end, 'strand': strand,
                'seq': core_seq, 'flank_left': left_seq, 'flank_right': right_seq,
                'label': label, 'source_type': 'Simulated'
            })
    df = pd.DataFrame(data)
    df.to_csv(temp_csv_path, index=False)
    return df

def run_motif_detection(in_csv, out_tsv):
    print("Running motif detection...")
    # 确保调用的是正确的 motif 检测脚本路径
    cmd = ["python", "05_detect_motifs_parallel.py", "--in_csv", in_csv, "--out_tsv", out_tsv, "--threads", "16"]
    subprocess.run(cmd, check=True)

def decode_with_peak_detection(token_logits, offset_mapping, raw_seq, threshold=0.25):
    """
    基于峰值检测的解码器
    核心思想：B/E 是稀疏信号，应该寻找局部最大值而非全局阈值
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    probs = torch.softmax(token_logits, dim=-1).cpu().numpy()

    for i, prob in enumerate(probs):
        print(f"位置 {i}: [{prob[0]:.2f}, {prob[1]:.2f}, {prob[2]:.2f}, {prob[3]:.2f}]")
    
    # 排除 PAD
    valid_mask = (offset_mapping.cpu().numpy().sum(axis=1) > 0)
    valid_len = valid_mask.sum()
    if valid_len == 0:
        return "", -1, -1
    
    probs = probs[:valid_len]
    
    # ===== Step 1: 找 Inside 核心区域 =====
    inside_score = probs[:, 2]
    inside_smooth = gaussian_filter1d(inside_score, sigma=1.5)
    
    # Inside 峰值检测
    inside_peaks, _ = find_peaks(inside_smooth, height=threshold, distance=3)
    
    if len(inside_peaks) == 0:
        return "", -1, -1
    
    # 找最高峰
    highest_peak = inside_peaks[np.argmax(inside_smooth[inside_peaks])]
    
    # 从峰值向两边扩展
    core_start = highest_peak
    core_end = highest_peak
    
    # 向左扩展（直到 Inside < threshold）
    for i in range(highest_peak - 1, -1, -1):
        if inside_smooth[i] < threshold:
            break
        core_start = i
    
    # 向右扩展
    for i in range(highest_peak + 1, valid_len):
        if inside_smooth[i] < threshold:
            break
        core_end = i
    
    # ===== Step 2: 在核心区域附近找 Begin =====
    # 搜索范围：core_start 前 20 tokens
    search_b_start = max(0, core_start - 20)
    search_b_end = core_start + 3
    
    begin_scores = probs[search_b_start:search_b_end, 1]
    
    # ✅ 关键：找 Begin 的局部最大值（而非全局阈值）
    if len(begin_scores) > 0:
        # 平滑（轻度）
        begin_smooth = gaussian_filter1d(begin_scores, sigma=1.0)
        
        # 找峰值
        b_peaks, properties = find_peaks(
            begin_smooth, 
            height=0.05,  # 最低阈值 5%
            prominence=0.03  # 要求显著性（比周围高 3%）
        )
        
        if len(b_peaks) > 0:
            # 选最接近 core_start 的峰值
            closest_b_peak = b_peaks[np.argmin(np.abs(b_peaks - (core_start - search_b_start)))]
            final_start = search_b_start + closest_b_peak
        else:
            # 如果没找到峰值，用最大值位置
            max_b_idx = np.argmax(begin_scores)
            if begin_scores[max_b_idx] > 0.05:
                final_start = search_b_start + max_b_idx
            else:
                final_start = core_start  # 退化到 core 边界
    else:
        final_start = core_start
    
    # ===== Step 3: 在核心区域附近找 End =====
    search_e_start = core_end - 3
    search_e_end = min(valid_len, core_end + 20)
    
    end_scores = probs[search_e_start:search_e_end, 3]
    
    if len(end_scores) > 0:
        end_smooth = gaussian_filter1d(end_scores, sigma=1.0)
        
        e_peaks, _ = find_peaks(
            end_smooth,
            height=0.05,
            prominence=0.03
        )
        
        if len(e_peaks) > 0:
            # 选最接近 core_end 的峰值
            closest_e_peak = e_peaks[np.argmin(np.abs(e_peaks - (core_end - search_e_start)))]
            final_end = search_e_start + closest_e_peak + 1
        else:
            max_e_idx = np.argmax(end_scores)
            if end_scores[max_e_idx] > 0.05:
                final_end = search_e_start + max_e_idx + 1
            else:
                final_end = core_end + 1
    else:
        final_end = core_end + 1
    
    # ===== Step 4: 映射回字符坐标 =====
    offsets = offset_mapping.cpu().numpy()
    char_start = offsets[final_start][0]
    char_end = offsets[min(final_end, valid_len-1)][1]
    
    refined_seq = raw_seq[char_start:char_end]
    
    if len(refined_seq) < 30:
        return "", -1, -1
    
    return refined_seq, char_start, char_end

def decode_span_search(token_logits, offset_mapping, raw_seq, min_len=30, max_len=300):
    """
    基于全局打分搜索最佳 SINE 区间
    Score = P(Begin) + P(End) + Avg(P(Inside))
    """

    # 1. 预处理
    # 转为 Log Probability 以便相加 (避免概率乘积下溢)
    log_probs = torch.log_softmax(token_logits, dim=-1)  # (L, 4) -> [O, B, I, E]
    seq_len = log_probs.shape[0]
    
    # 排除 PAD 和 CLS/SEP 的影响 (假设 offset_mapping=0 的是特殊token)
    valid_mask = (offset_mapping.sum(axis=1) > 0)
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0: return "", -1, -1
    
    start_search = valid_indices[0].item()
    end_search = valid_indices[-1].item()

    if end_search <= start_search:
        return "", -1, -1

    # 2. 提取分量
    b_scores = log_probs[:, 1]  # Begin LogProb
    i_scores = log_probs[:, 2]  # Inside LogProb
    e_scores = log_probs[:, 3]  # End LogProb

    search_len = end_search - start_search
    
    # 如果搜索区域太小，直接返回空
    if search_len <= 0:
        return "", -1, -1

    # 确保 k 不超过 search_len
    k_val = min(20, search_len)
    
    # 3. 快速筛选候选点 (避免 O(N^2) 全遍历)
    # 取 B 和 E 概率最高的前 20 个点作为候选
    top_starts = torch.topk(b_scores[start_search:end_search], k=k_val).indices + start_search
    top_ends = torch.topk(e_scores[start_search:end_search], k=k_val).indices + start_search
    
    best_score = -float('inf')
    best_s_idx, best_e_idx = -1, -1

    # 4. 计算 Inside 区域的累积和 (前缀和)，用于O(1)计算区间均值
    # i_cumsum[k] = sum(i_scores[:k])
    i_cumsum = torch.cumsum(i_scores, dim=0)

    # 5. 搜索最佳组合
    for s in top_starts.tolist():
        for e in top_ends.tolist():
            # 逻辑约束
            seg_len = e - s
            if seg_len < 5 or seg_len > max_len: continue  # 长度限制
            
            # 计算 Inside 得分: sum(Inside[s+1:e]) / len
            # 使用前缀和加速
            if e > s + 1:
                curr_i_sum = i_cumsum[e-1] - i_cumsum[s]
                avg_i_score = curr_i_sum / (e - s - 1)
            else:
                avg_i_score = 0
            
            # 总分公式：Begin + End + 权重 * Inside_Avg
            # 权重 2.0 是经验值，强调内部一致性
            score = b_scores[s] + e_scores[e] + 2.0 * avg_i_score
            
            if score > best_score:
                best_score = score
                best_s_idx = s
                best_e_idx = e

    # 6. 映射回字符坐标
    if best_s_idx != -1 and best_e_idx != -1:
        offsets = offset_mapping.cpu().numpy()
        char_start = offsets[best_s_idx][0]
        char_end = offsets[best_e_idx][1] # 注意 End token 的右边界
        
        # 简单校验
        if char_end > char_start and (char_end - char_start) >= min_len:
            refined_seq = raw_seq[char_start:char_end]
            return refined_seq, char_start, char_end

    return "", -1, -1

def decode_two_stage(token_logits, offset_mapping, raw_seq, threshold=0.5, debug=False):
    """
    两阶段解码器：
    Stage 1: 状态机找 Inside 连续段（粗定位）
    Stage 2: 峰值检测精修 Begin/End 边界（细调整）
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    
    probs = torch.softmax(token_logits, dim=-1).cpu().numpy()

    for i, prob in enumerate(probs):
        print(f"位置 {i}: [{prob[0]:.2f}, {prob[1]:.2f}, {prob[2]:.2f}, {prob[3]:.2f}]")
    
    # 排除 PAD/CLS
    valid_mask = (offset_mapping.cpu().numpy().sum(axis=1) > 0)
    valid_len = valid_mask.sum()
    if valid_len == 0:
        return "", -1, -1
    
    probs = probs[:valid_len]
    
    # ========== Stage 1: 找 Inside 连续段 ==========
    sine_score = probs[:, 1] + probs[:, 2] + probs[:, 3]  # B + I + E
    
    # 轻度平滑
    smoothed_score = gaussian_filter1d(sine_score, sigma=1)

    for i, prob in enumerate(smoothed_score):
        print(f"位置 {i}: [{prob:.2f}]")
    
    # 阈值检测
    sine_mask = (smoothed_score > threshold)
    
    if not sine_mask.any():
        threshold_fallback = threshold * 0.7
        sine_mask = (smoothed_score > threshold_fallback)
        if not sine_mask.any():
            return "", -1, -1
    
    # 空洞填充
    def fill_gaps(mask, max_gap=8):
        mask = mask.astype(bool)
        indices = np.where(mask)[0]
        if len(indices) < 2: return mask
        new_mask = mask.copy()
        for i in range(len(indices) - 1):
            curr, next_ = indices[i], indices[i+1]
            gap = next_ - curr - 1
            if 0 < gap <= max_gap:
                new_mask[curr+1:next_] = True
        return new_mask
    
    sine_mask = fill_gaps(sine_mask, max_gap=8)
    
    # 找最长连续段
    indices = np.where(sine_mask)[0]
    if len(indices) == 0:
        return "", -1, -1
    
    diff = np.diff(indices)
    split_locs = np.where(diff > 1)[0] + 1
    segments = np.split(indices, split_locs)
    best_segment = max(segments, key=len)
    
    if len(best_segment) < 5:
        return "", -1, -1
    
    # 粗定位的起止
    rough_start = best_segment[0]
    rough_end = best_segment[-1]
    
    if debug:
        print(f"\n[Stage 1] Rough Range: [{rough_start}, {rough_end}] ({rough_end - rough_start} tokens)")
    
    # ========== Stage 2: 精修边界 ==========
    
    # === 2.1 精修 Begin (左边界) ===
    # 搜索窗口：rough_start 前 15 tokens 到 rough_start 后 5 tokens
    search_b_start = max(0, rough_start - 15)
    search_b_end = min(valid_len, rough_start + 5)
    
    begin_scores = probs[search_b_start:search_b_end, 1]  # Begin 概率
    
    final_start = rough_start  # 默认值
    
    if len(begin_scores) > 3:
        # 平滑
        begin_smooth = gaussian_filter1d(begin_scores, sigma=1.0)
        
        # ✅ 策略 1：找显著峰值（优先）
        b_peaks, properties = find_peaks(
            begin_smooth,
            height=0.05,        # 最低 5%
            prominence=0.03,    # 显著性
            distance=3          # 峰值间距
        )
        
        if len(b_peaks) > 0:
            # 选最接近 rough_start 的峰值
            distances = np.abs(b_peaks - (rough_start - search_b_start))
            closest_peak = b_peaks[np.argmin(distances)]
            final_start = search_b_start + closest_peak
            
            if debug:
                print(f"[Stage 2.1] Found Begin peak at {final_start} (prob={begin_smooth[closest_peak]:.3f})")
        
        # ✅ 策略 2：如果没有峰值，找概率 > 0.1 的最左位置
        else:
            strong_b = np.where(begin_scores > 0.1)[0]
            if len(strong_b) > 0:
                final_start = search_b_start + strong_b[0]
                if debug:
                    print(f"[Stage 2.1] No peak, use threshold: {final_start} (prob={begin_scores[strong_b[0]]:.3f})")
            else:
                # 退化：保持 rough_start
                if debug:
                    print(f"[Stage 2.1] No signal, keep rough start")
    
    # === 2.2 精修 End (右边界) ===
    # 搜索窗口：rough_end 前 5 tokens 到 rough_end 后 20 tokens
    print(rough_end)
    search_e_start = max(0, rough_end - 30)
    search_e_end = min(valid_len, rough_end + 30)
    print(search_e_start)
    print(search_e_end)
    end_scores = probs[search_e_start:search_e_end, 3]  # End 概率
    print(len(end_scores))
    
    final_end = rough_end + 1  # 默认值
    
    if len(end_scores) > 3:
        end_smooth = gaussian_filter1d(end_scores, sigma=1.0)

        for i, item in enumerate(end_smooth):
            print(f'{i} : {item:.2f}')
        
        # ✅ 策略 1：找峰值（优先）
        e_peaks, _ = find_peaks(
            end_smooth,
            height=0.05,
            prominence=0.03,
            distance=3
        )
        print(e_peaks)
        if len(e_peaks) > 0:
            # ⚠️ 关键改进：处理尾部过长问题
            # 如果有多个峰值，选择第一个显著峰值（而非最接近的）
            # 因为尾部往往有噪声峰值
            
            # 过滤掉距离 rough_end 太远的峰值（> 15 tokens）
            print(e_peaks - (rough_end - search_e_start))
            valid_e_peaks = e_peaks[abs(e_peaks - (rough_end - search_e_start)) <=30]
            print(valid_e_peaks)

            print(end_smooth[valid_e_peaks])
            
            if len(valid_e_peaks) > 0:
                # 选最接近 rough_end 的峰值
                # distances = np.abs(valid_e_peaks - (rough_end - search_e_start))
                # closest_peak = valid_e_peaks[np.argmin(distances)]
                closest_peak = valid_e_peaks[np.argmax(end_smooth[valid_e_peaks])]
                print(closest_peak)
                final_end = search_e_start + closest_peak + 1

                print(f'{closest_peak}, {final_end}')
                
                if debug:
                    print(f"[Stage 2.2] Found End peak at {final_end-1} (prob={end_smooth[closest_peak]:.3f})")
            else:
                # 所有峰值都太远，说明可能是噪声
                if debug:
                    print(f"[Stage 2.2] Peaks too far, check threshold")
                # 退化到策略 2
                e_peaks = []
        
        # ✅ 策略 2：找概率 > 0.1 的最右位置
        if len(e_peaks) == 0:
            strong_e = np.where(end_scores > 0.1)[0]
            if len(strong_e) > 0:
                # ⚠️ 关键：如果有多个强信号，选第一个（避免尾部过长）
                # 但如果第一个距离 rough_end 太近（< 3 tokens），选最后一个
                first_strong = strong_e[0]
                last_strong = strong_e[-1]
                
                if first_strong - (rough_end - search_e_start) < 3:
                    # 太近，可能遗漏，选最后一个
                    final_end = search_e_start + last_strong + 1
                    if debug:
                        print(f"[Stage 2.2] First signal too close, use last: {final_end-1}")
                else:
                    # 选第一个（避免过长）
                    final_end = search_e_start + first_strong + 1
                    if debug:
                        print(f"[Stage 2.2] Use first strong signal: {final_end-1} (prob={end_scores[first_strong]:.3f})")
            else:
                # ✅ 策略 3：渐进式衰减检测（处理弱信号）
                # 从 rough_end 开始，找第一个 End 概率显著下降的位置
                baseline = end_scores[0] if len(end_scores) > 0 else 0.01
                
                for i in range(len(end_scores)):
                    # 如果概率下降到 baseline 的 30% 以下，且绝对值 < 0.05
                    if end_scores[i] < baseline * 0.3 and end_scores[i] < 0.05:
                        final_end = search_e_start + i
                        if debug:
                            print(f"[Stage 2.2] Decay cutoff at {final_end-1}")
                        break
                else:
                    # 没找到衰减点，保守截断
                    final_end = min(valid_len, rough_end + 8)
                    if debug:
                        print(f"[Stage 2.2] No clear signal, conservative cutoff")
    
    # ========== Stage 3: 映射回字符坐标 ==========
    offsets = offset_mapping.cpu().numpy()
    char_start = offsets[final_start][0]
    char_end = offsets[min(final_end, valid_len-1)][1]
    
    refined_seq = raw_seq[char_start:char_end]
    
    if len(refined_seq) < 30:
        return "", -1, -1
    
    if debug:
        print(f"[Stage 3] Final Range: [{final_start}, {final_end}] ({final_end - final_start} tokens)")
        print(f"           Char Range: [{char_start}, {char_end}] ({len(refined_seq)} bp)")
    
    return refined_seq, char_start, char_end

def decode_with_state_machine(token_logits, offset_mapping, raw_seq, threshold=0.5, min_len=30):
    """
    基于状态机的 OBIE 解码（改进版）
    
    改进点：
    1. 降低默认阈值 0.4 → 0.3
    2. 减少平滑强度 sigma=2 → sigma=1.5
    3. 允许更大的 gap (5 → 8 tokens)
    4. 更激进的边界扩展
    """
    # 1. 获取概率
    probs = torch.softmax(token_logits, dim=-1).cpu().numpy() 

    for i, prob in enumerate(probs):
        print(f"位置 {i}: [{prob[0]:.2f}, {prob[1]:.2f}, {prob[2]:.2f}, {prob[3]:.2f}]")
    
    # SINE Score = P(Begin) + P(Inside) + P(End)
    sine_score = probs[:, 1] + probs[:, 2] + probs[:, 3]
    
    # 排除 CLS/Pad
    valid_mask = (offset_mapping.cpu().numpy().sum(axis=1) > 0)
    sine_score = sine_score * valid_mask 
    
    # ✅ 修改 1：减少平滑强度（避免过度削弱峰值）
    from scipy.ndimage import gaussian_filter1d
    smoothed_score = gaussian_filter1d(sine_score, sigma=1)  # 原来是 5

    for i, prob in enumerate(smoothed_score):
        print(f"位置 {i}: [{prob:.2f}]")
    
    # ✅ 修改 2：降低阈值
    sine_mask = (smoothed_score > threshold) 
    
    # ✅ 修改 3：如果没找到，尝试更低的阈值
    if not sine_mask.any():
        threshold_fallback = threshold * 0.6  # 0.18
        sine_mask = (smoothed_score > threshold_fallback)
        
        if not sine_mask.any():
            return "", -1, -1
    
    # ✅ 修改 4：允许更大的 gap（8 tokens，约 30-40bp）
    sine_mask = fill_binary_gaps(sine_mask, max_gap=15)  # 原来是 5
    
    # 5. 提取片段
    indices = np.where(sine_mask)[0]
    if len(indices) == 0:
        return "", -1, -1
        
    # 找最长连续段
    diff = np.diff(indices)
    split_locs = np.where(diff > 1)[0] + 1
    segments = np.split(indices, split_locs)
    
    best_segment = max(segments, key=len)
    
    # ✅ 修改 5：降低最小 token 数要求
    if len(best_segment) < 5:  # 原来可能更高
        return "", -1, -1
    
    # 6. 映射回字符坐标
    offsets = offset_mapping.cpu().numpy()
    start_idx = best_segment[0]
    end_idx = best_segment[-1]
    
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    
    refined_seq = raw_seq[char_start:char_end]
    
    # ✅ 修改 6：降低最小长度要求
    if len(refined_seq) < min_len:
        return "", -1, -1
        
    return refined_seq, char_start, char_end


def fill_binary_gaps(mask, max_gap=8):
    """
    填充二值掩码中的小空洞（改进版）
    """
    mask = mask.astype(bool)
    n = len(mask)
    if n == 0: return mask
    
    # 转换为索引列表
    indices = np.where(mask)[0]
    if len(indices) < 2: return mask
    
    new_mask = mask.copy()
    for i in range(len(indices) - 1):
        curr = indices[i]
        next_ = indices[i+1]
        gap = next_ - curr - 1
        
        # ✅ 允许更大的 gap
        if 0 < gap <= max_gap:
            new_mask[curr+1 : next_] = True
            
    return new_mask

def decode_robus(token_logits, offset_mapping, raw_seq, threshold=0.4, min_len=30):
    """
    OBIE 解码 (4分类): 
    Labels: 0=Outside, 1=Begin, 2=Inside, 3=End
    
    策略：
    1. 计算前景概率 SINE Score = P(Begin) + P(Inside) + P(End)。
       我们不严格依赖 B/E 标签作为切割点，因为模型输出可能存在噪声（例如 O-I-I-E 或 B-I-I-O）。
       聚合概率能最大程度保证召回率。
    2. 平滑处理连接断点。
    3. 提取最长连续高置信度片段。
    """
    # 1. 获取概率 (L, 4)
    probs = torch.softmax(token_logits, dim=-1).cpu().numpy() 
    # 以两位小数打印
    for i, prob in enumerate(probs):
        print(f"位置 {i}: [{prob[0]:.2f}, {prob[1]:.2f}, {prob[2]:.2f}, {prob[3]:.2f}]")
    
    # 【适配 OBIE】前景概率 = P(1) + P(2) + P(3)
    # 0 是 Background
    if probs.shape[1] == 4:
        sine_score = probs[:, 1] + probs[:, 2] + probs[:, 3]
    else:
        # Fallback if model shape is different (unlikely with correct init)
        sine_score = probs[:, 1:].sum(axis=1)
    
    # 排除 CLS/Pad (Mapping 为 0 的位置)
    valid_mask = (offset_mapping.cpu().numpy().sum(axis=1) > 0)
    sine_score = sine_score * valid_mask 
    
    # 2. 平滑 (Window=5)
    # 这能把间断的 B...I...I...E 连成一条线，消除单个 token 的抖动
    smoothed_score = gaussian_smooth(sine_score, window_size=5)
    
    # 3. 生成掩码
    sine_mask = (smoothed_score > threshold)
    
    # 4. 空洞填充 (Gap Filling)
    # 允许最大 5 个 token 的断裂 (约 15-20bp) 被连接
    sine_mask = fill_binary_gaps(sine_mask, max_gap=5)
    
    # 5. 提取片段
    indices = np.where(sine_mask)[0]
    if len(indices) == 0:
        return "", -1, -1
        
    # 找最长连续段
    diff = np.diff(indices)
    split_locs = np.where(diff > 1)[0] + 1
    segments = np.split(indices, split_locs)
    
    best_segment = max(segments, key=len)
    
    # 6. 映射回字符坐标
    offsets = offset_mapping.cpu().numpy()
    start_idx = best_segment[0]
    end_idx = best_segment[-1]
    
    # 取 Start Token 的起始字符 和 End Token 的结束字符
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    
    # 边界保护
    char_start = max(0, char_start)
    char_end = min(len(raw_seq), char_end)

    refined_seq = raw_seq[char_start:char_end]
    
    # 7. 长度过滤
    if len(refined_seq) < min_len:
        return "", -1, -1
        
    return refined_seq, char_start, char_end

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True, help="Input BED file")
    parser.add_argument("--fasta", required=True, help="Reference Genome FASTA")
    parser.add_argument("--model_path", required=True, help="Path to best_model.pt")
    parser.add_argument("--backbone", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--out_file", default="predictions.tsv")
    parser.add_argument("--out_fasta", default="predictions_refined.fasta")
    parser.add_argument("--temp_dir", default="temp_pred_workspace")
    args = parser.parse_args()
    
    # 1. 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device( "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.temp_dir, exist_ok=True)
    temp_csv = os.path.join(args.temp_dir, "input_seqs.csv")
    temp_motif = os.path.join(args.temp_dir, "motifs.tsv")
    
    # 2. 数据准备
    genome = load_genome(args.fasta)
    df_seq = extract_sequences(args.bed, genome, temp_csv)
    
    if not os.path.exists(temp_motif):
        run_motif_detection(temp_csv, temp_motif)
    motif_df = pd.read_csv(temp_motif, sep='\t')
    
    inference_data = []
    # 重新读取以确保顺序和格式一致
    df_seq = pd.read_csv(temp_csv)
    for _, row in df_seq.iterrows():
        chrom = row['chrom']
        s, e = row['start'], row['end']
        strand = row['strand']
        uid = f"{chrom}:{s}-{e}({strand})"
        
        # 保持与训练完全一致的序列拼接逻辑 (RC处理)
        if strand == '-':
            rc = lambda x: str(Seq(x).reverse_complement()) if pd.notna(x) else ""
            full_seq = rc(row['flank_right']) + rc(row['seq']) + rc(row['flank_left'])
        else:
            full_seq = (row['flank_left'] if pd.notna(row['flank_left']) else "") + \
                       (row['seq'] if pd.notna(row['seq']) else "") + \
                       (row['flank_right'] if pd.notna(row['flank_right']) else "")
        inference_data.append((uid, full_seq))
        
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone, trust_remote_code=True)
    
    # 3. 初始化模型 【关键修改】
    # 训练时 num_token_labels=4 (OBIE)，这里必须一致，否则 load_state_dict 会报错
    model = MotifGuidedSINEClassifier(
        backbone, 
        hidden_dim=256, 
        num_classes=2, 
        num_token_labels=4,  # Changed from 3 to 4
        dropout=0.1
    )
    
    # 加载权重 (兼容 Checkpoint 字典和纯权重文件)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 如果是 Checkpoint (带有 epoch, model_state 等键)，提取真正的权重
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    # 自动处理 'module.' 前缀 (针对 DDP 模型)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    # 4. 推理
    ds = InferenceDataset(inference_data, motif_df, tokenizer, max_len=256)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    results = []
    print("Predicting with robust OBIE decoding...")
    
    with torch.no_grad():
        for batch in tqdm(dl):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            motif_mask = batch['motif_mask'].to(device)
            uids = batch['unique_id']
            raw_seqs = batch['raw_sequence']
            offset_mappings = batch['offset_mapping']
            
            # 模型前向
            global_logits, token_logits = model(input_ids, att_mask, motif_mask)
            
            # Global Classification (SINE vs Non-SINE)
            probs = torch.softmax(global_logits, dim=1)
            sine_probs = probs[:, 1].cpu().numpy()
            print(sine_probs)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            for i, uid in enumerate(uids):
                refined_seq = ""
                # 只有被判定为 SINE 的序列才进行边界修正
                if preds[i] == 1:
                    refined_seq, start_char, end_char = decode_span_search(
                        token_logits[i], 
                        offset_mappings[i], 
                        raw_seqs[i]
                    )
                    if not refined_seq:
                        refined_seq = "SEGMENTATION_FAILED"
                else:
                    refined_seq = "NON_SINE"
                
                results.append({
                    'unique_id': uid,
                    'SINE_prob': float(sine_probs[i]),
                    'Prediction': 'SINE' if preds[i] == 1 else 'Non-SINE',
                    'Refined_Sequence': refined_seq
                })
                
    # 5. 结果导出
    res_df = pd.DataFrame(results)
    df_seq['unique_id'] = df_seq.apply(lambda r: f"{r['chrom']}:{r['start']}-{r['end']}({r['strand']})", axis=1)
    
    final_df = pd.merge(df_seq[['chrom', 'start', 'end', 'label', 'strand', 'unique_id']], res_df, on='unique_id')
    final_df['SINE_prob'] = final_df['SINE_prob'].apply(lambda x: f"{x:.4f}")
    
    final_df.to_csv(args.out_file, sep='\t', index=False)
    print(f"✅ Prediction done: {args.out_file}")

    if args.out_fasta:
        from Bio.SeqRecord import SeqRecord
        sine_records = []
        for _, row in final_df.iterrows():
            # 仅导出成功提取出序列的 SINE 结果
            if row['Prediction'] == 'SINE' and row['Refined_Sequence'] not in ["SEGMENTATION_FAILED", "NON_SINE", ""]:
                seq_obj = Seq(row['Refined_Sequence'])
                rec = SeqRecord(
                    seq_obj,
                    id=f"{row['unique_id']}-{row['SINE_prob']}",
                    description=f"prob={row['SINE_prob']} orig_label={row['label']}"
                )
                sine_records.append(rec)
        
        if sine_records:
            with open(args.out_fasta, "w") as output_handle:
                SeqIO.write(sine_records, output_handle, "fasta")
            print(f"✅ Exported {len(sine_records)} refined sequences to {args.out_fasta}")

if __name__ == "__main__":
    main()