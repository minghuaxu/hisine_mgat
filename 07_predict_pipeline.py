#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_predict_pipeline.py (Robust BIO Segmentation)
================================================
功能升级：
1. 适配 BIO 标签体系 (0=Outside, 1=Begin, 2=Inside)。
2. 引入【鲁棒解码算法】：
   - 概率平滑 (Gaussian Smoothing)
   - 空洞填充 (Gap Filling)
   - 长度过滤 (Min Length Filtering)
3. 修复了破碎预测问题。
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

# 引入项目模块
sys.path.insert(0, str(Path(__file__).parent))
from model import MotifGuidedSINEClassifier

# ================= 配置 =================
FLANK_LEN = 150 
BATCH_SIZE = 1 # 推理可以适当大一点

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
    # 找 0 的区域
    mask = mask.astype(bool)
    padded = np.concatenate(([True], mask, [True])) # Pad true to handle boundaries
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == -1)[0]
    ends = np.where(diff == 1)[0]
    
    # starts 是 0 区域的起点，ends 是 0 区域的终点
    # 注意：这里的逻辑是反的，我们要找 0 (False) 的连续段，且两边是 1 (True)
    
    # 更简单的方法：直接遍历
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
    def __init__(self, data_list, motif_df, tokenizer, max_len=1024):
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
    cmd = ["python", "05_detect_motifs_parallel.py", "--in_csv", in_csv, "--out_tsv", out_tsv, "--threads", "16"]
    subprocess.run(cmd, check=True)

def decode_robust(token_logits, offset_mapping, raw_seq, threshold=0.4, min_len=30):
    """
    OBIE 解码 (4分类): 0=Outside, 1=Begin, 2=Inside, 3=End
    目标：提取 SINE 主体
    策略：将 B, I, E 的概率求和作为前景概率，平滑后提取最长连续区域
    """
    # 1. 获取概率
    # 注意：这里维度应该是 (L, 4)
    probs = torch.softmax(token_logits, dim=-1).cpu().numpy() 
    
    # 【关键修改】SINE Score = P(Begin) + P(Inside) + P(End)
    # 我们认为 1, 2, 3 都是 SINE 的一部分，只有 0 是背景
    sine_score = probs[:, 1] + probs[:, 2] + probs[:, 3]
    
    # --- 调试打印开始 ---
    print("\n" + "="*20 + " Token Probabilities Debug " + "="*20)
    
    # 打印 Begin (1)
    print(">>> P(Begin/1):")
    Plist = probs[:, 1]
    for i in range(len(Plist)):
        if i > 0 and i % 50 == 0: print()
        # 高亮显示大概率区域
        val = Plist[i]
        char = f"{val:.1f}" if val < 0.5 else f"[{val:.1f}]"
        print(f"{char:>5}", end=" ")
    print()

    # 打印 Inside (2)
    print("-" * 50)
    print(">>> P(Inside/2):")
    Ilist = probs[:, 2]
    for i in range(len(Ilist)):
        if i > 0 and i % 50 == 0: print()
        val = Ilist[i]
        char = f"{val:.1f}" if val < 0.5 else f"[{val:.1f}]"
        print(f"{char:>5}", end=" ")
    print()

    # 【新增】打印 End (3) - 这对检查边界非常重要
    print("-" * 50)
    print(">>> P(End/3):")
    Elist = probs[:, 3]
    for i in range(len(Elist)):
        if i > 0 and i % 50 == 0: print()
        val = Elist[i]
        char = f"{val:.1f}" if val < 0.5 else f"[{val:.1f}]"
        print(f"{char:>5}", end=" ")
    print("\n" + "="*60)
    # --- 调试打印结束 ---

    # 排除 CLS/Pad
    valid_mask = (offset_mapping.cpu().numpy().sum(axis=1) > 0)
    sine_score = sine_score * valid_mask 
    
    # 2. 平滑 (Window=5)
    # 这能把间断的 B...I...I...E 连成一条线
    smoothed_score = gaussian_smooth(sine_score, window_size=5)
    
    # 3. 生成掩码
    sine_mask = (smoothed_score > threshold)
    # 调试：打印掩码分布
    # print(f"Mask active count: {np.sum(sine_mask)}")
    
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
    
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    
    refined_seq = raw_seq[char_start:char_end]
    
    # 7. 长度过滤
    if len(refined_seq) < min_len:
        return "", -1, -1
        
    return refined_seq, char_start, char_end

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backbone", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--out_file", default="predictions.tsv")
    parser.add_argument("--out_fasta", default="predictions_refined.fasta")
    parser.add_argument("--temp_dir", default="temp_pred_workspace")
    args = parser.parse_args()
    
    os.makedirs(args.temp_dir, exist_ok=True)
    temp_csv = os.path.join(args.temp_dir, "input_seqs.csv")
    temp_motif = os.path.join(args.temp_dir, "motifs.tsv")
    
    genome = load_genome(args.fasta)
    df_seq = extract_sequences(args.bed, genome, temp_csv)
    
    if not os.path.exists(temp_motif):
        run_motif_detection(temp_csv, temp_motif)
    motif_df = pd.read_csv(temp_motif, sep='\t')
    
    inference_data = []
    df_seq = pd.read_csv(temp_csv)
    for _, row in df_seq.iterrows():
        chrom = row['chrom']
        s, e = row['start'], row['end']
        strand = row['strand']
        uid = f"{chrom}:{s}-{e}({strand})"
        if strand == '-':
            rc = lambda x: str(Seq(x).reverse_complement()) if pd.notna(x) else ""
            full_seq = rc(row['flank_right']) + rc(row['seq']) + rc(row['flank_left'])
        else:
            full_seq = (row['flank_left'] if pd.notna(row['flank_left']) else "") + \
                       (row['seq'] if pd.notna(row['seq']) else "") + \
                       (row['flank_right'] if pd.notna(row['flank_right']) else "")
        inference_data.append((uid, full_seq))
        
    print("Loading model...")
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone, trust_remote_code=True)
    
    # 3 分类 (BIO)
    model = MotifGuidedSINEClassifier(backbone, hidden_dim=256, num_classes=2, num_token_labels=3)
    
    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    ds = InferenceDataset(inference_data, motif_df, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    results = []
    
    print("Predicting with robust BIO decoding...")
    with torch.no_grad():
        for batch in tqdm(dl):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            motif_mask = batch['motif_mask'].to(device)
            uids = batch['unique_id']
            raw_seqs = batch['raw_sequence']
            offset_mappings = batch['offset_mapping']
            
            global_logits, token_logits = model(input_ids, att_mask, motif_mask)
            
            probs = torch.softmax(global_logits, dim=1)
            sine_probs = probs[:, 1].cpu().numpy()
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            for i, uid in enumerate(uids):
                refined_seq = ""
                # 如果全局分类是 SINE (Class 1)
                if preds[i] == 1:
                    # 使用鲁棒解码
                    refined_seq, _, _ = decode_robust(token_logits[i], offset_mappings[i], raw_seqs[i])
                    if not refined_seq:
                        refined_seq = "SEGMENTATION_FAILED"
                
                results.append({
                    'unique_id': uid,
                    'SINE_prob': float(sine_probs[i]),
                    'Prediction': 'SINE' if preds[i] == 1 else 'Non-SINE',
                    'Refined_Sequence': refined_seq
                })
                
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
            if row['Prediction'] == 'SINE' and row['Refined_Sequence'] and row['Refined_Sequence'] != "SEGMENTATION_FAILED":
                seq_obj = Seq(row['Refined_Sequence'])
                rec = SeqRecord(
                    seq_obj,
                    id=row['unique_id'],
                    description=f"prob={row['SINE_prob']} orig_label={row['label']}"
                )
                sine_records.append(rec)
        
        if sine_records:
            with open(args.out_fasta, "w") as output_handle:
                SeqIO.write(sine_records, output_handle, "fasta")
            print(f"✅ Exported {len(sine_records)} sequences to {args.out_fasta}")

if __name__ == "__main__":
    main()