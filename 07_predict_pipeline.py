#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_predict_pipeline.py (Segmentation Enabled)
=============================================
功能升级：
1. 适配多任务模型 (Global Class + Token Seg)。
2. 新增 [边界修正] 功能：利用 Segmentation Head 自动截取 SINE 主体。
3. 输出结果包含 refined_sequence (修正后的序列)。
"""

import argparse
import pandas as pd
import numpy as np
import torch
import subprocess
import os
import sys
import shutil
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
FLANK_LEN = 150 # 提取时保留的侧翼长度
BATCH_SIZE = 1

def gaussian_kernel_smooth(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if sigma <= 0: return mask
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return np.convolve(mask, kernel, mode='same')

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
        self.data = data_list # list of (uid, seq)
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
        # 截断处理
        if len(content_ids) > self.max_len - 1:
            content_ids = content_ids[:self.max_len - 1]
            
        # Token 映射 (关键：用于将预测的 Token 索引映射回原始序列坐标)
        mapping = []
        pos = 0
        for tid in content_ids:
            tok_len = max(1, len(self.tokenizer.decode([tid])))
            mapping.append((pos, pos + tok_len))
            pos += tok_len
            
        # 对齐 Mask
        if pos < len(base_mask): base_mask = base_mask[:pos]
        elif pos > len(base_mask): 
            base_mask = np.concatenate([base_mask, np.full(pos-len(base_mask), 0.3)])
            
        input_ids = torch.full((self.max_len,), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        token_mask = torch.zeros(self.max_len, dtype=torch.float32)
        
        # Mapping 也要 Padding 以便 Batch 处理
        padded_mapping = torch.zeros((self.max_len, 2), dtype=torch.long)
        
        input_ids[0] = self.cls_id
        attention_mask[0] = 1
        token_mask[0] = float(base_mask.mean()) if len(base_mask)>0 else 0.3
        
        valid_len = len(content_ids)
        input_ids[1 : 1+valid_len] = torch.tensor(content_ids, dtype=torch.long)
        attention_mask[1 : 1+valid_len] = 1
        
        # 填充 Mapping 和 TokenMask
        for i, (s, e) in enumerate(mapping):
            token_idx = i + 1
            seg = base_mask[s:e]
            token_mask[token_idx] = float(np.max(seg)) if len(seg)>0 else 0.3
            padded_mapping[token_idx] = torch.tensor([s, e])
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motif_mask': token_mask,
            'offset_mapping': padded_mapping, # [新增] 用于解码分割结果
            'raw_sequence': seq,              # [新增] 原始序列
            'unique_id': uid
        }

def load_genome(fasta_path):
    print(f"加载基因组: {fasta_path} ...")
    return SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

def extract_sequences(bed_path, genome_dict, temp_csv_path):
    print("提取序列并构建临时 CSV (Extraction Mode: Forward Strand)...")
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
            
            if chrom not in genome_dict:
                print(f"[Warn] Chrom {chrom} not found in fasta.")
                continue
                
            full_chrom_seq = genome_dict[chrom].seq
            
            core_seq = str(full_chrom_seq[start:end])
            l_start = max(0, start - FLANK_LEN)
            r_end = min(len(full_chrom_seq), end + FLANK_LEN)
            
            left_seq = str(full_chrom_seq[l_start:start])
            right_seq = str(full_chrom_seq[end:r_end])
            
            data.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand,
                'seq': core_seq,
                'flank_left': left_seq,
                'flank_right': right_seq,
                'label': label,
                'source_type': 'Simulated'
            })
            
    df = pd.DataFrame(data)
    df.to_csv(temp_csv_path, index=False)
    return df

def run_motif_detection(in_csv, out_tsv):
    print("运行 Motif 检测 (这可能需要一点时间)...")
    cmd = [
        "python", "05_detect_motifs_parallel.py",
        "--in_csv", in_csv,
        "--out_tsv", out_tsv,
        "--threads", "16" 
    ]
    subprocess.run(cmd, check=True)

def decode_segmentation(token_logits, offset_mapping, raw_seq):
    """
    根据 Token Logits 解码出 SINE 的精确边界
    Tag定义: 0=Bg, 1=TSD, 2=Body, 3=PolyA, 4=Pad
    策略: 寻找包含 Class 1, 2, 3 的连续区域
    """
    preds = torch.argmax(token_logits, dim=-1).cpu().numpy() # (Seq_Len,)
    offsets = offset_mapping.cpu().numpy() # (Seq_Len, 2)
    
    # 找到所有被预测为 SINE 组件 (TSD/Body/PolyA) 的索引
    # 忽略 [CLS] (index 0)
    sine_indices = np.where((preds >= 1) & (preds <= 3))[0]
    sine_indices = sine_indices[sine_indices > 0] # 排除 CLS
    
    if len(sine_indices) == 0:
        return "", -1, -1 # 没找到
        
    # 简单策略：取首尾作为边界 (覆盖整个 SINE)
    # 更高级策略可以是寻找最长连续段，这里用简单版足够有效
    start_idx = sine_indices[0]
    end_idx = sine_indices[-1]
    
    # 映射回原始序列坐标
    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    
    # 截取序列
    refined_seq = raw_seq[char_start:char_end]
    
    return refined_seq, char_start, char_end

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True, help="Input BED file")
    parser.add_argument("--fasta", required=True, help="Simulated genome FASTA")
    parser.add_argument("--model_path", required=True, help="Path to best_model.pt")
    parser.add_argument("--backbone", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--out_file", default="predictions.tsv")
    parser.add_argument("--out_fasta", default="predictions_refined.fasta")
    parser.add_argument("--temp_dir", default="temp_pred_workspace")
    args = parser.parse_args()
    
    os.makedirs(args.temp_dir, exist_ok=True)
    temp_csv = os.path.join(args.temp_dir, "input_seqs.csv")
    temp_motif = os.path.join(args.temp_dir, "motifs.tsv")
    
    # 1. 提取 & 检测 (保持不变)
    genome = load_genome(args.fasta)
    df_seq = extract_sequences(args.bed, genome, temp_csv)
    
    if not os.path.exists(temp_motif):
        run_motif_detection(temp_csv, temp_motif)
        
    motif_df = pd.read_csv(temp_motif, sep='\t')
    
    # 3. 准备数据
    inference_data = []
    df_seq = pd.read_csv(temp_csv)
    
    for _, row in df_seq.iterrows():
        chrom = row['chrom']
        s, e = row['start'], row['end']
        strand = row['strand']
        uid = f"{chrom}:{s}-{e}({strand})"
        
        # Strand 处理 (与训练一致)
        if strand == '-':
            rc = lambda x: str(Seq(x).reverse_complement()) if pd.notna(x) else ""
            full_seq = rc(row['flank_right']) + rc(row['seq']) + rc(row['flank_left'])
        else:
            full_seq = (row['flank_left'] if pd.notna(row['flank_left']) else "") + \
                       (row['seq'] if pd.notna(row['seq']) else "") + \
                       (row['flank_right'] if pd.notna(row['flank_right']) else "")
            
        inference_data.append((uid, full_seq))
        
    # 4. 加载模型 (注意 num_token_labels=5)
    print("加载模型...")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone, trust_remote_code=True)
    
    # [修改] 必须指定 num_token_labels=5
    model = MotifGuidedSINEClassifier(backbone, hidden_dim=256, num_classes=2, num_token_labels=5)
    
    # 加载权重
    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    # 5. 推理 & 分割
    ds = InferenceDataset(inference_data, motif_df, tokenizer)
    # 注意: batch_size=1 是最稳妥的，因为 raw_sequence 长度不一，但 collate_fn 默认处理不好字符串列表
    # 这里我们用 batch_size > 1，但要注意 raw_sequence 在 batch 里是个 tuple
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    results = []
    
    print("开始预测与边界修正...")
    with torch.no_grad():
        for batch in tqdm(dl):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            motif_mask = batch['motif_mask'].to(device)
            uids = batch['unique_id']
            raw_seqs = batch['raw_sequence'] # Tuple of strings
            offset_mappings = batch['offset_mapping'] # (B, L, 2)
            
            # [修改] 解包两个输出
            global_logits, token_logits = model(input_ids, att_mask, motif_mask)
            
            # 全局概率
            probs = torch.softmax(global_logits, dim=1)
            sine_probs = probs[:, 1].cpu().numpy()
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # 分割解码
            for i, uid in enumerate(uids):
                # 仅对预测为 SINE 的样本进行边界修正
                refined_seq = ""
                if preds[i] == 1:
                    refined_seq, _, _ = decode_segmentation(token_logits[i], offset_mappings[i], raw_seqs[i])
                    # 如果分割没找到有效区域，回退到原始 Core (这在逻辑上有点复杂，暂且留空或用全长)
                    if not refined_seq: 
                        refined_seq = "SEGMENTATION_FAILED"
                
                results.append({
                    'unique_id': uid,
                    'SINE_prob': float(sine_probs[i]),
                    'Prediction': 'SINE' if preds[i] == 1 else 'Non-SINE',
                    'Refined_Sequence': refined_seq
                })
                
    # 6. 保存结果
    res_df = pd.DataFrame(results)
    df_seq['unique_id'] = df_seq.apply(lambda r: f"{r['chrom']}:{r['start']}-{r['end']}({r['strand']})", axis=1)
    
    # Merge
    final_df = pd.merge(df_seq[['chrom', 'start', 'end', 'label', 'strand', 'unique_id']], res_df, on='unique_id')
    final_df['SINE_prob'] = final_df['SINE_prob'].apply(lambda x: f"{x:.4f}")
    
    final_df.to_csv(args.out_file, sep='\t', index=False)
    print(f"✅ 预测完成！TSV 已保存至: {args.out_file}")

    # 导出 FASTA (仅导出修正后的序列)
    if args.out_fasta:
        print(f"正在导出修正后的 SINE 序列至: {args.out_fasta} ...")
        from Bio.SeqRecord import SeqRecord
        sine_records = []
        for _, row in final_df.iterrows():
            if row['Prediction'] == 'SINE' and row['Refined_Sequence'] and row['Refined_Sequence'] != "SEGMENTATION_FAILED":
                
                # Refined_Sequence 已经是基于输入序列切出来的
                # 训练时我们对负链做了 RC，所以这里的 Refined_Sequence 已经是 SINE 的正义链了！
                # 不需要再根据 Strand 翻转了，直接保存即可。
                
                seq_obj = Seq(row['Refined_Sequence'])
                rec = SeqRecord(
                    seq_obj,
                    id=row['unique_id'],
                    description=f"prob={row['SINE_prob']} orig_label={row['label']} (Model-Refined Boundary)"
                )
                sine_records.append(rec)
        
        if sine_records:
            with open(args.out_fasta, "w") as output_handle:
                SeqIO.write(sine_records, output_handle, "fasta")
            print(f"✅ 成功导出 {len(sine_records)} 条修正后的序列。")

if __name__ == "__main__":
    main()