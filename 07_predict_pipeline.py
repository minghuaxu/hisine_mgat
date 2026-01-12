#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_predict_pipeline.py
"""
from peft import get_peft_model, LoraConfig, TaskType
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
from torchcrf import CRF
from model import MotifGuidedSINEClassifier


# ================= 配置 =================
FLANK_LEN = 150 
BATCH_SIZE = 1 # 推理时稍微调大 Batch Size 以加速

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
    def __init__(self, data_list, motif_df, tokenizer, max_len=100): 
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

def extract_sequence_from_tags(pred_tags, offset_mapping, raw_seq):
    """
    改进版：寻找最长的 B...E 或 I...I 片段，避免被侧翼的小噪音干扰。
    Tags: 0=O, 1=B, 2=I, 3=E
    """
    offsets = offset_mapping.cpu().numpy()
    n = len(pred_tags)
    
    # 1. 寻找所有的候选区间 (Candidate Segments)
    candidates = []
    
    i = 0
    while i < n:
        # 寻找 B (1) 或者 I (2) 开始的地方
        if pred_tags[i] in [1, 2]:
            start_idx = i
            # 向后找，直到遇到 O (0) 或者 序列结束
            # 我们寻找 B -> I...I -> E 的完整结构，或者连续的 I
            curr = i
            has_e = False
            while curr < n and pred_tags[curr] in [1, 2, 3]:
                if pred_tags[curr] == 3: # 找到了 E
                    has_e = True
                    break
                curr += 1
            
            end_idx = curr if has_e else (curr - 1)
            
            # 计算字符长度（用于评估哪个最长）
            if start_idx < len(offsets) and end_idx < len(offsets):
                c_start = offsets[start_idx][0]
                c_end = offsets[end_idx][1]
                if c_end > c_start:
                    candidates.append({
                        'start': start_idx,
                        'end': end_idx,
                        'char_len': c_end - c_start,
                        'has_e': has_e,
                        'has_b': pred_tags[start_idx] == 1
                    })
            i = curr + 1
        else:
            i += 1

    if not candidates:
        return "", -1, -1

    # 2. 策略：优先选择包含 B 和 E 的区间；如果有多个，选字符跨度最长的
    # 过滤出相对完整的区间
    strict_candidates = [c for c in candidates if c['has_b'] or c['has_e']]
    if not strict_candidates:
        strict_candidates = candidates # 实在没有就选纯 I 的
    
    # 按长度排序，取最大的
    best = max(strict_candidates, key=lambda x: x['char_len'])
    
    char_start = offsets[best['start']][0]
    char_end = offsets[best['end']][1]
    
    # 边界保护
    char_start = max(0, char_start)
    char_end = min(len(raw_seq), char_end)
    
    refined_seq = raw_seq[char_start:char_end]
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
    parser.add_argument("--min_len", type=int, default=50, help="Minimum length for refined SINE sequences (default: 50)")
    args = parser.parse_args()
    
    # 1. 设置设备
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device = torch.device( "cpu")
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
    
    # 必须应用 LoRA，否则权重键名不匹配
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=True,  # 预测模式
        r=8,                  # 必须与训练代码中的 lora_r 一致
        lora_alpha=32,        # 必须与训练代码中的 lora_alpha 一致
        target_modules=["query", "key", "value", "dense"] # 必须与训练代码一致
    )
    backbone = get_peft_model(backbone, peft_config)
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
    
    # 1. 提取真正的 state_dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    # 2. 智能清理键名
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # 移除 DDP 的 module.
        if name.startswith('module.'):
            name = name[7:]
        
        # 【关键修复】：如果权重里有 base_model.model，但模型实例里没有（或反之）
        # 我们统一把键名中的 base_model.model. 删掉，因为后面 load_state_dict 
        # 会通过灵活匹配来处理层级
        name = name.replace('base_model.model.', '')
        cleaned_state_dict[name] = v

    # 3. 同样的，我们也需要处理模型实例中的键名匹配
    # PEFT 模型加载最稳妥的方法是使用这种方式：
    current_model_dict = model.state_dict()
    final_state_dict = {}

    for k in current_model_dict.keys():
        # 寻找 cleaned_state_dict 中最匹配的后缀
        # 例如模型需要: backbone.base_model.model.esm.xxx
        # 我们的权重里有: backbone.esm.xxx
        short_k = k.replace('base_model.model.', '')
        if short_k in cleaned_state_dict:
            final_state_dict[k] = cleaned_state_dict[short_k]
        elif k in cleaned_state_dict:
            final_state_dict[k] = cleaned_state_dict[k]

    msg = model.load_state_dict(final_state_dict, strict=False)
    print(f"✅ Model loaded with info: {msg}")

    model.to(device)
    model.eval()
    
    # 4. 推理
    ds = InferenceDataset(inference_data, motif_df, tokenizer, max_len=100)
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
            
            # 前向传播
            # 推理模式下，model 返回 (global_logits, emissions, None)
            global_logits, emissions, _ = model(input_ids, att_mask, motif_mask, token_labels=None)
            # print(global_logits)

            # print(emissions)
            
            # 调用 Viterbi 解码
            # 返回 List[List[int]]，每个样本的 tag 序列
            # 注意：raw_model 处理 (如果是 DDP/DataParallel)
            raw_model = model.module if hasattr(model, 'module') else model
            decoded_tags_list = raw_model.decode(emissions, att_mask)
            # print(decoded_tags_list)
            
            # Global Classification (SINE vs Non-SINE)
            probs = torch.softmax(global_logits, dim=1)

            # temperature = 0.5  # 这个值越小，分布越尖锐（越自信
            # probs = torch.softmax(global_logits / temperature, dim=1)

            # print(probs)
            sine_probs = probs[:, 1].cpu().numpy()
            # print(sine_probs)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            for i, uid in enumerate(uids):
                refined_seq = ""
                # 只有被判定为 SINE 的序列才进行边界修正
                if preds[i] == 1:
                    # 使用解析函数提取序列
                    refined_seq, start_char, end_char = extract_sequence_from_tags(
                        decoded_tags_list[i], # List[int]
                        offset_mappings[i],   # Tensor (L, 2)
                        raw_seqs[i]           # str
                    )
                    if not refined_seq:
                        refined_seq = "SEGMENTATION_FAILED"
                    elif len(refined_seq) < args.min_len:
                        # 虽然模型认为是SINE，但分割出来的序列太短，标记为被过滤
                        refined_seq = f"TOO_SHORT"
                        # 可选：如果你希望这种情况在 Prediction 列也显示为过滤，可以取消下面注释
                        # final_pred = "Filtered" 
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
        # --- 修改：定义非法字符串列表，包含新增加的 TOO_SHORT ---
        invalid_tags = ["SEGMENTATION_FAILED", "NON_SINE", "","TOO_SHORT"]
        for _, row in final_df.iterrows():
            ref_seq = row['Refined_Sequence']
            # 仅导出成功提取出序列的 SINE 结果
            if row['Prediction'] == 'SINE' and ref_seq not in invalid_tags:
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