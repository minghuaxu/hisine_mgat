#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_build_masks.py
=================
功能：离线预计算 Mask 矩阵
输入：
  1. 序列文件 (test.csv)
  2. Motif坐标文件 (test_motif_pos.tsv)
输出：
  masks.pt (推荐: 包含 {unique_id: numpy_float_array} 的字典)
"""

import pandas as pd
import numpy as np
import torch
import csv
from Bio.Seq import Seq
from tqdm import tqdm
import argparse

# ================= 配置区 =================
# Mask 权重配置 (你可以根据需要调整)
WEIGHTS = {
    'background': 0.3,
    'A_box': 2.0,
    'B_box': 2.0,
    'polyA': 1.6,
    'TSD': 1.0  # TSD 稍微加重一点
}

def revcomp(s):
    if pd.isna(s) or s == "": return ""
    return str(Seq(s).reverse_complement())

def build_masks(csv_path, motif_tsv_path, out_path):
    # 1. 读取数据
    print(f"Loading data...")
    df_seq = pd.read_csv(csv_path)
    df_motif = pd.read_csv(motif_tsv_path, sep='\t')

    duplicates = df_motif[df_motif['unique_id'].duplicated(keep=False)]
    print(duplicates.sort_values('unique_id'))
    
    # 将 Motif 数据转为以 unique_id 为 key 的字典，方便快速查找
    # 这一步非常关键，避免在循环中重复 query dataframe
    motif_dict = df_motif.set_index('unique_id').to_dict('index')
    
    mask_storage = {}
    
    print(f"Building masks for {len(df_seq)} sequences...")
    
    for _, row in tqdm(df_seq.iterrows(), total=len(df_seq)):
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        strand = row['strand']
        
        # 1. 构造 Unique ID (必须与 03 步完全一致)
        unique_id = f"{chrom}:{start}-{end}({strand})"
        
        # 2. 重建拼接序列以获取正确的长度 (L)
        # 必须复刻 detect_motifs.py 的逻辑
        if strand == '-':
            core = revcomp(row['seq'])
            left = revcomp(row['flank_right']) # 负链交换左右
            right = revcomp(row['flank_left'])
        else:
            core = row['seq']
            left = row['flank_left']
            right = row['flank_right']
            
        full_seq_len = len(left) + len(core) + len(right)
        
        # 3. 初始化 Mask 数组
        mask = np.full(full_seq_len, WEIGHTS['background'], dtype=np.float32)
        
        # 4. 填充 Motif 权重
        if unique_id in motif_dict:
            m = motif_dict[unique_id]
            
            # 辅助函数：安全填充
            def fill_region(k_start, k_end, weight):
                s, e = m.get(k_start, -1), m.get(k_end, -1)
                if s != -1 and e != -1:
                    s, e = int(s), int(e)
                    # 边界保护
                    s = max(0, s)
                    e = min(full_seq_len, e)
                    if s < e:
                        # 取最大值覆盖 (例如 TSD 和 A-box 重叠)
                        mask[s:e] = np.maximum(mask[s:e], weight)

            # 填充各个区域
            fill_region('left_TSD_start', 'left_TSD_end', WEIGHTS['TSD'])
            fill_region('right_TSD_start', 'right_TSD_end', WEIGHTS['TSD'])
            fill_region('polyA_start', 'polyA_end', WEIGHTS['polyA'])
            fill_region('A_box_start', 'A_box_end', WEIGHTS['A_box'])
            fill_region('B_box_start', 'B_box_end', WEIGHTS['B_box'])
            
        # 5. 存入字典
        # 为了节省空间，可以转为 float16，但在训练时要转回 float32
        mask_storage[unique_id] = mask.astype(np.float32)
        # print(mask)

    # 6. 保存
    print(f"Saving masks to {out_path} ...")
    torch.save(mask_storage, out_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Original sequences (test.csv)")
    parser.add_argument("--motif", required=True, help="Motif coordinates (test_motif_pos.tsv)")
    parser.add_argument("--out", default="masks.pt", help="Output file (masks.pt)")
    args = parser.parse_args()
    
    build_masks(args.csv, args.motif, args.out)

