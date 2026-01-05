#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_filter_rrna_gff.py (Updated)
===============================
功能更新：
除了输出清洗后的 SINE，还会把被过滤掉的 rRNA 序列保存为单独的文件，
作为 Hard Negative 样本。
"""

import argparse
import pandas as pd
from collections import defaultdict

def parse_gff_for_rrna(gff_path):
    """
    解析 GFF，返回 rRNA 的区间列表。
    """
    rrna_intervals = defaultdict(list)
    count = 0
    
    print(f"[Info] 正在解析 GFF 文件: {gff_path}")
    
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            
            chrom = parts[0]
            feature_type = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            attributes = parts[8]
            
            is_rrna = False
            
            # 判断逻辑
            if feature_type == 'rRNA':
                is_rrna = True
            elif 'rRNA' in attributes:
                if "gene_biotype=rRNA" in attributes or "product=5S ribosomal RNA" in attributes or "product=18S ribosomal RNA" in attributes:
                     is_rrna = True

            if is_rrna:
                rrna_intervals[chrom].append((start - 1, end))
                count += 1

    print(f"[Info] 提取到 {count} 个 rRNA 基因区间。")
    return rrna_intervals

def is_overlapping(chrom, start, end, rrna_db):
    if chrom not in rrna_db:
        return False
    for r_start, r_end in rrna_db[chrom]:
        intersect_start = max(start, r_start)
        intersect_end = min(end, r_end)
        if intersect_end > intersect_start:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="根据 GFF 过滤 SINE 并保存 rRNA 负样本。")
    parser.add_argument("--input_tsv", required=True, help="输入的 SINE 候选表格")
    parser.add_argument("--gff", required=True, help="基因组 GFF 文件")
    parser.add_argument("--output_tsv", required=True, help="输出的正样本 (Clean SINE)")
    parser.add_argument("--output_negatives", required=True, help="输出的负样本 (rRNA Contamination)")
    args = parser.parse_args()

    # 1. 提取 rRNA 坐标
    rrna_db = parse_gff_for_rrna(args.gff)
    
    # 2. 读取候选 SINE
    try:
        df = pd.read_csv(args.input_tsv, sep='\t')
    except Exception:
        print("[Error] 读取输入 TSV 失败。")
        return

    if df.empty:
        print("[Warn] 输入文件为空，生成空的输出文件。")
        df.to_csv(args.output_tsv, sep='\t', index=False)
        df.to_csv(args.output_negatives, sep='\t', index=False)
        return

    print(f"[Info] 过滤前候选数: {len(df)}")
    
    # 3. 执行拆分
    # 使用布尔索引来拆分 DataFrame
    is_contamination = []
    
    for _, row in df.iterrows():
        chrom = str(row['chrom'])
        start = int(row['start'])
        end = int(row['end'])
        
        if is_overlapping(chrom, start, end, rrna_db):
            is_contamination.append(True)
        else:
            is_contamination.append(False)
            
    # 转换为 Series 以便切片
    mask = pd.Series(is_contamination)
    
    # 拆分数据
    df_negatives = df[mask]   # 脏数据 (Hard Negatives)
    df_clean = df[~mask]      # 干净数据 (Positives)
    
    # 4. 保存文件
    df_clean.to_csv(args.output_tsv, sep='\t', index=False)
    df_negatives.to_csv(args.output_negatives, sep='\t', index=False)
    
    print(f"[Info] 发现 rRNA 污染 (Hard Negatives): {len(df_negatives)}")
    print(f"[Info] 最终保留 SINE (Clean Positives): {len(df_clean)}")
    print(f"✅ 正样本已保存至: {args.output_tsv}")
    print(f"✅ 负样本已保存至: {args.output_negatives}")

if __name__ == "__main__":
    main()