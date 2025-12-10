#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_for_prediction.py (V3: Strict Periodicity Filter)
============================================================
预处理用于预测的输入TSV文件，并强力过滤串联重复。

改进点:
1. ✅ 周期性检测 (Self-Alignment): 检测任意长度(1bp-N/2)的重复单元。
2. ✅ K-mer 优势检测: 防止某种三核苷酸(如 CAGCAG) 占据主导。
3. ✅ 均聚物检测: 检测过长的 PolyA/T (虽然 SINE 有尾巴，但过长通常是重复)。

输入格式:
chrom    start    end    strand    seq    left    right
"""

import argparse
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

class StrictRepeatsFilter:
    def __init__(self, 
                 min_period_identity=0.75, 
                 max_homopolymer_ratio=0.4,
                 max_trimer_ratio=0.45):
        """
        Args:
            min_period_identity (float): 周期性检测的重合度阈值 (0.75 表示错位后75%碱基相同即视为重复)
            max_homopolymer_ratio (float): 单一碱基(如A)占全长的最大比例 (SINE尾巴通常 < 30%)
            max_trimer_ratio (float): 最频繁的3-mer占全长的最大比例
        """
        self.min_period_identity = min_period_identity
        self.max_homopolymer_ratio = max_homopolymer_ratio
        self.max_trimer_ratio = max_trimer_ratio

    def check_periodicity(self, seq_str):
        """
        检测自相关周期性。
        尝试将序列错位 1 到 len/2 的距离，计算重合度。
        """
        n = len(seq_str)
        # 最多检测到序列长度一半的周期
        max_period = n // 2
        
        # 为了速度，只检测 1 到 50 bp 的周期 (覆盖绝大多数微卫星和卫星DNA)
        # 如果序列特别长，可以适当放宽 range
        search_range = range(1, min(max_period, 60) + 1)
        
        for p in search_range:
            # 比较 seq[0:n-p] 和 seq[p:n]
            # 这是一个向量化的快速比较
            # Python 字符串比较较慢，转为 list 或使用 sum
            matches = sum(1 for i in range(n - p) if seq_str[i] == seq_str[i+p])
            score = matches / (n - p)
            
            if score > self.min_period_identity:
                return True, f"Period={p}bp (Score={score:.2f})"
        
        return False, "No Periodicity"

    def check_composition(self, seq_str):
        """检测成分偏差 (均聚物和 K-mer)"""
        n = len(seq_str)
        if n == 0: return False, "Empty"
        
        # 1. 均聚物 (Homopolymer)
        counts = Counter(seq_str)
        top_base_count = counts.most_common(1)[0][1]
        if top_base_count / n > self.max_homopolymer_ratio:
            return False, f"Homopolymer Dominance ({top_base_count/n:.2f})"
            
        # 2. 3-mer 优势 (检测类似 CAGCAGCAG 这种非均聚物重复)
        if n >= 3:
            trimers = [seq_str[i:i+3] for i in range(n-2)]
            trimer_counts = Counter(trimers)
            top_trimer_count = trimer_counts.most_common(1)[0][1]
            # 一个 trimer 占 3bp，所以覆盖度是 count * 3 / n
            # 但这里我们只看频次占比，约等于覆盖度
            if (top_trimer_count * 3) / n > self.max_trimer_ratio:
                return False, f"Trimer Dominance ({top_trimer_count*3/n:.2f})"
                
        return True, "Pass"

    def is_complex(self, seq_str):
        seq_str = seq_str.upper()
        
        # 1. 成分检查
        is_valid, msg = self.check_composition(seq_str)
        if not is_valid:
            return False, msg
            
        # 2. 周期性检查 (最严格的一步)
        is_periodic, msg = self.check_periodicity(seq_str)
        if is_periodic:
            return False, f"Tandem Repeat detected: {msg}"
            
        return True, "Pass"


def preprocess_prediction_input(
    input_tsv: str,
    output_fasta: str,
    output_tsv: str,
    filter_repeats: bool = True
):
    print(f"[INFO] 读取输入文件: {input_tsv}")
    df = pd.read_csv(input_tsv, sep='\t')
    
    # 初始化过滤器 (严格模式)
    # min_period_identity=0.70: 只要有70%的相似度就认为是重复 (容忍变异)
    repeat_filter = StrictRepeatsFilter(
        min_period_identity=0.70, 
        max_homopolymer_ratio=0.55, # 允许55%是单一碱基(宽容PolyA)
        max_trimer_ratio=0.50       # 允许50%是同一种3-mer
    )
    
    records = []
    valid_rows = []
    
    total_count = len(df)
    filtered_count = 0
    
    print(f"[INFO] 开始处理及强力过滤...")
    
    for idx, row in df.iterrows():
        unique_id = f"{row['chrom']}:{row['start']}-{row['end']}({row['strand']})"
        seq_core = str(row['seq'])
        
        # --- 过滤逻辑 ---
        if filter_repeats:
            is_valid, reason = repeat_filter.is_complex(seq_core)
            if not is_valid:
                # 打印前10个被过滤的，方便调试
                if filtered_count < 10:
                    print(f"  [Filter] {unique_id} | {reason}")
                filtered_count += 1
                continue
        # ----------------
        
        row_data = row.to_dict()
        row_data['unique_id'] = unique_id
        valid_rows.append(row_data)
        
        full_sequence = (str(row['left']) + seq_core + str(row['right'])).upper()
        record = SeqRecord(Seq(full_sequence), id=unique_id, description="")
        records.append(record)
    
    # 保存结果
    if valid_rows:
        output_df = pd.DataFrame(valid_rows)
        # 保持必要的列
        cols = ['chrom', 'start', 'end', 'strand', 'seq', 'left', 'right', 'unique_id']
        final_cols = [c for c in cols if c in output_df.columns]
        output_df = output_df[final_cols]
        output_df.to_csv(output_tsv, sep='\t', index=False)
        
        with open(output_fasta, "w") as f:
            SeqIO.write(records, f, "fasta")
            
    print(f"\n[统计]")
    print(f"  原始输入: {total_count}")
    print(f"  过滤掉:   {filtered_count}")
    print(f"  保留:     {len(valid_rows)}")
    print(f"  过滤比例: {filtered_count/total_count:.2%}")
    print(f"✅ 输出已保存: {output_tsv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", required=True)
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--no_filter", action="store_true")
    args = parser.parse_args()
    
    Path(args.output_fasta).parent.mkdir(parents=True, exist_ok=True)
    preprocess_prediction_input(
        args.input_tsv, args.output_fasta, args.output_tsv, filter_repeats=not args.no_filter
    )

if __name__ == "__main__":
    main()