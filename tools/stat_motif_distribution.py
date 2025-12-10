#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stat_motif_distribution.py (V2: Adapted for Tiered Status)
==========================================================
统计并对比正负样本的Motif特征分布。
适配 HIGH_CONF / MED_CONF / LOW_CONF 状态标签。
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_aggregate(file_paths):
    dfs = []
    for p in file_paths:
        path = Path(p)
        if not path.exists(): continue
        try:
            df = pd.read_csv(path, sep='\t')
            dfs.append(df)
        except: pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def calculate_stats(df, label_name):
    if df.empty: return {"Label": label_name, "Total": 0}
    
    total = len(df)
    has_A = df['A_pos'] != -1
    has_B = df['B_pos'] != -1
    has_polyA = df['polyA_len'] > 0
    has_TSD = df['TSD_len'] > 0
    
    status_counts = df['detection_status'].value_counts()
    
    stats = {
        "Label": label_name,
        "Total": total,
        
        "Has A-box": has_A.sum(),
        "Has A-box (%)": (has_A.sum() / total) * 100,
        "Has B-box": has_B.sum(),
        "Has B-box (%)": (has_B.sum() / total) * 100,
        "Has PolyA": has_polyA.sum(),
        "Has PolyA (%)": (has_polyA.sum() / total) * 100,
        "Has TSD": has_TSD.sum(),
        "Has TSD (%)": (has_TSD.sum() / total) * 100,
        
        "A + B Box": (has_A & has_B).sum(),
        "A + B Box (%)": ((has_A & has_B).sum() / total) * 100,
        
        # 新状态统计
        "HIGH_CONF": status_counts.get("HIGH_CONF", 0),
        "HIGH_CONF (%)": (status_counts.get("HIGH_CONF", 0) / total) * 100,
        
        "MED_CONF": status_counts.get("MED_CONF", 0),
        "MED_CONF (%)": (status_counts.get("MED_CONF", 0) / total) * 100,
        
        "LOW_CONF": status_counts.get("LOW_CONF", 0),
        "LOW_CONF (%)": (status_counts.get("LOW_CONF", 0) / total) * 100,
        
        "NO": status_counts.get("NO", 0),
        "NO (%)": (status_counts.get("NO", 0) / total) * 100,
        
        "Avg PolyA": df.loc[has_polyA, 'polyA_len'].mean() if has_polyA.any() else 0,
        "Avg TSD": df.loc[has_TSD, 'TSD_len'].mean() if has_TSD.any() else 0,
        "Avg A-Score": df.loc[has_A, 'A_score'].mean() if has_A.any() else 0,
    }
    return stats

def print_comparison(pos_stats, neg_stats):
    metrics = [
        ("Total", "{:,}"),
        ("--- Features ---", ""),
        ("Has A-box", "{:,} ({:.1f}%)"),
        ("Has B-box", "{:,} ({:.1f}%)"),
        ("Has PolyA", "{:,} ({:.1f}%)"),
        ("Has TSD", "{:,} ({:.1f}%)"),
        ("--- Status ---", ""),
        ("HIGH_CONF", "{:,} ({:.1f}%)"),
        ("MED_CONF", "{:,} ({:.1f}%)"),
        ("LOW_CONF", "{:,} ({:.1f}%)"),
        ("NO", "{:,} ({:.1f}%)"),
        ("--- Averages ---", ""),
        ("Avg PolyA", "{:.1f} bp"),
        ("Avg TSD", "{:.1f} bp"),
        ("Avg A-Score", "{:.2f}"),
    ]
    
    print(f"\n{'Metric':<25} | {'Positives':<20} | {'Negatives':<20}")
    print("-" * 70)
    for key, fmt in metrics:
        if key.startswith("---"):
            print(f"{key:<25} | {'-'*20} | {'-'*20}")
            continue
        
        if "%" in fmt:
            v_p = pos_stats.get(key, 0)
            p_p = pos_stats.get(key + " (%)", 0)
            s_p = fmt.format(v_p, p_p)
            
            v_n = neg_stats.get(key, 0)
            p_n = neg_stats.get(key + " (%)", 0)
            s_n = fmt.format(v_n, p_n)
        else:
            s_p = fmt.format(pos_stats.get(key, 0))
            s_n = fmt.format(neg_stats.get(key, 0))
            
        print(f"{key:<25} | {s_p:<20} | {s_n:<20}")
    print("-" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_motifs", nargs='+', required=True)
    parser.add_argument("--neg_motifs", nargs='+', required=True)
    args = parser.parse_args()
    
    pos_stats = calculate_stats(load_and_aggregate(args.pos_motifs), "Pos")
    neg_stats = calculate_stats(load_and_aggregate(args.neg_motifs), "Neg")
    print_comparison(pos_stats, neg_stats)

if __name__ == "__main__":
    main()


# python tools/stat_motif_distribution.py \
#     --pos_motifs results/*/positives/motifs.motifs.tsv \
#     --neg_motifs results/*/negatives/motifs.motifs.tsv

# # 检查负样本中 A-box 的实际分数分布
# awk 'BEGIN{FS="\t"} NR>1 && $5>=0 {print $6}' \
#     results/*/negatives/motifs.motifs.tsv | \
#     sort -n | head -20  # 看最低的20个分数