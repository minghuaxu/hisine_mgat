#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_training_data_by_ultra_v2.py
===================================
基于 ULTRA 检测结果的高级清洗脚本。

功能:
1. 解析 ULTRA 输出 (跳过头部元数据)。
2. 对每条序列合并所有重复区间，计算覆盖度。
3. 区分 "简单PolyA重复(Period=1)" 和 "复杂结构重复(Period>1)"。
4. 生成详细的诊断表格 (CSV)。
5. 输出清洗后的 FASTA 和 被剔除的 FASTA。
"""

import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path

def parse_ultra_output(ultra_file):
    """
    鲁棒地解析 ULTRA 输出文件，跳过 JSON 和 Log。
    返回 pandas DataFrame。
    """
    rows = []
    header = None
    is_table_started = False
    
    with open(ultra_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 寻找表头
            if line.startswith("SeqID") and "Start" in line and "Period" in line:
                header = line.split('\t')
                is_table_started = True
                continue
            
            # 读取表格内容
            if is_table_started:
                # 简单防错：确保列数匹配
                parts = line.split('\t')
                if len(parts) == len(header):
                    rows.append(parts)
    
    if not rows:
        print("[WARN] ULTRA 输出中没有找到重复记录。")
        return pd.DataFrame(columns=['SeqID', 'Start', 'End', 'Period', 'Score'])

    # 创建 DataFrame
    df = pd.DataFrame(rows, columns=header)
    
    # 类型转换
    df['Start'] = df['Start'].astype(int)
    df['End'] = df['End'].astype(int)
    df['Period'] = df['Period'].astype(int)
    df['Score'] = df['Score'].astype(float)
    return df

def calculate_merged_coverage(repeat_intervals):
    """
    计算区间并集的总长度。
    输入: [(start, end), (start, end), ...]
    输出: 总覆盖长度
    """
    if not repeat_intervals:
        return 0
    
    # 按 start 排序
    sorted_intervals = sorted(repeat_intervals, key=lambda x: x[0])
    
    merged = []
    for current in sorted_intervals:
        if not merged:
            merged.append(list(current))
        else:
            prev = merged[-1]
            if current[0] < prev[1]: # 有重叠
                prev[1] = max(prev[1], current[1])
            else:
                merged.append(list(current))
    
    total_len = sum(end - start for start, end in merged)
    return total_len

def analyze_repeats(df_ultra, fasta_lens):
    """
    汇总每条序列的重复信息
    """
    summary_data = []
    
    # 按 SeqID 分组
    grouped = df_ultra.groupby('SeqID')
    
    for seq_id, group in grouped:
        # 注意：ULTRA 的 ID 可能和 FASTA ID 有微小差异（如空格截断）
        # 这里假设完全匹配，或者需要在外部处理 ID 匹配
        if seq_id not in fasta_lens:
            # 尝试做简单的 ID 清洗匹配
            clean_id = seq_id.split()[0]
            if clean_id in fasta_lens:
                total_len = fasta_lens[clean_id]
            else:
                continue # 找不到对应序列长度，跳过
        else:
            total_len = fasta_lens[seq_id]
            
        # 1. 提取所有 Period=1 的区间 (PolyA/T)
        poly_intervals = group[group['Period'] == 1][['Start', 'End']].values.tolist()
        poly_cov_len = calculate_merged_coverage(poly_intervals)
        
        # 2. 提取所有 Period > 1 的区间 (卫星/微卫星/复杂重复)
        complex_intervals = group[group['Period'] > 1][['Start', 'End']].values.tolist()
        complex_cov_len = calculate_merged_coverage(complex_intervals)
        
        # 3. 提取所有区间
        all_intervals = group[['Start', 'End']].values.tolist()
        total_rep_len = calculate_merged_coverage(all_intervals)
        
        max_period = group['Period'].max()
        max_score = group['Score'].max()
        
        summary_data.append({
            'SeqID': seq_id,
            'SeqLength': total_len,
            'PolyA_Cov_Len': poly_cov_len,
            'Complex_Cov_Len': complex_cov_len, # 关键指标：复杂重复长度
            'Total_Rep_Len': total_rep_len,
            'PolyA_Ratio': poly_cov_len / total_len,
            'Complex_Ratio': complex_cov_len / total_len, # 关键指标：复杂重复比例
            'Total_Rep_Ratio': total_rep_len / total_len,
            'Max_Period': max_period,
            'Max_Score': max_score
        })
        
    return pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser(description="利用 ULTRA 结果清洗 SINE 训练数据 (V2)")
    parser.add_argument("--input_fa", required=True, help="正样本 FASTA")
    parser.add_argument("--ultra_out", required=True, help="ULTRA 原始输出文件")
    parser.add_argument("--clean_fa", required=True, help="输出：清洗后的正样本")
    parser.add_argument("--bad_fa", required=True, help="输出：被剔除的样本 (作为负样本)")
    parser.add_argument("--report_csv", required=True, help="输出：详细诊断报告")
    
    # 阈值设置
    parser.add_argument("--max_complex_ratio", type=float, default=0.3, 
                        help="复杂重复(Period>1)最大允许比例。默认 0.3 (30%)")
    parser.add_argument("--max_total_ratio", type=float, default=0.5, 
                        help="总重复(含PolyA)最大允许比例。默认 0.5")
    
    args = parser.parse_args()
    
    # 1. 读取 FASTA 获取长度
    print("[INFO] 读取 FASTA 序列长度...")
    fasta_lens = {}
    fasta_records = {} # 缓存 SeqRecord 以便写入
    for record in SeqIO.parse(args.input_fa, "fasta"):
        # ULTRA 通常输出 ID 的第一部分 (空格前)
        sid = record.id.split()[0]
        sid = sid.split('(')[0]
        print(sid)
        # 有时候 ULTRA 输出的 ID 包含 | 或 : 等符号，需要确保一致性
        # 这里使用原始 ID 的第一段作为 key
        fasta_lens[sid] = len(record.seq)
        fasta_records[sid] = record
    print(f"  - 加载了 {len(fasta_lens)} 条序列")

    # 2. 解析 ULTRA
    print("[INFO] 解析 ULTRA 输出...")
    df_ultra = parse_ultra_output(args.ultra_out)
    print(f"  - 提取到 {len(df_ultra)} 条重复记录")
    
    # 3. 汇总分析
    print("[INFO] 汇总计算覆盖度...")
    df_stats = analyze_repeats(df_ultra, fasta_lens)
    
    if df_stats.empty:
        print("[WARN] 没有匹配的统计信息，可能 ULTRA ID 与 FASTA ID 不匹配，或没有检测到重复。")
        # 直接复制原文件
        with open(args.clean_fa, 'w') as f:
            SeqIO.write(fasta_records.values(), f, "fasta")
        return

    # 4. 判定好坏
    # 逻辑：
    # Keep = True 如果:
    #   Complex_Ratio < 0.3  (不能有太长的卫星/微卫星)
    #   AND
    #   Total_Rep_Ratio < 0.85 (不能整条序列都是 PolyA)
    
    def is_bad(row):
        if row['Complex_Ratio'] > args.max_complex_ratio:
            return True, "High_Complex_Repeat"
        if row['Total_Rep_Ratio'] > args.max_total_ratio:
            return True, "High_Total_Repeat"
        return False, "Pass"

    df_stats[['Is_Bad', 'Reason']] = df_stats.apply(lambda x: pd.Series(is_bad(x)), axis=1)
    
    # 保存报告
    df_stats.sort_values('Complex_Ratio', ascending=False).to_csv(args.report_csv, index=False)
    print(f"[INFO] 诊断报告已保存: {args.report_csv}")
    
    # 5. 分流写入 FASTA
    bad_ids = set(df_stats[df_stats['Is_Bad'] == True]['SeqID'])
    # 注意：还有一些序列可能根本没被 ULTRA 检测出重复，这些默认是 Good
    
    cnt_clean = 0
    cnt_bad = 0
    
    with open(args.clean_fa, "w") as f_clean, open(args.bad_fa, "w") as f_bad:
        for sid, record in fasta_records.items():
            if sid in bad_ids:
                # 写入 Bad
                # 稍微改下 ID 标记原因
                reason = df_stats.loc[df_stats['SeqID']==sid, 'Reason'].values[0]
                record.id = f"{record.id}_REPEAT_{reason}"
                record.description = ""
                SeqIO.write(record, f_bad, "fasta")
                cnt_bad += 1
            else:
                # 写入 Clean
                SeqIO.write(record, f_clean, "fasta")
                cnt_clean += 1
                
    print("\n[RESULT] 过滤完成")
    print(f"  - 保留 (Clean): {cnt_clean}")
    print(f"  - 剔除 (Bad):   {cnt_bad} -> 请将此文件加入负样本训练！")
    print(f"  - 输出文件: {args.clean_fa}, {args.bad_fa}")

if __name__ == "__main__":
    main()


# python tools/filter_training_data_by_ultra.py \
#     --input_fa  "results/predictions/rice_pred/rice_pred_predicted_SINEs.fa"    \
#     --ultra_out "/homeb/xuminghua/tools/ULTRA/pred_SINES_repeat.txt"    \
#     --clean_fa  "results/predictions/rice_pred/rice_pred_predicted_SINEs.clean.fa"  \
#     --bad_fa results/data_split/hard_negatives_repeats.fa \
#     --report_csv diagnosis_repeats.csv

