#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_predictions_post_process.py
==================================
预测后处理脚本：结合模型预测结果与 Motif 检测结果。

功能：
1. 读取模型预测的 TSV。
2. 读取 Motif 检测的 TSV。
3. 应用"一票否决"规则：如果模型说是 SINE，但 Motif 脚本说完全没特征(NO & 0-tail & No-Box)，则过滤掉。
4. 输出过滤后的 TSV 和 FASTA。
"""

import pandas as pd
import os
import argparse
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser(description="SINE 预测结果后处理过滤")
    # 默认路径设置为你当前的路径，方便直接运行
    parser.add_argument("--pred_tsv", default='results/predictions/rice_pred/rice_pred_predictions.tsv', help="模型预测的TSV")
    parser.add_argument("--pred_fasta", default='results/predictions/rice_pred/rice_pred_predicted_SINEs.fa', help="模型预测生成的FASTA")
    parser.add_argument("--motif_tsv", default='results/predictions/rice_pred/motifs.motifs.tsv', help="Motif检测结果TSV")
    parser.add_argument("--out_prefix", default='results/predictions/rice_pred/final_filtered', help="输出文件前缀")
    
    args = parser.parse_args()

    # 检查文件
    for f in [args.pred_tsv, args.pred_fasta, args.motif_tsv]:
        if not os.path.exists(f):
            print(f"❌ 错误: 文件不存在: {f}")
            return

    print("=== 开始加载数据 ===")
    
    # 1. 读取数据
    try:
        pred_df = pd.read_csv(args.pred_tsv, sep='\t')
        motif_df = pd.read_csv(args.motif_tsv, sep='\t')
        print(f"预测记录: {len(pred_df)} | Motif记录: {len(motif_df)}")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 准备 Motif 信息
    try:
        # 构建 ID: chrom:start-end(strand)
        motif_df['id'] = motif_df.apply(
            lambda x: f"{x['chrom']}:{x['original_start']}-{x['original_end']}({x['strand']})", 
            axis=1
        )
        
        # 提取关键列
        cols_needed = ['id', 'detection_status', 'polyA_len', 'A_pos', 'B_pos']
        # 填充缺失列以防万一
        for c in cols_needed:
            if c not in motif_df.columns: motif_df[c] = -1 if 'pos' in c or 'len' in c else 'NO'
            
        motif_info = motif_df[cols_needed].set_index('id')
        
    except Exception as e:
        print(f"❌ 处理 Motif 数据失败: {e}")
        return

    # 3. 合并数据 (Left Join)
    merged = pred_df.merge(motif_info, on='id', how='left')
    
    # 填充合并后产生的 NaN
    merged['detection_status'] = merged['detection_status'].fillna('NO')
    merged['polyA_len'] = pd.to_numeric(merged['polyA_len'], errors='coerce').fillna(0)
    merged['A_pos'] = pd.to_numeric(merged['A_pos'], errors='coerce').fillna(-1)
    merged['B_pos'] = pd.to_numeric(merged['B_pos'], errors='coerce').fillna(-1)

    # 4. 应用核心过滤规则
    print("\n=== 应用过滤规则 ===")
    
    def filter_rule(row):
        if row['prediction'] == 'non-SINE':
            return 'non-SINE'
        
        # 获取信息
        status = str(row.get('detection_status', 'NO'))
        poly_len = float(row.get('polyA_len', 0))
        prob_sine = float(row.get('prob_SINE', 0.0)) # 假设预测结果里有这一列
        
        # 1. 结构验证通过 -> 保留
        if status in ['HIGH_CONF', 'MED_CONF', 'LOW_CONF']:
            return 'SINE'
        
        # 2. 结构没通过(NO)，但有尾巴 -> 保留 (可能是 TSD 没找对)
        if poly_len > 5:
            return 'SINE'
            
        # 3. 结构全无，但模型置信度极高 -> 保留 (可能是古老/截断 SINE)
        # 阈值可以根据你的需要调整，比如 0.98 或 0.99
        if prob_sine > 0.99:
            return 'SINE'
            
        # 4. 既没结构，模型也不太确定 -> 杀掉
        return 'non-SINE (Filtered)'

    merged['final_prediction'] = merged.apply(filter_rule, axis=1)

    # 5. 统计
    initial_sine = len(merged[merged['prediction'] == 'SINE'])
    final_sine = len(merged[merged['final_prediction'] == 'SINE'])
    filtered_count = initial_sine - final_sine
    
    print(f"模型初步预测 SINE: {initial_sine}")
    print(f"规则过滤掉 (FP)  : {filtered_count}")
    print(f"最终保留 SINE    : {final_sine}")
    if initial_sine > 0:
        print(f"过滤比例: {filtered_count/initial_sine:.2%}")

    # 6. 保存 TSV
    out_tsv = f"{args.out_prefix}.tsv"
    merged.to_csv(out_tsv, sep='\t', index=False)
    print(f"\n✅ 最终 TSV 已保存: {out_tsv}")

    # 7. 生成过滤后的 FASTA
    # 我们需要根据 filtered result 里的 ID，去原始 fasta 里捞出对应的序列
    print("\n=== 生成最终 FASTA 文件 ===")
    
    # 获取最终判定为 SINE 的 ID 集合
    valid_ids = set(merged[merged['final_prediction'] == 'SINE']['id'])
    
    out_fasta = f"{args.out_prefix}.fa"
    keep_count = 0
    
    with open(out_fasta, "w") as f_out:
        # 读取原始预测的 FASTA (注意：原始 FASTA ID 可能带有额外标签，如 #SINE#0.99)
        # 我们主要匹配 ID 的前缀部分
        for record in SeqIO.parse(args.pred_fasta, "fasta"):
            # 解析 ID：通常是 "NC_xxx:100-200(+)" 或者是 "NC_xxx:100-200(+)#SINE#..."
            # 我们取 # 之前的部分作为 key
            seq_id_key = record.id.split('#')[0]
            
            if seq_id_key in valid_ids:
                # 这是一个真 SINE
                SeqIO.write(record, f_out, "fasta")
                keep_count += 1
    
    print(f"✅ 最终 FASTA 已保存: {out_fasta}")
    print(f"   共写入序列: {keep_count} 条")
    
    # 双重检查
    if keep_count != final_sine:
        print(f"⚠️ 警告: TSV保留数 ({final_sine}) 与 FASTA写入数 ({keep_count}) 不一致。")
        print("   原因可能是 FASTA ID 与 TSV ID 格式不完全匹配，请检查 split('#') 逻辑。")

if __name__ == "__main__":
    main()