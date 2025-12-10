#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_direct_blast.py
========================
直接评估脚本：将预测序列 BLAST 回 SINE 标准库。

目的：
1. 绕过基因组坐标，直接从序列相似性判断是否为 SINE。
2. 区分 "已知 SINE"、"潜在新 SINE" 和 "纯粹垃圾"。
3. 计算预测序列对标准序列的覆盖度。
"""

import argparse
import subprocess
import os
import pandas as pd
from Bio import SeqIO
from pathlib import Path

def run_blast(query_fa, db_fa, out_file, threads=8):
    """
    构建临时数据库并运行 BLASTN
    """
    db_prefix = "temp_sine_db"
    
    # 1. 建库
    print("[INFO] Building BLAST database...")
    cmd_db = f"makeblastdb -in {db_fa} -dbtype nucl -out {db_prefix}"
    subprocess.run(cmd_db, shell=True, check=True, stdout=subprocess.DEVNULL)
    
    # 2. 比对
    # -max_target_seqs 1: 只看最佳匹配
    # -outfmt 6: qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore
    print("[INFO] Running BLASTN...")
    cmd_blast = (
        f"blastn -query {query_fa} -db {db_prefix} "
        f"-out {out_file} -outfmt '6 qseqid sseqid pident length qlen slen qstart qend evalue bitscore' "
        f"-num_threads {threads} -max_target_seqs 1 -max_hsps 1"
    )
    subprocess.run(cmd_blast, shell=True, check=True)
    
    # 3. 清理数据库文件
    for f in os.listdir("."):
        if f.startswith(db_prefix):
            os.remove(f)

def analyze_results(blast_out, query_fa, output_csv):
    """
    分析 BLAST 结果
    """
    # 1. 读取所有查询序列ID
    all_queries = set()
    for record in SeqIO.parse(query_fa, "fasta"):
        # BLAST 可能会截断 ID，通常取空格前部分
        seq_id = record.id.split()[0]
        all_queries.add(seq_id)
    
    print(f"[INFO] Total Predicted Sequences: {len(all_queries)}")
    
    # 2. 读取 BLAST 结果
    # 列名对应 outfmt 6
    cols = ['qseqid', 'sseqid', 'pident', 'length', 'qlen', 'slen', 'qstart', 'qend', 'evalue', 'bitscore']
    try:
        df = pd.read_csv(blast_out, sep='\t', names=cols)
    except pd.errors.EmptyDataError:
        print("[ERROR] BLAST 结果为空！没有序列比对上参考库。")
        return

    # 3. 计算指标
    # Query Coverage: 预测序列有多少比例匹配上了参考库
    df['q_cov'] = (df['length'] / df['qlen']) * 100
    
    # Subject Coverage: 参考序列被还原了多少 (可选)
    df['s_cov'] = (df['length'] / df['slen']) * 100
    
    # 4. 分类判定
    # 这里的阈值可以根据需要调整
    # Strict Match: 覆盖度 > 80% 且 相似度 > 80%
    # Partial Match: 覆盖度 > 50%
    
    def classify(row):
        if row['q_cov'] >= 80 and row['pident'] >= 80:
            return "Known_SINE (High Conf)"
        elif row['q_cov'] >= 50 and row['pident'] >= 75:
            return "Known_SINE (Partial)"
        elif row['evalue'] < 1e-10:
            return "Potential_Homolog"
        else:
            return "Weak_Hit"

    df['classification'] = df.apply(classify, axis=1)
    
    # 5. 统计未比对上的序列 (Novel or Junk)
    mapped_ids = set(df['qseqid'])
    unmapped_ids = all_queries - mapped_ids
    
    # 6. 生成报告
    print("\n" + "="*60)
    print("BLAST Direct Evaluation Report")
    print("="*60)
    
    counts = df['classification'].value_counts()
    total_mapped = len(df)
    total_unmapped = len(unmapped_ids)
    
    print(f"Total Queries:      {len(all_queries)}")
    print(f"Mapped to Ref:      {total_mapped} ({total_mapped/len(all_queries):.1%})")
    print(f"  - High Conf:      {counts.get('Known_SINE (High Conf)', 0)}")
    print(f"  - Partial:        {counts.get('Known_SINE (Partial)', 0)}")
    print(f"  - Homolog:        {counts.get('Potential_Homolog', 0)}")
    print(f"  - Weak Hit:       {counts.get('Weak_Hit', 0)}")
    print("-" * 60)
    print(f"No Match (FP/Novel): {total_unmapped} ({total_unmapped/len(all_queries):.1%})")
    print("="*60)
    
    # 保存详细结果
    # 将未比对的也加进去方便查看
    unmapped_df = pd.DataFrame({'qseqid': list(unmapped_ids), 'classification': 'No_Match'})
    final_df = pd.concat([df, unmapped_df], ignore_index=True)
    
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Detailed report saved to: {output_csv}")
    
    # 导出 No Match 的 ID 列表，方便后续提取序列去跑 BlastN (NCBI) 查成分
    no_match_file = output_csv.replace(".csv", "_no_match_ids.txt")
    with open(no_match_file, 'w') as f:
        for uid in unmapped_ids:
            f.write(f"{uid}\n")
    print(f"[INFO] No-match IDs saved to: {no_match_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_fa", required=True, help="你的预测结果 FASTA (例如 final_filtered.fa)")
    parser.add_argument("--ref_fa", required=True, help="SINE 标准库 FASTA (RepBase/Dfam)")
    parser.add_argument("--out_prefix", default="blast_eval", help="输出前缀")
    parser.add_argument("--threads", type=int, default=8)
    
    args = parser.parse_args()
    
    temp_blast_out = f"{args.out_prefix}.blast6"
    
    # 1. 运行 BLAST
    run_blast(args.pred_fa, args.ref_fa, temp_blast_out, args.threads)
    
    # 2. 分析
    analyze_results(temp_blast_out, args.pred_fa, f"{args.out_prefix}_report.csv")

if __name__ == "__main__":
    main()

"""
python tools/evaluate_direct_blast.py \
    --pred_fa /homeb/xuminghua/hisine_classifier/results/predictions/rice_pred/final_filtered_noSINE.fa \
    --ref_fa /homeb/xuminghua/hisine_classifier/data/Dfam_RepBase/DNA.fa \
    --out_prefix results/predictions/rice_pred/direct_eval
"""
