#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_contamination.py
=======================
清洗正样本：剔除那些能比对到非 SINE 库 (DNA, LTR, TIR) 的序列。

逻辑：
1. 将输入的 SINE 候选序列 (Query) 比对到 Negative Reference (Target)。
2. 如果比对覆盖度 > 阈值 (如 70%)，则视为污染 (MITEs, fragments, rDNA)。
3. 输出清洗后的 FASTA 和 TSV。
"""

import argparse
import subprocess
import os
import pandas as pd
from Bio import SeqIO
from pathlib import Path

def run_minimap2(query_fa, ref_fa, threads=4):
    """运行 minimap2 获取比对结果 (PAF 格式)"""
    cmd = [
        "minimap2",
        "-x", "asm5",       # 假设差异较小
        "-t", str(threads),
        ref_fa,             # Target: 污染库
        query_fa            # Query: SINE 候选
    ]
    
    # 运行并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def parse_paf_and_get_bad_ids(paf_content, coverage_thr=0.6):
    """
    解析 PAF 格式，找出污染序列的 ID。
    PAF 格式 col 0: query_name, col 1: query_len, col 2: query_start, col 3: query_end
    """
    bad_ids = set()
    
    for line in paf_content.splitlines():
        if not line: continue
        cols = line.split('\t')
        if len(cols) < 12: continue
        
        q_name = cols[0]
        q_len = int(cols[1])
        q_start = int(cols[2])
        q_end = int(cols[3])
        
        aln_len = q_end - q_start
        coverage = aln_len / q_len
        
        # 如果覆盖度超过阈值，视为污染
        # 例如：这个 SINE 候选有 60% 以上的长度匹配到了 LTR 或 DNA 转座子
        if coverage > coverage_thr:
            bad_ids.add(q_name)
            
    return bad_ids

def main():
    parser = argparse.ArgumentParser(description="清洗 SINE 正样本中的污染")
    parser.add_argument("--input_fa", required=True, help="原始正样本 FASTA")
    parser.add_argument("--input_tsv", required=True, help="原始正样本 TSV")
    parser.add_argument("--neg_refs", nargs='+', required=True, help="污染源参考库列表 (DNA.fa LTR.fa ...)")
    parser.add_argument("--out_prefix", required=True, help="输出文件前缀")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cov_thr", type=float, default=0.6, help="污染覆盖度阈值 (默认 0.6)")
    
    args = parser.parse_args()
    
    # 1. 合并所有污染源库到一个临时文件
    temp_ref = f"{args.out_prefix}.temp_neg_ref.fa"
    print(f"[INFO] 合并污染库到 {temp_ref} ...")
    with open(temp_ref, 'w') as outfile:
        for ref in args.neg_refs:
            if Path(ref).exists():
                with open(ref, 'r') as infile:
                    outfile.write(infile.read())
            else:
                print(f"[WARN] 污染库不存在: {ref}")
    
    # 2. 运行 Minimap2
    print(f"[INFO] 正在比对查重 (Coverage Threshold: {args.cov_thr})...")
    paf_out = run_minimap2(args.input_fa, temp_ref, threads=args.threads)
    
    # 3. 解析并获取黑名单
    bad_ids = parse_paf_and_get_bad_ids(paf_out, coverage_thr=args.cov_thr)
    print(f"[INFO] 发现 {len(bad_ids)} 条污染序列 (MITEs/LTR-fragments/rDNA)。")
    
    # 4. 过滤并写入 FASTA
    clean_fasta_path = f"{args.out_prefix}.fa"
    total_seqs = 0
    kept_seqs = 0
    
    with open(clean_fasta_path, "w") as f:
        for record in SeqIO.parse(args.input_fa, "fasta"):
            total_seqs += 1
            if record.id not in bad_ids:
                SeqIO.write(record, f, "fasta")
                kept_seqs += 1
    
    # 5. 过滤并写入 TSV (如果有 TSV 的话)
    clean_tsv_path = f"{args.out_prefix}.tsv"
    if Path(args.input_tsv).exists():
        df = pd.read_csv(args.input_tsv, sep='\t')
        # 假设 FASTA ID 对应 TSV 的某种构建逻辑
        # 这里假设 FASTA ID 就是 chrom:start-end(strand) 这种格式，或者 TSV 转换来的
        # 通常 extract_positives 生成的 FASTA ID 比较复杂，这里我们需要谨慎
        
        # 更好的方法：重新遍历 DataFrame，重构 ID 进行比对
        # 你的 extract_positives 生成的 FASTA ID 通常是 ">chrom:start-end(strand)"
        
        # 让我们通过 DataFrame 的行来过滤
        # 先构建一个 ID 列
        df['temp_id'] = df.apply(lambda row: f"{row['chrom']}:{row['start']}-{row['end']}({row['strand']})", axis=1)
        
        # 过滤
        df_clean = df[~df['temp_id'].isin(bad_ids)].drop(columns=['temp_id'])
        df_clean.to_csv(clean_tsv_path, sep='\t', index=False)
    
    # 清理临时文件
    if Path(temp_ref).exists():
        Path(temp_ref).unlink()
        
    print(f"[RESULT] 清洗完成: {total_seqs} -> {kept_seqs} (剔除了 {total_seqs - kept_seqs} 条)")
    print(f"  输出: {clean_fasta_path}")
    print(f"  输出: {clean_tsv_path}")

if __name__ == "__main__":
    main()