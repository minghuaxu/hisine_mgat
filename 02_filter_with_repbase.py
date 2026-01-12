#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggressive_clean_with_repbase_v3.py
===================================
版本 V3 更新说明：
1. [新增] 标记机制：正样本降级为负样本时，修改其 source_type 以便区分 (Converted_...)。
2. [新增] 保护机制：source_type 为 'Hard_rRNA' 的样本被视为已知难例，全部保留，不参与下采样。
3. 基础功能：RepBase 双重 80% 覆盖度验证 + 长度安全检查 + 简单负样本下采样。

处理流程：
1. 读取数据，确保有 source_type 列。
2. 对原始正样本进行 BLAST 验证。
   - 失败者：Label 1 -> 0, source_type -> 'Converted_RepBase_Fail'
3. 对剩余正样本进行长度检查 (>600bp)。
   - 失败者：Label 1 -> 0, source_type -> 'Converted_Too_Long'
4. 数据分层与合并：
   - 最终正样本 (Keep All)
   - 转换后的难例负样本 (Converted, Keep All)
   - Hard_rRNA (Keep All)
   - 普通简单负样本 (Sampled to max limit)
"""

import pandas as pd
import argparse
from pathlib import Path
import logging
import subprocess
import tempfile
import os
import numpy as np
from Bio import SeqIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_blast_install():
    """检查是否安装了 BLAST+"""
    try:
        subprocess.run(["blastn", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["makeblastdb", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("未找到 NCBI BLAST+ 工具。请确保安装了 blastn 和 makeblastdb。")
        exit(1)

def prepare_repbase_db(repbase_fasta, db_dir):
    """准备 BLAST 数据库"""
    logger.info(f"正在读取 RepBase 文件: {repbase_fasta}")
    repbase_lengths = {}
    try:
        for record in SeqIO.parse(repbase_fasta, "fasta"):
            repbase_lengths[record.id] = len(record.seq)
    except Exception as e:
        logger.error(f"读取 RepBase FASTA 失败: {e}")
        exit(1)
    
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    db_name = Path(repbase_fasta).stem
    db_path = os.path.join(db_dir, db_name)
    
    if not os.path.exists(db_path + ".nhr"):
        logger.info("构建 RepBase BLAST 数据库...")
        cmd = ["makeblastdb", "-in", repbase_fasta, "-dbtype", "nucl", "-out", db_path, "-logfile", os.devnull]
        subprocess.run(cmd, check=True)
    else:
        logger.info("使用现有 BLAST 数据库。")
        
    return db_path, repbase_lengths

def run_blast_validation(pos_df, db_path, repbase_lengths, coverage_threshold=0.8):
    """运行 BLAST 并返回通过验证的 UID 集合"""
    validated_uids = set()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_query:
        query_fasta_path = tmp_query.name
        valid_seqs = pos_df.dropna(subset=['seq'])
        for _, row in valid_seqs.iterrows():
            tmp_query.write(f">{row['uid']}\n{row['seq']}\n")
    
    logger.info(f"BLAST: 创建查询文件 {len(valid_seqs)} 条序列。")
    
    try:
        # qseqid sseqid length qlen
        cmd = [
            "blastn", "-query", query_fasta_path, "-db", db_path,
            "-outfmt", "6 qseqid sseqid length qlen",
            "-evalue", "1e-5", "-perc_identity", "70", "-num_threads", "8"
        ]
        blast_process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        blast_results = blast_process.stdout.strip().split('\n')
        
    except subprocess.CalledProcessError as e:
        logger.error(f"BLASTN 运行失败: {e.stderr}")
        os.unlink(query_fasta_path)
        exit(1)
    
    logger.info(f"BLAST: 获得 {len(blast_results)} 条命中，正在筛选...")
    
    count_passed = 0
    for line in blast_results:
        if not line: continue
        parts = line.split('\t')
        qseqid = parts[0]
        sseqid = parts[1]
        align_len = int(parts[2])
        q_len = int(parts[3])
        s_len = repbase_lengths.get(sseqid)
        
        if s_len is None: continue
            
        q_cov = align_len / q_len if q_len > 0 else 0
        s_cov = align_len / s_len if s_len > 0 else 0
        
        if q_cov >= coverage_threshold and s_cov >= coverage_threshold:
            if qseqid not in validated_uids:
                validated_uids.add(qseqid)
                count_passed += 1
                
    logger.info(f"BLAST: {count_passed} 个样本通过双重 {coverage_threshold*100}% 验证。")
    os.unlink(query_fasta_path)
    return validated_uids

def aggressive_clean_v3(input_csv, output_csv, repbase_fasta, temp_dir='./tmp_blast', coverage=0.8, max_easy_negatives=50000):
    check_blast_install()
    
    logger.info(f"读取原始数据: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 1. 预处理列
    if 'source_type' not in df.columns:
        df['source_type'] = 'unknown'
        logger.warning("未发现 'source_type' 列，已创建默认值 'unknown'")
    else:
        # 填充 NaN
        df['source_type'] = df['source_type'].fillna('unknown')

    if 'uid' not in df.columns:
         df['uid'] = df.apply(lambda row: f"{row['chrom']}:{row['start']}-{row['end']}", axis=1)

    # 备份原始标签
    if 'original_label' not in df.columns:
        df['original_label'] = df['label']

    original_pos = df[df['label'] == 1].shape[0]
    logger.info(f"初始正样本数: {original_pos}")

    # --- 阶段 1: 激进过滤正样本 (BLAST + Length) ---
    if original_pos > 0:
        db_path, repbase_lengths = prepare_repbase_db(repbase_fasta, temp_dir)
        pos_df = df[df['label'] == 1].copy()
        validated_uids = run_blast_validation(pos_df, db_path, repbase_lengths, coverage)
        
        # [逻辑] 降级未通过验证的
        # 且更新 source_type 为 'Converted_RepBase_Fail'
        downgrade_mask = (df['label'] == 1) & (~df['uid'].isin(validated_uids))
        num_blast_downgraded = downgrade_mask.sum()
        
        if num_blast_downgraded > 0:
            df.loc[downgrade_mask, 'label'] = 0
            df.loc[downgrade_mask, 'source_type'] = 'Converted_RepBase_Fail'
            logger.info(f"BLAST清洗: {num_blast_downgraded} 个样本降级并标记为 'Converted_RepBase_Fail'")

        # [逻辑] 安全检查 (>600bp)
        if 'core_len' not in df.columns:
             df['core_len'] = df['end'] - df['start']
        
        # 注意：只检查依然是 label=1 的样本
        safety_mask = (df['label'] == 1) & (df['core_len'] > 600)
        num_safety_downgraded = safety_mask.sum()
        
        if num_safety_downgraded > 0:
            df.loc[safety_mask, 'label'] = 0
            df.loc[safety_mask, 'source_type'] = 'Converted_Too_Long'
            logger.info(f"长度清洗: {num_safety_downgraded} 个样本降级并标记为 'Converted_Too_Long'")
            
    else:
        logger.warning("无正样本，跳过 BLAST。")

    # --- 阶段 2: 数据分层与下采样 ---
    logger.info("--- 开始数据分层处理 ---")
    
    # A. 最终正样本 (Keep All)
    final_pos_df = df[df['label'] == 1].copy()
    
    # B. 转换后的难例负样本 (Converted, Keep All)
    # 特征：label=0, 但 original_label=1 (或者检查 source_type 是否包含 Converted)
    converted_neg_df = df[df['source_type'].str.startswith('Converted_')].copy()
    
    # C. 特殊难例 rRNA (Hard_rRNA, Keep All)
    # 特征：label=0, source_type='Hard_rRNA'
    # 注意：如果原本 Hard_rRNA 标了 1 且被 BLAST 降级了，它会归入 B 类，这没问题。
    # 这里主要捕获原始就是负样本的 rRNA
    rRNA_df = df[(df['label'] == 0) & (df['source_type'] == 'Hard_rRNA')].copy()
    
    # D. 普通简单负样本 (Easy Negatives, Subsample)
    # 特征：label=0, 不是 Converted, 不是 Hard_rRNA
    easy_neg_mask = (df['label'] == 0) & \
                    (~df['source_type'].str.startswith('Converted_')) & \
                    (df['source_type'] != 'Hard_rRNA')
    easy_neg_df = df[easy_neg_mask].copy()
    
    # 统计
    n_pos = len(final_pos_df)
    n_conv = len(converted_neg_df)
    n_rRNA = len(rRNA_df)
    n_easy = len(easy_neg_df)
    
    logger.info(f"数据分层统计:")
    logger.info(f"  1. 最终正样本 (SINE): {n_pos}")
    logger.info(f"  2. 转化难例 (Converted): {n_conv} (全部保留)")
    logger.info(f"  3. rRNA难例 (Hard_rRNA): {n_rRNA} (全部保留)")
    logger.info(f"  4. 简单负样本 (Easy): {n_easy} (待采样)")
    
    # 执行下采样
    if max_easy_negatives is not None and n_easy > max_easy_negatives:
        logger.info(f"正在对简单负样本采样: {n_easy} -> {max_easy_negatives}")
        easy_neg_df = easy_neg_df.sample(n=max_easy_negatives, random_state=42)
    else:
        logger.info("简单负样本数量未超标，全部保留。")

    # --- 阶段 3: 合并 ---
    final_df = pd.concat([final_pos_df, converted_neg_df, rRNA_df, easy_neg_df], ignore_index=True)
    
    # 打乱
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 清理辅助列 (保留 source_type 以便后续分析)
    cols_to_drop = ['uid', 'core_len', 'original_label']
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore')

    logger.info("================ 清洗 V3 总结 ================")
    logger.info(f"输出文件: {output_csv}")
    logger.info(f"总样本数: {len(final_df)}")
    logger.info(f"  - Label 1: {len(final_df[final_df['label']==1])}")
    logger.info(f"  - Label 0: {len(final_df[final_df['label']==0])}")
    
    # 打印 source_type 分布，确认标记成功
    logger.info("Label 0 的来源分布:")
    logger.info(final_df[final_df['label']==0]['source_type'].value_counts().to_string())

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="激进清洗 V3: 标记降级样本 + 保护 rRNA + 简单负样本下采样")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--repbase_fasta", required=True)
    parser.add_argument("--temp_dir", default="./tmp_blast_v3")
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--max_easy_negatives", type=int, default=5000)
    
    args = parser.parse_args()
    
    aggressive_clean_v3(
        args.input_csv, 
        args.output_csv, 
        args.repbase_fasta, 
        args.temp_dir, 
        args.coverage,
        args.max_easy_negatives
    )