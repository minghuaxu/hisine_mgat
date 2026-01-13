#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_prepare_dataset.py
=====================
功能：
1. 扫描所有生成的 TSV 文件 (正样本, Hard Neg, CDS Neg, Random Neg)。
2. 合并为一个巨大的 DataFrame。
3. 添加标签 (Label: 1=SINE, 0=Others) 和 来源类型 (Type)。
4. 过滤无效序列 (含 N 过多, 长度过短等)。
5. 按 8:1:1 随机切分为 Train/Val/Test 集合。
"""
import hashlib
import argparse
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from Bio.Seq import Seq

def load_and_label(pattern, label, source_type, base_dir):
    """
    递归查找文件并加载，添加标签
    """
    search_path = os.path.join(base_dir, "**", pattern)
    files = glob.glob(search_path, recursive=True)
    
    print(f"[Info] 正在加载 {source_type}: 找到 {len(files)} 个文件...")
    
    dfs = []
    for f in files:
        try:
            # 某些文件可能是空的，skip
            if os.path.getsize(f) == 0: continue
            
            df = pd.read_csv(f, sep='\t')
            if df.empty: continue
            
            # 统一列名 (防止某些文件列名不一致)
            # 必须包含 'seq' 列
            if 'seq' not in df.columns:
                print(f"[Warn] 文件 {f} 缺少 'seq' 列，跳过。")
                continue
                
            # 添加元数据
            df['label'] = label
            df['source_type'] = source_type
            df['filename'] = os.path.basename(f)
            
            # 只保留核心列
            cols = ['chrom', 'start', 'end', 'strand', 'seq', 'flank_left', 'flank_right', 'label', 'source_type']
            # 如果有其他列想保留可以加，这里只留核心
            dfs.append(df[cols])
        except Exception as e:
            print(f"[Error] 读取 {f} 失败: {e}")
            
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def get_split_group(chrom_name):
    """
    根据染色体名称的哈希值决定它属于哪个集合。
    返回: 'train', 'val', 或 'test'
    """
    # 使用 md5 哈希保证跨平台和跨运行的一致性
    chrom_str = str(chrom_name).strip()
    hash_val = int(hashlib.md5(chrom_str.encode('utf-8')).hexdigest(), 16)
    
    # 取模 100，生成 0-99 的整数
    mod_val = hash_val % 100
    
    # 8:1:1 划分
    if mod_val < 80:
        return 'train'
    elif mod_val < 90:
        return 'val'
    else:
        return 'test'
    
def filter_sequences(df, min_len=50, max_n_ratio=0.1):
    """清洗序列"""
    total = len(df)

    # 1. 侧翼序列空值过滤
    # notna() 过滤掉 Pandas 的 NaN (float类型)
    # str.len() > 0 过滤掉空字符串 ""
    mask_flank = (
        df['flank_left'].notna() & 
        df['flank_right'].notna() & 
        (df['flank_left'].astype(str).str.strip().str.len() > 0) & 
        (df['flank_right'].astype(str).str.strip().str.len() > 0)
    )
    
    # 1. 长度过滤
    mask_len = df['seq'].str.len() >= min_len
    
    # 2. N 含量过滤
    # 计算 N 的比例
    def get_n_ratio(s):
        return s.count('N') / len(s) if len(s) > 0 else 1.0
    
    mask_n = df['seq'].apply(get_n_ratio) <= max_n_ratio
    
    df_clean = df[mask_flank & mask_len & mask_n].copy()
    
    # 3. 统一大写
    df_clean['seq'] = df_clean['seq'].str.upper()
    df_clean['flank_left'] = df_clean['flank_left'].str.upper()
    df_clean['flank_right'] = df_clean['flank_right'].str.upper()
    
    print(f"[Filter] 过滤前: {total}, 过滤后: {len(df_clean)} (剔除短序列或N过多、侧翼序列为空)")
    return df_clean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="包含 results/sines 的根目录")
    parser.add_argument("--out_dir", required=True, help="训练集输出目录")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 加载各类数据
    # 正样本
    df_pos = load_and_label("*_v1_final.tsv", 1, "SINE", args.data_dir)
    
    # 负样本 - Hard (rRNA)
    df_hard = load_and_label("*_v1_hard_negative_rrna.tsv", 0, "Hard_rRNA", args.data_dir)
    
    # 负样本 - Medium (CDS)
    df_cds = load_and_label("*_neg_cds.tsv", 0, "Neg_CDS", args.data_dir)
    
    # 负样本 - Easy (Random)
    df_rand = load_and_label("*_neg_random.tsv", 0, "Neg_Random", args.data_dir)
    
    # 2. 合并
    df_all = pd.concat([df_pos, df_hard, df_cds, df_rand], ignore_index=True)
    print(f"\n[Summary] 原始数据总条数: {len(df_all)}")
    print(df_all['source_type'].value_counts())
    
    # 3. 清洗
    df_all = filter_sequences(df_all)
    
    # 4. 切分数据集 (Train/Val/Test = 8:1:1)

    # 4. 按染色体切分数据集 (Chromosome-based Split)
    print("\n[Split] 正在执行按染色体划分 (防止同源序列泄露)...")

    # 应用哈希函数分配组别
    df_all['split_group'] = df_all['chrom'].apply(get_split_group)

    train_df = df_all[df_all['split_group'] == 'train'].copy()
    val_df   = df_all[df_all['split_group'] == 'val'].copy()
    test_df  = df_all[df_all['split_group'] == 'test'].copy()

    # 移除辅助列
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['split_group'], inplace=True)
    
    # 打印统计信息，检查是否某个物种的数据量在某个集合中过少
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size:   {len(val_df)}")
    print(f"Test set size:  {len(test_df)}")

    # 可选：检查各集合的物种覆盖情况（防止某个物种的所有染色体都运气不好掉进了测试集）
    print(train_df['source_type'].value_counts())

    # stratify=df_all['label'] 保证验证集里正负比例和训练集一致
    # train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, shuffle=True, stratify=df_all['source_type'])
    # val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True, stratify=temp_df['source_type'])
    
    # 5. 保存
    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ 数据集制作完成！")
    print(f"Train: {len(train_df)} -> {train_path}")
    print(f"Val:   {len(val_df)}   -> {val_path}")
    print(f"Test:  {len(test_df)}  -> {test_path}")

if __name__ == "__main__":
    main()

"""
python -m scripts.04_prepare_dataset \
    --data_dir /homeb/xuminghua/sine_dataset/results/sines \
    --out_dir /homeb/xuminghua/sine_dataset/dataset_v1

[Summary] 原始数据总条数: 217649
source_type
Neg_Random    108000
Neg_CDS        54000
SINE           47395
Hard_rRNA       8254
Name: count, dtype: int64
[Filter] 过滤前: 217649, 过滤后: 217649 (剔除短序列或N过多)

✅ 数据集制作完成！
Train: 174119 -> /homeb/xuminghua/sine_dataset/dataset_v1/train.csv
Val:   21765   -> /homeb/xuminghua/sine_dataset/dataset_v1/val.csv
Test:  21765  -> /homeb/xuminghua/sine_dataset/dataset_v1/test.csv
"""