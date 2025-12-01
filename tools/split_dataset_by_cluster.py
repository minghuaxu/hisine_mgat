#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset_by_cluster.py
===========================
使用 CD-HIT-EST 进行聚类划分，防止数据泄露。

逻辑：
1. 100% 去重：完全相同的序列视为同一实体。
2. 80% 聚类：使用 CD-HIT-EST 将相似序列聚类。
3. 按簇划分：保证同一簇的序列只会同时出现在训练集或验证集。
"""

import argparse
import subprocess
import os
import sys
import random
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict

def check_cdhit_installed():
    """检查 cd-hit-est 是否可用"""
    from shutil import which
    if which("cd-hit-est") is None:
        print("[ERROR] 找不到 'cd-hit-est' 命令。请先安装: conda install -c bioconda cd-hit")
        sys.exit(1)
    else:
        print("[INFO] cd-hit-est 已安装。")

def run_cdhit(input_fasta, output_prefix, threshold=0.8, threads=8):
    """
    调用 cd-hit-est 进行聚类
    
    参数:
        -c: 相似度阈值 (0.8)
        -n: word size (0.8时推荐为5)
        -d: 0 (在.clstr文件中使用完整header，不截断)
        -M: 内存限制 (MB)
    """
    output_rep = f"{output_prefix}_cdhit.fa"
    
    # 根据阈值自动调整 word size (-n)
    # CD-HIT 官方建议: 0.9-1.0 -> 5, 0.8-0.9 -> 5, 0.7-0.8 -> 4
    word_size = 5
    if threshold < 0.8:
        word_size = 4
        
    cmd = [
        "cd-hit-est",
        "-i", input_fasta,
        "-o", output_rep,
        "-c", str(threshold),
        "-n", str(word_size),
        "-d", "0",  # 关键：使用完整header，方便后续解析
        "-T", str(threads),
        "-M", "16000" # 16GB RAM limit
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] CD-HIT 运行失败")
        sys.exit(1)
        
    return f"{output_rep}.clstr"

def parse_clstr_file(clstr_path):
    """
    解析 CD-HIT .clstr 文件
    返回: clusters = [ [id1, id2...], [id3...], ... ]
    """
    clusters = []
    current_cluster = []
    
    print(f"[INFO] 解析聚类结果: {clstr_path}")
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
            else:
                # 解析行，例如: 
                # 0	200nt, >Unique_ID_1... *
                # 1	198nt, >Unique_ID_2... at 95.00%
                try:
                    # 提取 > 之后的内容
                    part_after_gt = line.split('>', 1)[1]
                    # 提取 ID (直到遇到... 或空格)
                    # CD-HIT 的 .clstr 文件中 ID 后面通常跟着 "..."
                    seq_id = part_after_gt.split('...')[0].strip()
                    current_cluster.append(seq_id)
                except IndexError:
                    print(f"[WARN] 无法解析行: {line}")
                    continue
                    
    if current_cluster:
        clusters.append(current_cluster)
        
    return clusters

def main():
    parser = argparse.ArgumentParser(description="使用CD-HIT按簇划分训练/验证集")
    parser.add_argument("--input_fasta", required=True, help="总FASTA文件")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例 (默认0.2)")
    parser.add_argument("--threshold", type=float, default=0.8, help="聚类相似度阈值 (默认0.8)")
    parser.add_argument("--threads", type=int, default=8, help="线程数")
    
    args = parser.parse_args()
    
    check_cdhit_installed()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # -----------------------------------------------------------
    # 1. 读取序列并进行 100% 精确去重 (预处理)
    # CD-HIT 也能去重，但为了 ID 管理方便，我们自己先做一步映射
    # -----------------------------------------------------------
    print("[INFO] 1. 读取并去重原始序列...")
    
    # seq_map: 序列字符串 -> [原始ID列表]
    # 我们只拿每组相同序列的第一个ID去给CD-HIT聚类
    seq_to_ids = defaultdict(list)
    unique_reps = {} # unique_id -> sequence_string
    
    count_total = 0
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        count_total += 1
        seq_str = str(record.seq).upper()
        
        # 如果是第一次见到这个序列
        if seq_str not in seq_to_ids:
            unique_reps[record.id] = seq_str
        
        seq_to_ids[seq_str].append(record.id)
    
    print(f"  原始序列总数: {count_total}")
    print(f"  唯一序列数量: {len(unique_reps)} (将用于聚类)")
    
    # 写入去重后的临时FASTA
    temp_unique_fasta = os.path.join(args.out_dir, "unique_for_clustering.fa")
    with open(temp_unique_fasta, "w") as f:
        for uid, seq in unique_reps.items():
            f.write(f">{uid}\n{seq}\n")
            
    # -----------------------------------------------------------
    # 2. 运行 CD-HIT-EST
    # -----------------------------------------------------------
    print(f"[INFO] 2. 运行 CD-HIT-EST (阈值={args.threshold})...")
    clstr_file = run_cdhit(
        temp_unique_fasta, 
        os.path.join(args.out_dir, "cdhit_out"), 
        threshold=args.threshold, 
        threads=args.threads
    )
    
    # -----------------------------------------------------------
    # 3. 解析聚类并划分
    # -----------------------------------------------------------
    print("[INFO] 3. 解析聚类并划分数据集...")
    
    # 这里的 clusters 包含的是 unique_reps 的 ID
    clusters = parse_clstr_file(clstr_file)
    print(f"  生成簇数量: {len(clusters)}")
    
    # 打乱簇的顺序
    random.seed(42)
    random.shuffle(clusters)
    
    # 计算划分界限
    split_idx = int(len(clusters) * (1 - args.val_ratio))
    train_clusters = clusters[:split_idx]
    val_clusters = clusters[split_idx:]
    
    # -----------------------------------------------------------
    # 4. 还原 ID 并写入结果
    # -----------------------------------------------------------
    print("[INFO] 4. 还原所有ID并保存...")
    
    final_train_ids = []
    final_val_ids = []
    
    # 辅助函数：根据 rep_id 找回所有原始 ID
    def expand_ids(rep_id_list):
        expanded = []
        for rep_id in rep_id_list:
            # 找到对应的序列
            seq = unique_reps.get(rep_id)
            if seq:
                # 找到该序列对应的所有原始ID
                all_ids = seq_to_ids[seq]
                expanded.extend(all_ids)
            else:
                # 防御性编程，理论上不应进入这里
                print(f"[WARN] ID {rep_id} 在原始映射中未找到")
        return expanded

    # 扩展训练集
    for cluster in train_clusters:
        final_train_ids.extend(expand_ids(cluster))
        
    # 扩展验证集
    for cluster in val_clusters:
        final_val_ids.extend(expand_ids(cluster))
        
    print(f"  最终训练集序列数: {len(final_train_ids)}")
    print(f"  最终验证集序列数: {len(final_val_ids)}")
    print(f"  实际验证集比例: {len(final_val_ids)/count_total:.2%}")
    
    # 保存文件
    train_out = os.path.join(args.out_dir, "train_ids.txt")
    val_out = os.path.join(args.out_dir, "val_ids.txt")
    
    with open(train_out, "w") as f:
        f.write("\n".join(final_train_ids) + "\n")
        
    with open(val_out, "w") as f:
        f.write("\n".join(final_val_ids) + "\n")
        
    print(f"[SUCCESS] 完成! ID列表已保存至:")
    print(f"  - {train_out}")
    print(f"  - {val_out}")
    print("请修改训练脚本以加载这些文件。")

if __name__ == "__main__":
    main()