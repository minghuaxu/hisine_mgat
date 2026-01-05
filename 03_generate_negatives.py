#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_generate_negatives.py (Updated with Flanks)
==============================================
功能：生成 CDS 负样本和随机基因组背景负样本。
更新点：为负样本添加与正样本一致的 flank_left 和 flank_right。
"""

import argparse
import pandas as pd
import random
from Bio import SeqIO
from collections import defaultdict
import intervaltree

def load_intervals(tsv_path):
    """加载正样本区间 (禁区)"""
    tree = defaultdict(intervaltree.IntervalTree)
    if not tsv_path: return tree
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if df.empty: return tree
        for _, row in df.iterrows():
            chrom = str(row['chrom'])
            start = int(row['start'])
            end = int(row['end'])
            tree[chrom].addi(start, end)
    except Exception:
        pass
    return tree

def parse_gff_cds(gff_path):
    """解析 GFF 获取 CDS 区间"""
    cds_tree = defaultdict(intervaltree.IntervalTree)
    print(f"[Info] 解析 GFF CDS: {gff_path}")
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            if parts[2] == 'CDS':
                chrom = parts[0]
                start = int(parts[3]) - 1 # 0-based
                end = int(parts[4])
                cds_tree[chrom].addi(start, end)
    return cds_tree

def get_random_length():
    """生成符合 SINE 长度分布的随机长度 (50-300bp)"""
    l = int(random.gauss(150, 50))
    return max(50, min(400, l))

def get_flanked_sequence(genome, chrom, start, end, flank_size):
    """
    提取核心序列及其左右侧翼。
    如果侧翼超出染色体边界，则该样本无效 (返回 None)。
    """
    chrom_len = len(genome[chrom])
    
    # 边界检查
    if start < flank_size or end + flank_size > chrom_len:
        return None
    
    # 提取
    # 注意：Python切片是 [start:end)，不包含 end
    seq_core = str(genome[chrom][start:end].seq)
    
    seq_left = str(genome[chrom][start-flank_size : start].seq)
    seq_right = str(genome[chrom][end : end+flank_size].seq)
    
    # 简单的质量控制
    full_seq = seq_left + seq_core + seq_right
    if full_seq.count('N') / len(full_seq) > 0.1:
        return None
        
    return seq_core, seq_left, seq_right

def generate_seqs_from_intervals(genome, intervals, count, label, flank_size, forbid_tree=None):
    """从指定区间集合中随机切取序列 (带侧翼)"""
    seqs = []
    chroms = list(intervals.keys())
    attempts = 0
    max_attempts = count * 20
    
    while len(seqs) < count and attempts < max_attempts:
        attempts += 1
        chrom = random.choice(chroms)
        if not intervals[chrom]: continue
        
        chrom_len = len(genome[chrom])
        
        # 随机选一个区间
        iv = random.choice(list(intervals[chrom]))
        iv_len = iv.end - iv.begin
        target_len = get_random_length()
        if iv_len < target_len: continue

        # 2. 检查 GFF 记录的区间是否超出了 FASTA 序列的长度
        # 如果 iv.end 超过了 chrom_len，缩小可用区间
        effective_end = min(iv.end, chrom_len)
        if (effective_end - iv.begin) < target_len:
            continue
        
        # 在有效区间内随机切
        rand_start = random.randint(iv.begin, effective_end - target_len)
        rand_end = rand_start + target_len
        
        if forbid_tree and forbid_tree[chrom].overlaps(rand_start, rand_end):
            continue
            
        # 获取带侧翼的序列
        res = get_flanked_sequence(genome, chrom, rand_start, rand_end, flank_size)
        if not res: continue
        core, left, right = res
        
        seqs.append({
            "chrom": chrom,
            "start": rand_start,
            "end": rand_end,
            "strand": "+",
            "seq": core,
            "flank_left": left,
            "flank_right": right,
            "label": label
        })
    
    return seqs

def generate_random_background(genome, count, label, flank_size, forbid_trees=[]):
    """生成全基因组随机背景 (带侧翼)"""
    seqs = []
    chroms = list(genome.keys())
    attempts = 0
    
    while len(seqs) < count and attempts < count * 30:
        attempts += 1
        chrom = random.choice(chroms)
        chrom_len = len(genome[chrom])
        target_len = get_random_length()
        
        # 必须预留足够空间给 core + flanks
        if chrom_len < target_len + 2 * flank_size: continue
        
        # 随机范围要避开染色体两端
        rand_start = random.randint(flank_size, chrom_len - target_len - flank_size)
        rand_end = rand_start + target_len
        
        is_bad = False
        for tree in forbid_trees:
            if tree[chrom].overlaps(rand_start, rand_end):
                is_bad = True
                break
        if is_bad: continue
        
        res = get_flanked_sequence(genome, chrom, rand_start, rand_end, flank_size)
        if not res: continue
        core, left, right = res
        
        seqs.append({
            "chrom": chrom,
            "start": rand_start,
            "end": rand_end,
            "strand": "+",
            "seq": core,
            "flank_left": left,
            "flank_right": right,
            "label": label
        })
    return seqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--gff", required=True)
    parser.add_argument("--positive_tsv", required=True)
    parser.add_argument("--out_cds", required=True)
    parser.add_argument("--out_random", required=True)
    parser.add_argument("--num_cds", type=int, default=3000)
    parser.add_argument("--num_random", type=int, default=6000)
    parser.add_argument("--flank_size", type=int, default=150, help="侧翼长度")
    args = parser.parse_args()
    
    print("Loading genome...")
    genome = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
    
    pos_tree = load_intervals(args.positive_tsv)
    cds_tree = parse_gff_cds(args.gff)
    
    print(f"Generating CDS negatives (with {args.flank_size}bp flanks)...")
    cds_recs = generate_seqs_from_intervals(genome, cds_tree, args.num_cds, "CDS", args.flank_size, forbid_tree=pos_tree)
    pd.DataFrame(cds_recs).to_csv(args.out_cds, sep='\t', index=False)
    
    print(f"Generating Random negatives (with {args.flank_size}bp flanks)...")
    rand_recs = generate_random_background(genome, args.num_random, "Background", args.flank_size, forbid_trees=[pos_tree, cds_tree])
    pd.DataFrame(rand_recs).to_csv(args.out_random, sep='\t', index=False)
    
    print(f"Done. CDS: {len(cds_recs)}, Random: {len(rand_recs)}")

if __name__ == "__main__":
    main()