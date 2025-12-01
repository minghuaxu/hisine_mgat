#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_negatives.py (Bug Fixed Version)
===========================================
负样本构建脚本 - 最终修正版 v2

修复历史:
1. 修复了超长序列未裁剪的问题。
2. 修复了按照 SINE 长度分布采样的问题。
3. [本次修复] 修复了边界坐标未钳制(Clamp)导致的坐标与序列长度不一致 Bug。
"""
import argparse
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta
from Bio import SeqIO
from Bio.Seq import Seq

from sine_classifier.utils import (
    run_command,
    merge_intervals,
    invert_intervals,
)
from sine_classifier.sam_parser import parse_sam_and_extract_seqs

# ======================================================================
# 辅助函数
# ======================================================================

def revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())

def load_sine_ref_lengths(fasta_path: str) -> list:
    """
    读取金标准 SINE 参考库 FASTA，获取长度分布。
    """
    lengths = []
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            lengths.append(len(record.seq))
        
        # 过滤极端异常值
        lengths = [l for l in lengths if 40 <= l <= 1500]
        
        if not lengths:
            print(f"[WARN] SINE 参考库 {fasta_path} 为空或无有效序列，使用默认分布 (100-300bp)。")
            return list(range(100, 301))
            
        print(f"[INFO] 已加载 SINE 参考库长度分布: 数量={len(lengths)}, 均值={np.mean(lengths):.1f}, 范围=[{min(lengths)}, {max(lengths)}]")
        return lengths
        
    except Exception as e:
        print(f"[WARN] 读取 SINE 参考库失败: {e}，使用默认分布。")
        return list(range(100, 301))

def extract_window_with_flanks(genome, chrom, start, end, strand, flank_size):
    """
    原子操作：提取 Core + Flanks 并处理反向互补。
    注意：传入的 start/end 必须已经是合法的基因组坐标（已 Clamp）。
    """
    chrom_len = len(genome[chrom])
    
    # 1. 提取核心 (Genomic + Strand)
    # 这里不再需要 max/min，因为外层已经保证了坐标合法
    # 但为了双重保险，保留边界检查不会有害
    safe_start = max(0, start)
    safe_end = min(chrom_len, end)
    
    # 提取基因组上的序列 (始终是正链方向)
    genomic_core = str(genome[chrom][safe_start:safe_end])
    
    # 2. 提取侧翼 (Genomic + Strand)
    # Left (Genomic upstream)
    l_start = max(0, safe_start - flank_size)
    genomic_left = str(genome[chrom][l_start:safe_start])
    
    # Right (Genomic downstream)
    r_end = min(chrom_len, safe_end + flank_size)
    genomic_right = str(genome[chrom][safe_end:r_end])
    
    # 3. 根据 Strand 处理方向
    if strand == '-':
        final_core = revcomp(genomic_core)
        final_left_flank = revcomp(genomic_right)
        final_right_flank = revcomp(genomic_left)
    else:
        final_core = genomic_core
        final_left_flank = genomic_left
        final_right_flank = genomic_right
        
    return final_core, final_left_flank, final_right_flank

def process_and_crop_hits(hits: list, genome_fa: str, pos_length_pool: list, flank_size: int = 150) -> list:
    """
    根据 SINE 参考库的长度分布，对 TE 负样本进行动态裁剪。
    [Fix] 增加了坐标边界检查。
    """
    genome = Fasta(genome_fa, sequence_always_upper=True)
    processed = []
    count_cropped = 0
    
    for rec in hits:
        orig_seq_len = len(rec['seq'])
        chrom = rec['chrom']
        
        # 获取染色体长度，用于边界检查
        chrom_len = len(genome[chrom])
        
        # 1. 随机选择目标长度
        target_len = random.choice(pos_length_pool)
        
        orig_start = rec['start']
        center = orig_start + (orig_seq_len // 2)
        
        # 2. 如果序列比目标长度短很多，保留原样
        if orig_seq_len < target_len * 0.8:
            if orig_seq_len >= 40: 
                processed.append(rec)
            continue
            
        # 3. 执行裁剪
        count_cropped += 1
        
        # 计算理想的基因组坐标
        half_len = target_len // 2
        raw_start = center - half_len
        raw_end = raw_start + target_len
        
        # [关键修复] 在这里进行 Clamp (钳制)，防止越界
        # 如果 raw_start < 0，拉回 0；如果 raw_end > chrom_len，拉回 chrom_len
        valid_start = max(0, raw_start)
        valid_end = min(chrom_len, raw_end)
        
        # 如果因为在边界导致截取后长度过短（例如只剩10bp），则丢弃
        if valid_end - valid_start < 40:
            continue
        
        try:
            # 传入修正后的 valid_start 和 valid_end
            core, left, right = extract_window_with_flanks(
                genome, chrom, valid_start, valid_end, rec['strand'], flank_size
            )
            
            # 更新记录，使用修正后的坐标
            new_rec = rec.copy()
            new_rec['start'] = valid_start
            new_rec['end'] = valid_end
            new_rec['seq'] = core
            new_rec['flank_left'] = left
            new_rec['flank_right'] = right
            
            processed.append(new_rec)
            
        except Exception as e:
            continue
        
    if count_cropped > 0:
        print(f"[INFO] 根据 SINE 长度分布重塑了 {count_cropped} 条序列 (已处理边界坐标)。")
        
    return processed

# ----------------------------------------------------------------------
# GFF 解析
# ----------------------------------------------------------------------
def parse_gff_to_exclude_regions(gff_file: str) -> defaultdict:
    exclude = defaultdict(list)
    exclude_types = {
        "transposable_element", "repeat_region", "gene", 
        "rRNA", "tRNA", "exon", "CDS", "mobile_genetic_element"
    }
    try:
        with open(gff_file) as fh:
            for line in fh:
                if line.startswith("#"): continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5: continue
                if parts[2] in exclude_types:
                    start = max(0, int(parts[3]) - 1)
                    end = int(parts[4])
                    exclude[parts[0]].append((start, end))
    except FileNotFoundError:
        return exclude
    for c in exclude: exclude[c] = merge_intervals(exclude[c])
    return exclude

# ----------------------------------------------------------------------
# 背景采样
# ----------------------------------------------------------------------
def sample_background_windows(genome_fa: str, allowed_regions: dict, n_samples: int, pos_length_pool: list, flank_size: int) -> list:
    if n_samples == 0: return []
    genome = Fasta(genome_fa, sequence_always_upper=True)
    sampled_records = []
    
    max_req = max(pos_length_pool) + 2 * flank_size
    chrom_list = [c for c, rs in allowed_regions.items() if rs and any((r[1]-r[0]) >= max_req for r in rs)]
    if not chrom_list: return []

    attempts = 0
    max_attempts = n_samples * 200

    while len(sampled_records) < n_samples and attempts < max_attempts:
        attempts += 1
        chrom = random.choice(chrom_list)
        target_len = random.choice(pos_length_pool) 
        req_len = target_len + 2 * flank_size
        
        valid_gaps = [g for g in allowed_regions[chrom] if (g[1]-g[0]) >= req_len]
        if not valid_gaps: continue

        gap_start, gap_end = random.choice(valid_gaps)
        
        # 背景提取通常选在中间，不太会触碰边界，但保持一致性
        start = random.randint(gap_start + flank_size, gap_end - target_len - flank_size)
        end = start + target_len

        # 这里的 start/end 已经保证在染色体内部，无需 Clamp
        core, left, right = extract_window_with_flanks(genome, chrom, start, end, '+', flank_size)

        sampled_records.append({
            "chrom": chrom, "start": start, "end": end, "strand": "+",
            "seq": core, "flank_left": left, "flank_right": right,
            "type": "BG"
        })

    return sampled_records

# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def extract_te_negatives_from_library(
    genome_fa: str, mm2_idx_path: str, te_ref_fa: str, label: str,
    out_prefix: str, threads: int, flank_size: int,
    min_mapq, cov_thr: float, min_as_score: int, max_de_divergence: float,
    pos_length_pool: list
) -> list:
    
    if not te_ref_fa: return []
    print(f"\n[INFO] Processing negative TE library: {label} ({te_ref_fa})")

    sam_path = f"{out_prefix}.{label}.sam"
    minimap2_cmd = [
        "minimap2", "-a", "-N", "50000", "--secondary=yes", "--score-N=0",
        "-t", str(threads), mm2_idx_path, te_ref_fa,
    ]
    with open(sam_path, "w") as out_sam:
        run_command(minimap2_cmd, stdout=out_sam)

    raw_hits = parse_sam_and_extract_seqs(
        sam_filepath=sam_path, genome_fa=genome_fa,
        min_mapq=min_mapq, flank_size=flank_size,
        min_coverage_ratio=cov_thr, min_as_score=min_as_score,
        max_de_divergence=max_de_divergence,
    )
    
    cropped_hits = process_and_crop_hits(
        raw_hits, 
        genome_fa=genome_fa, 
        pos_length_pool=pos_length_pool,
        flank_size=flank_size
    )

    for rec in cropped_hits:
        rec["type"] = label

    print(f"[INFO] Collected {len(cropped_hits)} negatives for {label}.")
    return cropped_hits

def downsample_to_size(records: list, target_n: int, label: str) -> list:
    if target_n <= 0 or not records: return []
    if len(records) <= target_n: return records
    return random.sample(records, target_n)

def write_negatives_to_files(records: list, prefix: str):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    tsv_path = f"{prefix}.tsv"
    fa_path = f"{prefix}.fa"

    if not records:
        with open(tsv_path, "w") as f:
            f.write("chrom\tstart\tend\tstrand\tseq\tflank_left\tflank_right\ttype\n")
        open(fa_path, "w").close()
        return

    random.shuffle(records)
    df = pd.DataFrame(records)
    df.to_csv(tsv_path, sep="\t", index=False)

    with open(fa_path, "w") as f:
        for _, row in df.iterrows():
            header = f">{row['chrom']}:{row['start']}-{row['end']}({row['strand']})|{row['type']}"
            full_seq = f"{row['flank_left']}{row['seq']}{row['flank_right']}"
            f.write(f"{header}\n{full_seq}\n")
    print(f"✅ Wrote {len(df)} records to {tsv_path} and {fa_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--gff", required=True)
    parser.add_argument("--pos_tsv", required=True)
    parser.add_argument("--sine_ref", required=True)
    parser.add_argument("--pos_count", type=int, required=True)
    parser.add_argument("--dna_ref")
    parser.add_argument("--ltr_ref")
    parser.add_argument("--tir_ref")
    parser.add_argument("--sine_ref_neg")
    parser.add_argument("--ratio_bg", type=float, default=1.0)
    parser.add_argument("--out_prefix", default="negatives")
    parser.add_argument("--flank", type=int, default=150)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min_mapq", type=int, default=0)
    parser.add_argument("--cov_thr", type=float, default=0.8)
    parser.add_argument("--min_as_score", type=int, default=100)
    parser.add_argument("--max_de_divergence", type=float, default=0.25)
    args = parser.parse_args()

    pos_length_pool = load_sine_ref_lengths(args.sine_ref)

    n_pos = int(args.pos_count)
    exclude_regions = parse_gff_to_exclude_regions(args.gff)
    
    try:
        pos_df = pd.read_csv(args.pos_tsv, sep="\t")
        if not pos_df.empty:
            for _, row in pos_df.iterrows():
                exclude_regions[row["chrom"]].append((max(0, int(row["start"])-args.flank), int(row["end"])+args.flank))
    except: pass

    for c in exclude_regions: exclude_regions[c] = merge_intervals(exclude_regions[c])

    genome = Fasta(args.genome)
    allowed_regions = defaultdict(list)
    for c in genome.keys():
        allowed_regions[c] = invert_intervals(exclude_regions.get(c, []), len(genome[c]))

    genome_path = Path(args.genome)
    mm2_idx_path = str(genome_path) + ".mmi"
    if Path(mm2_idx_path).exists(): Path(mm2_idx_path).unlink()
    
    run_command(["minimap2", "-d", mm2_idx_path, "-k", "12", "-w", "5", str(genome_path)])

    te_negatives_all = []
    min_mapq = None if args.min_mapq <= 0 else args.min_mapq
    
    te_specs = [("DNA", args.dna_ref), ("LTR", args.ltr_ref), ("TIR", args.tir_ref), ("SINE_NEG", args.sine_ref_neg)]
    
    for label, ref_fa in te_specs:
        if not ref_fa: continue
        hits = extract_te_negatives_from_library(
            args.genome, mm2_idx_path, ref_fa, label, args.out_prefix,
            args.threads, args.flank, min_mapq, args.cov_thr, args.min_as_score, args.max_de_divergence,
            pos_length_pool
        )
        te_negatives_all.extend(downsample_to_size(hits, n_pos, label))

    n_bg = int(n_pos * args.ratio_bg)
    bg_hits = sample_background_windows(args.genome, allowed_regions, n_bg, pos_length_pool, args.flank)

    all_negatives = te_negatives_all + bg_hits
    write_negatives_to_files(all_negatives, args.out_prefix)

if __name__ == "__main__":
    main()