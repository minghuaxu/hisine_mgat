#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_detect_motifs.py
===================
检测SINE序列中的关键Motif特征

功能:
1. A-box和B-box motif扫描 (PWM-based)
2. PolyA/T尾部检测
3. Target Site Duplication (TSD) 检测
4. 输出统一的坐标系统

输出:
- motifs.tsv: 原始检测结果
- motifs.unified_coordinates.tsv: 统一坐标系统
- motifs.updated_boundaries.tsv: 更新后的序列边界
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.Seq import Seq


# ====================== PWM定义 ======================

# 默认A-box PWM (TATA-box like)
# 形状: (4, 10) 对应 A, C, G, T
A_BOX_PWM = np.array([
    [0.1, 0.1, 0.8, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
]).T

# 默认B-box PWM
B_BOX_PWM = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25, 0.25],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
]).T

NUC2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ====================== 工具函数 ======================

def revcomp(s: str) -> str:
    """反向互补序列"""
    return str(Seq(s).reverse_complement())


def pwm_to_logodds(pwm: np.ndarray, bg: float = 0.25, eps: float = 1e-6) -> np.ndarray:
    """将PWM转换为log-odds矩阵"""
    pwm = np.asarray(pwm, dtype=np.float64)
    # 归一化每列
    col_sums = pwm.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    pwm_norm = pwm / col_sums
    # 计算log-odds
    return np.log((pwm_norm + eps) / bg)


def scan_best_with_z(seq: str, log_pwm: np.ndarray, n_shuf: int = 30):
    """
    扫描PWM并计算Z-score
    
    返回: (best_score, best_pos, z_score)
    """
    seq = seq.upper()
    motif_len = log_pwm.shape[1]
    
    if len(seq) < motif_len:
        return -999.0, -1, 0.0
    
    # 扫描最佳位置
    best_score, best_pos = -999.0, -1
    for i in range(len(seq) - motif_len + 1):
        score = 0.0
        valid = True
        for j in range(motif_len):
            nuc = seq[i + j]
            if nuc not in NUC2IDX:
                valid = False
                break
            score += log_pwm[NUC2IDX[nuc], j]
        
        if valid and score > best_score:
            best_score, best_pos = score, i

    # 如果不需要Z-score或序列太短，计算Z-score
    if n_shuf <= 0 or best_pos == -1 or len(seq) < motif_len + 2:
        return float(best_score), int(best_pos), 0.0 # Z-score return 0
    
    # 随机打乱背景
    shuffled_scores = []
    seq_list = list(seq)
    for _ in range(n_shuf):
        np.random.shuffle(seq_list)
        shuffled_seq = "".join(seq_list)
        score, _, _ = scan_best_with_z(shuffled_seq, log_pwm, n_shuf=0)
        shuffled_scores.append(score)
    
    mean_bg = float(np.mean(shuffled_scores))
    std_bg = float(np.std(shuffled_scores)) + 1e-8
    z_score = (best_score - mean_bg) / std_bg
    
    return float(best_score), int(best_pos), float(z_score)


def hamming(a: str, b: str) -> int:
    """计算Hamming距离"""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


# ====================== PolyA/T检测 ======================

def detect_polyA_TSD(
    sine_seq: str,
    left_flank: str,
    right_flank: str,
    posA: int = None,
    posB: int = None,
    search_win: int = 60,
    min_poly: int = 7,
    max_poly: int = 70,
    min_purity: float = 0.72,
    min_tsd: int = 4,
    max_tsd: int = 28,
    max_mm_rate: float = 0.25,
    max_slide: int = 15
):
    """
    检测polyA/T尾部和TSD
    
    策略:
    1. 在SINE 3'端附近搜索polyA/T
    2. 在polyA/T之后搜索右侧TSD
    3. 在SINE 5'端附近搜索左侧TSD
    4. 确保左侧TSD不覆盖A/B-box
    
    返回: (polyA_info, tsd_info) 或 (None, None)
    """
    sine_seq = sine_seq.upper()
    left_flank = left_flank.upper()
    right_flank = right_flank.upper()
    
    left_len = len(left_flank)
    sine_len = len(sine_seq)
    full_seq = left_flank + sine_seq + right_flank
    
    # 1. 判断主要碱基是A还是T (基于尾部10bp)
    tail10 = (sine_seq[-10:] + right_flank[:10]).upper()
    primary_base = 'A' if tail10.count('A') >= tail10.count('T') else 'T'
    secondary_base = 'T' if primary_base == 'A' else 'A'
    
    # 2. PolyA/T搜索区域
    tail_start = max(0, sine_len - search_win)
    search_region = sine_seq[tail_start:] + right_flank[:search_win]
    region_global_start = left_len + tail_start
    
    # 查找所有候选polyA/T
    candidates = []
    for base in [primary_base, secondary_base]:
        n = len(search_region)
        for i in range(n):
            for length in range(min_poly, min(max_poly + 1, n - i + 1)):
                sub = search_region[i:i+length]
                purity = sub.count(base) / length
                if purity >= min_purity:
                    g_start = region_global_start + i
                    g_end = region_global_start + i + length
                    score = length * purity * (2 if base == primary_base else 1)
                    candidates.append((score, base, g_start, g_end, length, purity))
    
    if not candidates:
        return None, None
    
    # 按分数排序
    candidates.sort(reverse=True)
    best = candidates[0]
    _, poly_base, poly_start, poly_end, poly_len, poly_purity = best
    
    polyA_info = {
        'base': poly_base,
        'start': poly_start,
        'end': poly_end,
        'len': poly_len,
        'purity': round(poly_purity, 3)
    }
    
    # 3. TSD搜索
    tsd_right_from = poly_end
    tsd_right_region = full_seq[tsd_right_from : tsd_right_from + max_slide + max_tsd]
    
    # 左侧搜索区域
    left_search_from = max(0, left_len + sine_len - search_win - 60)
    left_search_region = full_seq[left_search_from : left_len + sine_len]
    
    # 最早的box位置 (防止TSD覆盖box)
    earliest_box_global = None
    if posA is not None and posA >= 0:
        earliest_box_global = left_len + posA
    if posB is not None and posB >= 0:
        eb = left_len + posB
        earliest_box_global = eb if earliest_box_global is None else min(earliest_box_global, eb)
    
    # 查找最佳TSD对
    best_tsd = None
    for tsd_len in range(max_tsd, min_tsd - 1, -1):
        max_mm = int(tsd_len * max_mm_rate + 0.99)
        
        for slide in range(min(max_slide + 1, len(tsd_right_region) - tsd_len + 1)):
            right_tsd = tsd_right_region[slide:slide + tsd_len]
            
            for lpos in range(len(left_search_region) - tsd_len + 1):
                left_tsd = left_search_region[lpos:lpos + tsd_len]
                mm = hamming(left_tsd, right_tsd)
                
                if mm > max_mm:
                    continue
                
                # 检查左侧TSD是否覆盖box
                tsd_end_global = left_search_from + lpos + tsd_len
                if earliest_box_global is not None and tsd_end_global > earliest_box_global:
                    continue
                
                match_rate = 1 - mm / tsd_len
                this_tsd = {
                    'seq': right_tsd,
                    'len': tsd_len,
                    'mm': mm,
                    'match_rate': round(match_rate, 3),
                    'left_start': left_search_from + lpos,
                    'left_end': left_search_from + lpos + tsd_len,
                    'right_start': tsd_right_from + slide,
                    'right_end': tsd_right_from + slide + tsd_len,
                }
                
                if best_tsd is None or tsd_len > best_tsd['len'] or \
                   (tsd_len == best_tsd['len'] and mm < best_tsd['mm']):
                    best_tsd = this_tsd
                
                if mm == 0:
                    return polyA_info, best_tsd
    
    return polyA_info, best_tsd


# ====================== 主程序 ======================

def main():
    parser = argparse.ArgumentParser(
        description="检测SINE序列中的Motif特征"
    )
    parser.add_argument("--in_tsv", required=True, help="输入TSV文件")
    parser.add_argument("--out_prefix", required=True, help="输出文件前缀")
    parser.add_argument("--pwmA", default=None, help="自定义A-box PWM (JASPAR格式)")
    parser.add_argument("--pwmB", default=None, help="自定义B-box PWM (JASPAR格式)")
    parser.add_argument("--new_flank_len", type=int, default=100, help="更新后的侧翼长度")
    parser.add_argument("--fast", action="store_true", help="跳过Z-score计算以加速（预测时推荐）")
    
    args = parser.parse_args()
    
    # 加载PWM
    logA = pwm_to_logodds(A_BOX_PWM)
    logB = pwm_to_logodds(B_BOX_PWM)
    
    if args.pwmA:
        from Bio import motifs
        with open(args.pwmA) as f:
            m = motifs.read(f, "jaspar")
            counts = np.array([m.counts[b] for b in "ACGT"])
            logA = pwm_to_logodds(counts + 0.1)
        print(f"[INFO] 已加载自定义A-box PWM: {args.pwmA}")
    
    if args.pwmB:
        from Bio import motifs
        with open(args.pwmB) as f:
            m = motifs.read(f, "jaspar")
            counts = np.array([m.counts[b] for b in "ACGT"])
            logB = pwm_to_logodds(counts + 0.1)
        print(f"[INFO] 已加载自定义B-box PWM: {args.pwmB}")
    
    # 创建输出目录
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    # 打开输出文件
    out_motifs = open(f"{args.out_prefix}.motifs.tsv", "w")
    out_unified = open(f"{args.out_prefix}.unified_coordinates.tsv", "w")
    out_updated = open(f"{args.out_prefix}.updated_boundaries.tsv", "w")
    
    # 写入表头
    motifs_header = [
        "chrom", "original_start", "original_end", "strand",
        "A_pos", "A_score", "A_z", "B_pos", "B_score", "B_z",
        "polyA_base", "polyA_len", "polyA_purity", "polyA_start", "polyA_end",
        "TSD_seq", "TSD_len", "TSD_mm", "TSD_match_rate",
        "left_TSD_start", "left_TSD_end", "right_TSD_start", "right_TSD_end",
        "detection_status"
    ]
    out_motifs.write("\t".join(motifs_header) + "\n")
    
    unified_header = [
        "chrom", "original_start", "original_end", "strand",
        "original_sine_start_rel", "original_sine_end_rel",
        "new_sine_start", "new_sine_end",
        "left_TSD_start", "left_TSD_end", "right_TSD_start", "right_TSD_end",
        "polyA_start", "polyA_end", "A_box_start", "A_box_end", 
        "B_box_start", "B_box_end", "unique_id"
    ]
    out_unified.write("\t".join(unified_header) + "\n")
    out_updated.write("chrom\tstart\tend\tstrand\tseq\tleft\tright\tunique_id\n")
    
    total = success = partial = 0
    
    print(f"[INFO] 开始处理: {args.in_tsv}")
    
    with open(args.in_tsv) as f:
        next(f)  # 跳过表头
        for line in f:
            if not line.strip():
                continue
            
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                print(f"[WARN] 跳过格式错误的行: {line[:50]}")
                continue
            
            chrom, s, e, strand, core, left, right = parts[:7]
            orig_start, orig_end = int(s), int(e)
            total += 1
            
            # 处理负链: 统一到正链
            if strand == "-":
                core = revcomp(core)
                left, right = revcomp(right), revcomp(left)
            
            left_len = len(left)
            sine_len = len(core)
            full_seq = left + core + right
            
            n_shuf = 0 if args.fast else 30  # 设置 shuffle 次数
    
            # 调用时
            scoreA, posA, zA = scan_best_with_z(core, logA, n_shuf=n_shuf)
            scoreB, posB, zB = scan_best_with_z(core, logB, n_shuf=n_shuf)

            # 修改逻辑：如果分数不够高，强制视为 -1 (未检测到)
            # 例如：如果 Z-score < 2.0 或 score < 0，则忽略
            if zA < 2.0: posA = -1
            if zB < 2.0: posB = -1
            
            # 检测polyA/T和TSD
            polyA_info, tsd_info = detect_polyA_TSD(
                sine_seq=core,
                left_flank=left,
                right_flank=right,
                posA=posA if posA >= 0 else None,
                posB=posB if posB >= 0 else None,
            )
            
            A_len = logA.shape[1]
            B_len = logB.shape[1]
            
            # 判断检测状态
            status = "NO"
            if polyA_info and tsd_info:
                status = "YES"
                success += 1
            elif polyA_info:
                status = "POLYA_ONLY"
                partial += 1
            
            # 准备输出数据
            row_motifs = {
                "polyA_base": polyA_info['base'] if polyA_info else "N",
                "polyA_len": polyA_info['len'] if polyA_info else 0,
                "polyA_purity": polyA_info['purity'] if polyA_info else 0,
                "polyA_start": polyA_info['start'] if polyA_info else -1,
                "polyA_end": polyA_info['end'] if polyA_info else -1,
                "TSD_seq": tsd_info['seq'] if tsd_info else "",
                "TSD_len": tsd_info['len'] if tsd_info else 0,
                "TSD_mm": tsd_info['mm'] if tsd_info else -1,
                "TSD_match_rate": tsd_info['match_rate'] if tsd_info else 0,
                "left_TSD_start": tsd_info['left_start'] if tsd_info else -1,
                "left_TSD_end": tsd_info['left_end'] if tsd_info else -1,
                "right_TSD_start": tsd_info['right_start'] if tsd_info else -1,
                "right_TSD_end": tsd_info['right_end'] if tsd_info else -1,
            }
            
            # 更新边界 (仅在完全成功时)
            if status == "YES":
                new_sine_start = row_motifs["left_TSD_end"]
                new_sine_end = row_motifs["right_TSD_start"]
                new_left = full_seq[max(0, new_sine_start - args.new_flank_len):new_sine_start]
                new_right = full_seq[new_sine_end:new_sine_end + args.new_flank_len]
                new_core = full_seq[new_sine_start:new_sine_end]
            else:
                new_left = left[-args.new_flank_len:]
                new_right = right[:args.new_flank_len]
                new_core = core
                new_sine_start = new_sine_end = -1
            
            # 创建unique_id
            unique_id = f"{chrom}:{orig_start}-{orig_end}({strand})"
            
            # 写入unified coordinates
            out_unified.write("\t".join(map(str, [
                chrom, orig_start, orig_end, strand,
                len(left), len(left) + len(core),  # original relative
                new_sine_start, new_sine_end,
                row_motifs["left_TSD_start"], row_motifs["left_TSD_end"],
                row_motifs["right_TSD_start"], row_motifs["right_TSD_end"],
                row_motifs["polyA_start"], row_motifs["polyA_end"],
                left_len + posA if posA >= 0 else -1,
                left_len + posA + A_len if posA >= 0 else -1,
                left_len + posB if posB >= 0 else -1,
                left_len + posB + B_len if posB >= 0 else -1,
                unique_id
            ])) + "\n")
            
            # 写入motifs
            out_motifs.write("\t".join(map(str, [
                chrom, orig_start, orig_end, strand,
                posA if posA >= 0 else -1,
                f"{scoreA:.3f}", f"{zA:.2f}",
                posB if posB >= 0 else -1,
                f"{scoreB:.3f}", f"{zB:.2f}",
                row_motifs["polyA_base"], row_motifs["polyA_len"], row_motifs["polyA_purity"],
                row_motifs["polyA_start"], row_motifs["polyA_end"],
                row_motifs["TSD_seq"], row_motifs["TSD_len"], row_motifs["TSD_mm"],
                row_motifs["TSD_match_rate"],
                row_motifs["left_TSD_start"], row_motifs["left_TSD_end"],
                row_motifs["right_TSD_start"], row_motifs["right_TSD_end"],
                status
            ])) + "\n")
            
            # 写入updated boundaries
            out_updated.write(f"{chrom}\t{orig_start}\t{orig_end}\t{strand}\t"
                            f"{new_core}\t{new_left}\t{new_right}\t{unique_id}\n")
    
    out_motifs.close()
    out_unified.close()
    out_updated.close()
    
    # 打印统计
    print("\n" + "="*60)
    print("Motif检测完成")
    print("="*60)
    print(f"总序列数:         {total}")
    print(f"完全成功 (YES):   {success}")
    print(f"部分成功 (POLYA): {partial}")
    print(f"失败 (NO):        {total - success - partial}")
    print(f"成功率:           {100*(success+partial)/total:.1f}%")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  - {args.out_prefix}.motifs.tsv")
    print(f"  - {args.out_prefix}.unified_coordinates.tsv")
    print(f"  - {args.out_prefix}.updated_boundaries.tsv")


if __name__ == "__main__":
    main()