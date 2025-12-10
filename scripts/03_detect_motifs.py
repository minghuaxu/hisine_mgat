#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_detect_motifs.py (V7: Golden Standard)
=========================================
检测 SINE 序列特征 - 黄金标准版

核心改进 (基于朋友建议 + 鲁棒性修正):
1. ✅ 绝对正义链原则: SINE 3' 尾巴必须是 PolyA。禁止 PolyT。
2. ✅ TSD 锚定 + 逆向验证: 先定 TSD，再看 TSD 前面是不是紧跟 A 尾巴。
3. ✅ 智能尾巴提取: 检查紧邻 Right TSD 前方的序列，兼容 Tail 在 Core 内或 Core 外的情况。
4. ✅ 零修改原则: 绝不修改 Minimap2 提取的核心序列。
"""

import argparse
import sys
import numpy as np
from Bio.Seq import Seq
from pathlib import Path

# ====================== PWM定义 (不变) ======================
A_BOX_PWM = np.array([
    [0.1, 0.1, 0.8, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], 
    [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], 
    [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], 
    [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
]).T

B_BOX_PWM = np.array([
    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], 
    [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], 
    [1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.0, 0.5, 0.0, 0.5], 
    [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
]).T

NUC2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

# ====================== 工具函数 ======================

def revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())

def pwm_to_logodds(pwm: np.ndarray, bg: float = 0.25, eps: float = 1e-6) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=np.float64)
    col_sums = pwm.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    pwm_norm = pwm / col_sums
    return np.log((pwm_norm + eps) / bg)

def scan_best_with_z(seq: str, log_pwm: np.ndarray, n_shuf: int = 30):
    seq = seq.upper()
    motif_len = log_pwm.shape[1]
    if len(seq) < motif_len: return -999.0, -1, 0.0
    
    best_score, best_pos = -999.0, -1
    for i in range(len(seq) - motif_len + 1):
        score = 0.0
        valid = True
        for j in range(motif_len):
            nuc = seq[i + j]
            if nuc not in NUC2IDX: valid = False; break
            score += log_pwm[NUC2IDX[nuc], j]
        if valid and score > best_score:
            best_score, best_pos = score, i

    if n_shuf <= 0 or best_pos == -1:
        return float(best_score), int(best_pos), 0.0
    
    shuffled_scores = []
    seq_list = list(seq)
    for _ in range(n_shuf):
        np.random.shuffle(seq_list)
        shuffled_seq = "".join(seq_list)
        s_best = -999.0
        for i in range(len(shuffled_seq) - motif_len + 1):
            sc = 0.0
            for j in range(motif_len):
                nuc = shuffled_seq[i+j]
                if nuc in NUC2IDX: sc += log_pwm[NUC2IDX[nuc], j]
            if sc > s_best: s_best = sc
        shuffled_scores.append(s_best)
    
    mean_bg = float(np.mean(shuffled_scores))
    std_bg = float(np.std(shuffled_scores)) + 1e-8
    z_score = (best_score - mean_bg) / std_bg
    return float(best_score), int(best_pos), float(z_score)

def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)

def is_low_complexity(seq):
    if len(seq) < 4: return True
    dimers = [seq[i:i+2] for i in range(len(seq)-1)]
    if not dimers: return True
    most_common = max(set(dimers), key=dimers.count)
    if dimers.count(most_common) > len(seq) * 0.75: return True
    return False

# ====================== V7: Golden Standard Logic ======================

def detect_structure_v7(core, left, right):
    """
    V7 策略:
    1. 锚定: 假设 Left TSD 就在 Left Flank 末端。
    2. 搜索: 在 Core尾部 + Right头部 寻找 Right TSD。
    3. 验证: 检查 Right TSD 紧前面的序列是否为 PolyA。
    """
    core = core.upper()
    left = left.upper()
    right = right.upper()
    
    L_len = len(left)
    C_len = len(core)
    
    # 搜索区域: Core 后 50bp + Right 前 100bp
    search_overlap = 50
    search_limit = 100
    
    # 搜索序列起始的全局坐标
    search_start_global = L_len + max(0, C_len - search_overlap)
    search_seq = core[max(0, C_len - search_overlap):] + right[:search_limit]
    
    # 全局序列
    full_seq = left + core + right
    
    candidates = []
    
    # 遍历 TSD 长度
    for t_len in range(25, 5, -1): # 6bp - 25bp
        if L_len < t_len: continue
        
        left_tsd_cand = left[-t_len:]
        if is_low_complexity(left_tsd_cand): continue
        
        max_mm = max(1, int(t_len * 0.15))
        
        # 在右侧搜索
        for i in range(len(search_seq) - t_len + 1):
            right_tsd_cand = search_seq[i : i+t_len]
            mm = hamming(left_tsd_cand, right_tsd_cand)
            
            if mm <= max_mm:
                # 找到 TSD 对
                r_start_global = search_start_global + i
                r_end_global = r_start_global + t_len
                
                # === 黄金验证逻辑 ===
                # 提取 Right TSD 紧前面的序列 (Tail Candidate)
                # 检查窗口: 25bp
                check_len = 25
                # 确保不越过 Left Flank (即 SINE 长度至少为 0)
                tail_start_global = max(L_len, r_start_global - check_len)
                tail_end_global = r_start_global
                
                tail_region = full_seq[tail_start_global : tail_end_global]
                
                if not tail_region: continue
                
                # 计算统计量
                seq_len = len(tail_region)
                cnt_A = tail_region.count('A')
                purity_A = cnt_A / seq_len
                
                def get_max_run(s, char):
                    mx, cur = 0, 0
                    for c in s:
                        if c == char: cur += 1
                        else: mx = max(mx, cur); cur = 0
                    return max(mx, cur)
                
                run_A = get_max_run(tail_region, 'A')
                
                # === 判定规则 (严禁 PolyT) ===
                valid_tail = False
                
                # 1. 强 PolyA 信号 (连续 8个 A)
                if run_A >= 8:
                    valid_tail = True
                # 2. 中等 PolyA 信号 (连续 5个 A 且 纯度及格)
                elif run_A >= 5 and purity_A >= 0.6:
                    valid_tail = True
                # 3. 高纯度 A 区域 (无长连续，但全是 A，比如 AAGAAGA)
                elif purity_A >= 0.8 and seq_len >= 8:
                    valid_tail = True
                
                # 注意：这里完全没有 PolyT 的逻辑分支。
                # 如果是 TTTTT，run_A=0，purity_A=0 -> Valid=False。直接过滤。
                
                if valid_tail:
                    # 计算得分 (TSD长度奖励, 错配惩罚, PolyA奖励)
                    score = t_len * 10 - mm * 5 + run_A * 2
                    candidates.append({
                        'score': score,
                        'polyA': {
                            'base': 'A',
                            'start': tail_start_global, # 近似开始
                            'end': tail_end_global,
                            'len': seq_len,
                            'purity': round(purity_A, 3)
                        },
                        'tsd': {
                            'seq': right_tsd_cand,
                            'len': t_len,
                            'mm': mm,
                            'match_rate': round(1.0 - mm/t_len, 3),
                            'left_start': L_len - t_len,
                            'left_end': L_len,
                            'right_start': r_start_global,
                            'right_end': r_end_global
                        }
                    })

    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]['polyA'], candidates[0]['tsd']
    
    return None, None

# ====================== 主程序 ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsv", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--pwmA", default=None)
    parser.add_argument("--pwmB", default=None)
    parser.add_argument("--new_flank_len", type=int, default=100)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    
    # Load PWMs
    logA = pwm_to_logodds(A_BOX_PWM)
    logB = pwm_to_logodds(B_BOX_PWM)
    if args.pwmA:
        from Bio import motifs
        with open(args.pwmA) as f:
            m = motifs.read(f, "jaspar")
            logA = pwm_to_logodds(np.array([m.counts[b] for b in "ACGT"]) + 0.1)
    if args.pwmB:
        from Bio import motifs
        with open(args.pwmB) as f:
            m = motifs.read(f, "jaspar")
            logB = pwm_to_logodds(np.array([m.counts[b] for b in "ACGT"]) + 0.1)
    
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    out_motifs = open(f"{args.out_prefix}.motifs.tsv", "w")
    out_unified = open(f"{args.out_prefix}.unified_coordinates.tsv", "w")
    out_updated = open(f"{args.out_prefix}.updated_boundaries.tsv", "w")
    
    # Headers
    out_motifs.write("\t".join(["chrom", "original_start", "original_end", "strand", "A_pos", "A_score", "A_z", "B_pos", "B_score", "B_z", "polyA_base", "polyA_len", "polyA_purity", "polyA_start", "polyA_end", "TSD_seq", "TSD_len", "TSD_mm", "TSD_match_rate", "left_TSD_start", "left_TSD_end", "right_TSD_start", "right_TSD_end", "detection_status"]) + "\n")
    out_unified.write("\t".join(["chrom", "original_start", "original_end", "strand", "original_sine_start_rel", "original_sine_end_rel", "new_sine_start", "new_sine_end", "left_TSD_start", "left_TSD_end", "right_TSD_start", "right_TSD_end", "polyA_start", "polyA_end", "A_box_start", "A_box_end", "B_box_start", "B_box_end", "unique_id"]) + "\n")
    out_updated.write("chrom\tstart\tend\tstrand\tseq\tleft\tright\tunique_id\n")
    
    total = 0
    with open(args.in_tsv) as f:
        next(f)
        for line in f:
            if not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7: continue
            chrom, s, e, strand, core, left, right = parts[:7]
            orig_start, orig_end = int(s), int(e)
            total += 1
            
            if strand == "-":
                core = revcomp(core)
                left, right = revcomp(right), revcomp(left)
            
            left_len, sine_len = len(left), len(core)
            # 全局序列
            n_shuf = 0 if args.fast else 30
            
            # --- 扫描 Box ---
            scoreA, posA, zA = scan_best_with_z(core, logA, n_shuf=n_shuf)
            scoreB, posB, zB = scan_best_with_z(core, logB, n_shuf=n_shuf)
            
            # 分层阈值
            Z_STRICT, Z_RELAX, ABS_MIN = 3.5, 2.5, -20.0
            valid_A = (posA >= 0 and zA >= Z_RELAX and scoreA > ABS_MIN)
            valid_B = (posB >= 0 and zB >= Z_RELAX and scoreB > ABS_MIN)
            final_posA, final_posB = -1, -1
            
            if valid_A and valid_B:
                dist = posB - posA
                if 15 <= dist <= 100: final_posA, final_posB = posA, posB
                else:
                    if zA >= Z_STRICT: final_posA = posA
                    if zB >= Z_STRICT: final_posB = posB
            else:
                if valid_A and zA >= Z_STRICT: final_posA = posA
                if valid_B and zB >= Z_STRICT: final_posB = posB
            
            # --- 检测 PolyA/TSD (V7) ---
            polyA_info, tsd_info = detect_structure_v7(core, left, right)
            
            # 状态分级
            status = "NO"
            has_boxes = (final_posA >= 0 or final_posB >= 0)
            has_pair = (final_posA >= 0 and final_posB >= 0)
            has_struct = (polyA_info is not None and tsd_info is not None)
            
            if has_pair and has_struct: status = "HIGH_CONF"
            elif (has_boxes) and has_struct: status = "MED_CONF"
            elif has_boxes or has_struct: status = "LOW_CONF"
            
            # 准备输出
            A_len, B_len = logA.shape[1], logB.shape[1]
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

            # ========================================================
            # V7: 仅做 Flank 标准化截断 (Read Only)
            # ========================================================
            new_left = left[-args.new_flank_len:]
            new_right = right[:args.new_flank_len]
            new_core = core
            
            offset_shift = max(0, len(left) - args.new_flank_len)
            
            def adjust_coord(c):
                if c == -1: return -1
                return c - offset_shift

            if has_struct:
                det_sine_start = adjust_coord(row_motifs["left_TSD_end"])
                det_sine_end = adjust_coord(row_motifs["right_TSD_start"])
            else:
                det_sine_start, det_sine_end = -1, -1
            
            unique_id = f"{chrom}:{orig_start}-{orig_end}({strand})"
            
            out_unified.write("\t".join(map(str, [
                chrom, orig_start, orig_end, strand, 
                len(new_left), len(new_left) + len(new_core), 
                det_sine_start, det_sine_end,
                adjust_coord(row_motifs["left_TSD_start"]), adjust_coord(row_motifs["left_TSD_end"]),
                adjust_coord(row_motifs["right_TSD_start"]), adjust_coord(row_motifs["right_TSD_end"]),
                adjust_coord(row_motifs["polyA_start"]), adjust_coord(row_motifs["polyA_end"]),
                adjust_coord(left_len + final_posA) if final_posA >= 0 else -1,
                adjust_coord(left_len + final_posA + A_len) if final_posA >= 0 else -1,
                adjust_coord(left_len + final_posB) if final_posB >= 0 else -1,
                adjust_coord(left_len + final_posB + B_len) if final_posB >= 0 else -1,
                unique_id
            ])) + "\n")
            
            out_motifs.write("\t".join(map(str, [
                chrom, orig_start, orig_end, strand,
                final_posA if final_posA >= 0 else -1, f"{scoreA:.3f}", f"{zA:.2f}",
                final_posB if final_posB >= 0 else -1, f"{scoreB:.3f}", f"{zB:.2f}",
                row_motifs["polyA_base"], row_motifs["polyA_len"], row_motifs["polyA_purity"],
                row_motifs["polyA_start"], row_motifs["polyA_end"],
                row_motifs["TSD_seq"], row_motifs["TSD_len"], row_motifs["TSD_mm"],
                row_motifs["TSD_match_rate"],
                row_motifs["left_TSD_start"], row_motifs["left_TSD_end"],
                row_motifs["right_TSD_start"], row_motifs["right_TSD_end"],
                status
            ])) + "\n")
            
            out_updated.write(f"{chrom}\t{orig_start}\t{orig_end}\t{strand}\t{new_core}\t{new_left}\t{new_right}\t{unique_id}\n")
    
    out_motifs.close(); out_unified.close(); out_updated.close()
    print(f"处理完成: {total} 条序列")

if __name__ == "__main__":
    main()