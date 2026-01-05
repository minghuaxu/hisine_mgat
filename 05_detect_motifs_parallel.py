#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_detect_motifs_parallel.py
============================
功能：SINE Motif 并行检测脚本
特点：
1. 多进程并行加速 (Multiprocessing)
2. 进度条显示 (tqdm)
3. 保持与串行版完全一致的输出格式和坐标逻辑
"""

import argparse
import sys
import re
import csv
import pandas as pd
import numpy as np
from Bio.Seq import Seq
import multiprocessing
from functools import partial
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable # 如果没装 tqdm，回退到普通迭代

# ====================== PWM 数据准备 ======================

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

# 全局变量 (用于子进程共享只读数据)
GLOBAL_LOG_A = None
GLOBAL_LOG_B = None

# ====================== 工具函数 ======================

def revcomp(s: str) -> str:
    if not s: return ""
    return str(Seq(s).reverse_complement())

def pwm_to_logodds(pwm: np.ndarray, bg: float = 0.25, eps: float = 1e-6) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=np.float64)
    col_sums = pwm.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    pwm_norm = pwm / col_sums
    return np.log((pwm_norm + eps) / bg)

def scan_best_with_z(seq: str, log_pwm: np.ndarray, n_shuf: int = 20):
    seq = seq.upper()
    motif_len = log_pwm.shape[1]
    if len(seq) < motif_len: return -999.0, -1, 0.0
    
    # 快速扫描最佳位置
    scores = []
    best_score = -999.0
    best_pos = -1
    
    # 简单的滑动窗口
    for i in range(len(seq) - motif_len + 1):
        score = 0.0
        valid = True
        for j in range(motif_len):
            nuc = seq[i + j]
            if nuc not in NUC2IDX: 
                valid = False; break
            score += log_pwm[NUC2IDX[nuc], j]
        
        if valid and score > best_score:
            best_score = score
            best_pos = i

    if n_shuf <= 0 or best_pos == -1:
        return float(best_score), int(best_pos), 0.0
    
    # 计算 Z-score (Shuffle)
    # 这里的计算量是最大的瓶颈，并行化收益主要来自这里
    shuffled_scores = []
    seq_arr = list(seq)
    for _ in range(n_shuf):
        np.random.shuffle(seq_arr)
        s_shuf = "".join(seq_arr)
        curr_best = -999.0
        for i in range(len(s_shuf) - motif_len + 1):
            sc = 0.0
            valid = True
            for j in range(motif_len):
                if s_shuf[i+j] in NUC2IDX:
                    sc += log_pwm[NUC2IDX[s_shuf[i+j]], j]
                else:
                    valid = False; break
            if valid and sc > curr_best:
                curr_best = sc
        shuffled_scores.append(curr_best)
    
    mean_bg = np.mean(shuffled_scores)
    std_bg = np.std(shuffled_scores) + 1e-8
    z_score = (best_score - mean_bg) / std_bg
    return float(best_score), int(best_pos), float(z_score)

def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)

def is_low_complexity(seq):
    if len(seq) < 4: return True
    dimers = [seq[i:i+2] for i in range(len(seq)-1)]
    if not dimers: return True
    most_common = max(set(dimers), key=dimers.count)
    return dimers.count(most_common) > len(seq) * 0.75

def detect_structure(core, left, right):
    """
    返回相对于 Left+Core+Right 拼接序列的坐标
    """
    core = core.upper()
    left = left.upper()
    right = right.upper()
    
    L_len = len(left)
    C_len = len(core)
    full_seq = left + core + right

    # === 1. 独立检测 PolyA (Blind Search at 3' end) ===
    # 逻辑：在 Core 的末尾 (或 Right 的开头) 寻找 PolyA
    # 我们关注拼接点附近的 50bp 窗口
    polyA_window_start = L_len + C_len - 50 
    polyA_window_end = L_len + C_len + 20 # 允许溢出到 Right 一点点
    
    # 提取待检测区域
    check_region = full_seq[max(0, polyA_window_start) : min(len(full_seq), polyA_window_end)]
    # print(check_region)
    
    best_polyA = None
    
    # 简单的滑动窗口找最密集的 A
    if len(check_region) >= 10:
        # 寻找最长连续 A 或高纯度 A 区
        # 这里用正则或者简单统计
        # 为了速度，我们找 "Window 10bp, A count >= 8"
        max_score = 8
        for i in range(len(check_region) - 10):
            sub = check_region[i : i+15] # 看 15bp
            # 使用正则找出 sub 中所有连续的 A 段
            # re.finditer 会返回每个匹配对象的起点和终点
            a_matches = list(re.finditer(r'A+', sub))
            
            if a_matches:
                # 在这 15bp 里，找出最长的一段连续 A
                longest_match = max(a_matches, key=lambda m: m.end() - m.start())
                
                # 这段连续 A 的长度
                current_consecutive_len = longest_match.end() - longest_match.start()
                
                # 如果当前这一段比之前记录的都要长，则更新
                if current_consecutive_len > max_score:
                    max_score = current_consecutive_len
                    
                    # 计算精确坐标：
                    # polyA_window_start: check_region 在整条序列中的起点偏移
                    # i: sub 在 check_region 中的偏移
                    # longest_match.start(): 连续 A 在 sub 中的起点偏移
                    p_start = polyA_window_start + i + longest_match.start()
                    p_end = polyA_window_start + i + longest_match.end()
                    
                    best_polyA = (p_start, p_end)
    # print(full_seq[best_polyA[0]:best_polyA[1]])

    search_overlap = 50
    search_limit = 100
    
    search_start_global = L_len + max(0, C_len - search_overlap)
    search_seq = core[max(0, C_len - search_overlap):] + right[:search_limit]
    
    candidates = []
    
    for t_len in range(25, 5, -1):
        if L_len < t_len: continue
        
        left_tsd_cand = left[-t_len:]
        if is_low_complexity(left_tsd_cand): continue
        
        max_mm = max(1, int(t_len * 0.15))
        
        for i in range(len(search_seq) - t_len + 1):
            right_tsd_cand = search_seq[i : i+t_len]
            mm = hamming(left_tsd_cand, right_tsd_cand)
            
            if mm <= max_mm:
                r_start_global = search_start_global + i
                r_end_global = r_start_global + t_len
                
                check_len = 25
                tail_start_global = max(L_len, r_start_global - check_len)
                tail_end_global = r_start_global
                tail_region = full_seq[tail_start_global : tail_end_global]
                
                if not tail_region: continue
                
                cnt_A = tail_region.count('A')
                purity_A = cnt_A / len(tail_region)
                
                max_run, cur = 0, 0
                for c in tail_region:
                    if c == 'A': cur += 1
                    else: max_run = max(max_run, cur); cur = 0
                max_run = max(max_run, cur)
                
                valid_tail = False
                if max_run >= 8: valid_tail = True
                elif max_run >= 5 and purity_A >= 0.6: valid_tail = True
                elif purity_A >= 0.8 and len(tail_region) >= 8: valid_tail = True
                
                if valid_tail:
                    score = t_len * 10 - mm * 5 + max_run * 2
                    candidates.append({
                        'score': score,
                        'polyA': (tail_start_global, tail_end_global),
                        'tsd': (L_len - t_len, L_len, r_start_global, r_end_global)
                    })

    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]['polyA'], candidates[0]['tsd']

    # === 3. 最终决策 ===
    # 如果没找到 TSD，但我们在第一步找到了强 PolyA（且位置合理，在 Core 末尾）
    # 我们依然返回 PolyA，但 TSD 设为 None
    if best_polyA is not None:
        # 检查位置是否在 Core 的 3' 端附近
        # PolyA 的 start 应该在 L_len + C_len 附近
        p_start, p_end = best_polyA
        # 如果 PolyA 确实是在 SINE 的尾巴上 (允许 30bp 误差)
        if abs(p_start - (L_len + C_len)) < 50 or p_start < L_len + C_len:
             return best_polyA, None
    
    return None, None

# ====================== Worker & Init ======================

def worker_init(logA, logB):
    """
    初始化子进程，避免重复传递大的对象（虽然这里数组不大，但好习惯）
    """
    global GLOBAL_LOG_A, GLOBAL_LOG_B
    GLOBAL_LOG_A = logA
    GLOBAL_LOG_B = logB

def process_single_row(row_data, fast_mode=False):
    """
    处理单行数据的 Worker 函数
    """
    # 即使是并行，异常捕获也是必要的，防止单个数据错误导致整个程序崩溃
    try:
        row = row_data
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        label = row['label']
        source = row['source_type']
        strand = row['strand']
        
        unique_id = f"{chrom}:{start}-{end}({strand})"
        
        # 处理序列方向
        if strand == '-':
            core_seq = revcomp(row['seq'])
            left_seq = revcomp(row['flank_right']) # 交换
            right_seq = revcomp(row['flank_left']) # 交换
        else:
            core_seq = row['seq'].upper()
            left_seq = row['flank_left'].upper()
            right_seq = row['flank_right'].upper()
        
        L_offset = len(left_seq)
        n_shuf = 0 if fast_mode else 20
        
        # 使用全局变量里的 LogPWM
        scoreA, posA, zA = scan_best_with_z(core_seq, GLOBAL_LOG_A, n_shuf)
        scoreB, posB, zB = scan_best_with_z(core_seq, GLOBAL_LOG_B, n_shuf)
        
        Z_THRESH = 2.5
        ABS_THRESH = -15.0
        
        final_A_start, final_A_end = -1, -1
        final_B_start, final_B_end = -1, -1
        
        has_A = (posA >= 0 and zA >= Z_THRESH and scoreA > ABS_THRESH)
        has_B = (posB >= 0 and zB >= Z_THRESH and scoreB > ABS_THRESH)
        
        if has_A:
            final_A_start = L_offset + posA
            final_A_end = final_A_start + GLOBAL_LOG_A.shape[1]
            
        if has_B:
            final_B_start = L_offset + posB
            final_B_end = final_B_start + GLOBAL_LOG_B.shape[1]
            
        polyA_res, tsd_res = detect_structure(core_seq, left_seq, right_seq)
        
        # 返回结果字典
        return {
            "chrom": chrom,
            "start": start,
            "end": end,
            "label": label,
            "source_type": source,
            "unique_id": unique_id,
            "left_TSD_start": tsd_res[0] if tsd_res else -1,
            "left_TSD_end": tsd_res[1] if tsd_res else -1,
            "right_TSD_start": tsd_res[2] if tsd_res else -1,
            "right_TSD_end": tsd_res[3] if tsd_res else -1,
            "polyA_start": polyA_res[0] if polyA_res else -1,
            "polyA_end": polyA_res[1] if polyA_res else -1,
            "A_box_start": final_A_start,
            "A_box_end": final_A_end,
            "B_box_start": final_B_start,
            "B_box_end": final_B_end
        }
    except Exception as e:
        print(f"Error processing {row.get('chrom', '?')}:{row.get('start', '?')}: {e}")
        return None

# ====================== 主程序 ======================

def main():
    parser = argparse.ArgumentParser(description="SINE Motif Detector (Parallel)")
    parser.add_argument("--in_csv", required=True, help="Input CSV file")
    parser.add_argument("--out_tsv", required=True, help="Output TSV file")
    parser.add_argument("--fast", action="store_true", help="Skip Z-score shuffling (Very Fast)")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help="Number of threads/processes")
    args = parser.parse_args()

    # 预计算 PWM
    logA = pwm_to_logodds(A_BOX_PWM)
    logB = pwm_to_logodds(B_BOX_PWM)
    
    print(f"Reading input file: {args.in_csv}")
    print(f"Using {args.threads} CPU cores.")
    
    # 1. 读取所有数据到内存 (CSV 通常不会大到撑爆内存，如果真特别大可以用生成器优化)
    # rows = []
    # with open(args.in_csv, "r", encoding="utf-8-sig") as f:
    #     reader = csv.DictReader(f)
    #     rows = list(reader)
    # 读取
    df = pd.read_csv(args.in_csv, encoding="utf-8-sig")

    # 去重：基于指定列，保留第一条(keep='first')
    df_unique = df.drop_duplicates(subset=['chrom', 'start', 'end', 'strand'], keep='first')

    # 转回字典列表 (如果你的后续代码必须要求是 rows = [{}, {}] 格式)
    rows = df_unique.to_dict('records')

    total_rows = len(rows)
    print(f"Total sequences to process: {total_rows}")
    
    # 2. 准备输出
    fieldnames = [
        "chrom", "start", "end", "label", "source_type", "unique_id",
        "left_TSD_start", "left_TSD_end",
        "right_TSD_start", "right_TSD_end",
        "polyA_start", "polyA_end",
        "A_box_start", "A_box_end",
        "B_box_start", "B_box_end"
    ]
    
    # 3. 并行处理
    # 使用 partial 固定 fast_mode 参数
    worker_func = partial(process_single_row, fast_mode=args.fast)
    
    results = []
    
    # 启动进程池
    with multiprocessing.Pool(processes=args.threads, initializer=worker_init, initargs=(logA, logB)) as pool:
        # imap_unordered 通常比 imap 稍微快一点点，且不需要等待前面的任务完成即可 yield
        # chunksize 设置为 100 可以减少 IPC 开销
        iterator = pool.imap(worker_func, rows, chunksize=100)
        
        # 使用 tqdm 显示进度
        for res in tqdm(iterator, total=total_rows, unit="seq"):
            if res is not None:
                results.append(res)
    
    # 4. 写入文件
    print(f"Writing results to {args.out_tsv}...")
    with open(args.out_tsv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print("Done!")

if __name__ == "__main__":
    # Windows 下 multiprocessing 必须放在 if __name__ == "__main__" 下
    multiprocessing.freeze_support()
    main()