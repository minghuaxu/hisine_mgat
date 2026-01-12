#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_detect_motifs_parallel.py
============================
功能：SINE Motif 并行检测脚本
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

def calculate_complexity(seq):
    """简单计算序列复杂度，避免匹配到 poly-N"""
    if not seq: return 0
    return len(set(seq)) / len(seq)

def detect_structure(core, left, right):
    """
    修复版结构检测：
    强制验证 Core 结尾与 Right TSD 之间的 'Gap' 必须是 PolyA/Tail。
    防止 TSD 匹配到远处，导致将中间的非相关序列误判为 PolyA。
    """
    core = core.upper()
    left = left.upper()
    right = right.upper()
    
    full_seq = left + core + right
    
    L_len = len(left)
    C_len = len(core)
    
    # === 参数设置 ===
    min_tsd_len = 4
    max_tsd_len = 25
    search_limit = 60    # 稍微缩减搜索范围，100bp 对 SINE 尾部通常太远了，除非有超长 PolyA
    overlap_search = 15  # 允许 Core 注释偏后，向 Core 内部回溯的距离
    
    candidates = []

    # 预计算 Left TSD 候选 (Left Flank 的末端)
    # 为了性能，只计算一次
    left_tsds = {} # len -> seq
    for t_len in range(min_tsd_len, max_tsd_len + 1):
        cand = left[-t_len:]
        if not is_low_complexity(cand):
            left_tsds[t_len] = cand

    # 构建右侧搜索空间：从 Core 末端回溯 15bp 开始，往右延伸
    # 搜索串 = Core尾部片段 + Right Flank
    search_seq_base = core[max(0, C_len - overlap_search):] + right[:search_limit]
    
    # 这里的 offset 是 search_seq_base[0] 在全局坐标系(L+C+R)中的绝对位置
    search_seq_global_offset = L_len + max(0, C_len - overlap_search)
    
    # 理论上的 Core 结束位置 (全局坐标)
    annotated_core_end = L_len + C_len

    # === 1. TSD 驱动搜索 ===
    # 从长到短遍历
    for t_len in range(max_tsd_len, min_tsd_len - 1, -1):
        if t_len not in left_tsds: continue
        left_tsd = left_tsds[t_len]
        
        # 允许错配数
        max_mm = int(t_len * 0.20) #稍微收紧错配标准
        
        # 在右侧滑动寻找
        # 限制：Right TSD 不应该离 Core 结尾太远，除非中间全是 A
        for i in range(len(search_seq_base) - t_len + 1):
            right_tsd_cand = search_seq_base[i : i+t_len]
            mm = hamming(left_tsd, right_tsd_cand)
            
            if mm <= max_mm:
                # 计算 Right TSD 在全局的坐标
                r_start_global = search_seq_global_offset + i
                r_end_global = r_start_global + t_len
                
                # === 关键修复：Gap 验证 ===
                # Gap 是从 [理论 Core 结束] 到 [Right TSD 开始] 的区域
                # 注意：r_start_global 可能小于 annotated_core_end (说明 TSD 在 Core 内部，这是允许的)
                
                # 定义潜在的 PolyA 区域：
                # 起点：取 (annotated_core_end - 20) 和 (r_start_global - 50) 的较后者
                # 意思是：我们重点看 Right TSD 紧挨着的那一段，且不能脱离 Core 太远
                poly_check_start = max(L_len, min(annotated_core_end - 15, r_start_global - 30))
                poly_check_end = r_start_global
                
                # 提取这段序列 (可能跨越 Core 和 Right)
                
                gap_seq = full_seq[poly_check_start : poly_check_end]
                
                # 1. 距离惩罚：计算 TSD 距离“理论 Core 结尾”有多远
                # 如果 r_start_global 远大于 annotated_core_end，说明 TSD 漂移到了右侧深处
                dist_from_core = r_start_global - annotated_core_end
                
                valid_gap = True
                is_poly_a = False
                
                if len(gap_seq) == 0:
                    a_content = 0
                else:
                    a_content = gap_seq.count('A') / len(gap_seq)
                    t_content = gap_seq.count('T') / len(gap_seq)
                    
                    # === 强逻辑校验 ===
                    # 如果 TSD 在 Core 结尾之后很远 (>10bp)，由于中间必须是 PolyA
                    # 所以如果 A content 不够高，这绝对是个错误的匹配
                    if dist_from_core > 10:
                        if max(a_content, t_content) < 0.6: # 要求至少 60% 是 A/T
                            valid_gap = False
                    
                    if max(a_content, t_content) > 0.4:
                        is_poly_a = True

                if not valid_gap:
                    continue # 跳过这个错误的 TSD 候选

                # === 打分 ===
                # 基础分：长度 - 错配
                score = (t_len * 2) - (mm * 4)
                
                # PolyA 加分
                if is_poly_a:
                    score += 6
                
                # 距离惩罚：离 Core 越远扣分越多 (防止匹配到无限远)
                if dist_from_core > 0:
                    score -= (dist_from_core * 0.1) 
                
                # 完美匹配奖励
                if mm == 0: score += 3
                
                candidates.append({
                    'score': score,
                    'tsd': (L_len - t_len, L_len, r_start_global, r_end_global),
                    'polyA_range': (poly_check_start, poly_check_end),
                    'debug_dist': dist_from_core
                })

    # === 2. 选择最佳结果 ===
    best_tsd = None
    best_polyA = None
    
    if candidates:
        # 按分数排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_cand = candidates[0]
        
        # 阈值判定
        if best_cand['score'] >= 8: 
            best_tsd = best_cand['tsd']
            # 细化 PolyA：简单返回 Gap 区域即可
            best_polyA = best_cand['polyA_range']

    # === 3. 兜底策略 (无 TSD 但有强 PolyA) ===
    if best_tsd is None:
        # 仅在 Core 结尾附近寻找
        check_region_start = max(0, annotated_core_end - 20)
        check_region_end = min(len(full_seq), annotated_core_end + 30)
        region = full_seq[check_region_start : check_region_end]
        
        # 必须是连续的 A，或者高密度的 A
        if "AAAAA" in region or region.count('A') >= 8:
             # 这种情况下，我们不知道 TSD 在哪，但确认有 PolyA
             # 可以返回 PolyA 坐标，TSD 留空
             # 这里的坐标需要转换回 global
             match = re.search(r'A{5,}', region)
             if match:
                 s = check_region_start + match.start()
                 e = check_region_start + match.end()
                 best_polyA = (s, e)
            
    return best_polyA, best_tsd

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