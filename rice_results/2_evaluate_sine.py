import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SINE Model Performance on Genome Level")
    parser.add_argument("--ref_out", required=True, help="BLAST output of Repbase vs Genome")
    parser.add_argument("--pred_out", required=True, help="BLAST output of Model vs Genome")
    parser.add_argument("--gap", type=int, default=10, help="Max gap to merge nearby hits (bp)")
    parser.add_argument("--full_ratio", type=float, default=0.8, help="Min length ratio (mapped_len / query_len)")
    parser.add_argument("--min_overlap_ref", type=float, default=0.5, help="Min overlap coverage on Reference (%)")
    parser.add_argument("--min_overlap_pred", type=float, default=0.5, help="Min overlap coverage on Prediction (%)")
    parser.add_argument("--output_prefix", default="eval_result", help="Prefix for output files")
    return parser.parse_args()

class GenomicInterval:
    def __init__(self, chrom, start, end, qid, qlen):
        self.chrom = chrom
        self.start = int(start)
        self.end = int(end)
        self.qid = qid
        self.qlen = int(qlen)
        self.matched = False # 标记是否被匹配过
        self.match_id = "-"  # 匹配到的对方ID

    @property
    def length(self):
        return self.end - self.start + 1

    def __repr__(self):
        return f"{self.chrom}:{self.start}-{self.end}"

def load_and_process_blast(filepath, gap, full_ratio):
    """
    解析BLAST，标准化坐标，合并Gap，过滤长度
    """
    # 读取BLAST -outfmt "6 ... qlen"
    # 列: 0:qseqid 1:sseqid 2:pident 3:length 4:mismatch 5:gapopen 6:qstart 7:qend 8:sstart 9:send 10:evalue 11:bitscore 12:qlen
    cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", 
            "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen"]
    
    try:
        df = pd.read_csv(filepath, sep="\t", names=cols)
    except pd.errors.EmptyDataError:
        print(f"[WARN] File {filepath} is empty.")
        return {}

    # 1. 坐标标准化 (处理反向互补，确保 start < end)
    df['g_start'] = df[['sstart', 'send']].min(axis=1)
    df['g_end'] = df[['sstart', 'send']].max(axis=1)

    # 按染色体和起始位置排序
    df = df.sort_values(by=['sseqid', 'g_start'])

    # 按 Query ID 分组处理合并
    # 注意：同一个SINE可能在基因组由多条BLAST记录组成（因为Gap或Indel）
    # 但不同的SINE拷贝是独立的。
    # 这里为了简化，我们先按 (sseqid, qseqid) 分组，即同一个序列比对到同一条染色体的记录
    
    grouped = df.groupby(['sseqid', 'qseqid'])
    
    final_intervals = defaultdict(list) # {chrom: [Intervals]}

    for (chrom, qid), group in grouped:
        group = group.sort_values('g_start')
        
        # Merging logic
        merged_hits = []
        if group.empty: continue
        
        # 初始化第一个块
        recs = group.to_dict('records')
        curr_start = recs[0]['g_start']
        curr_end = recs[0]['g_end']
        curr_qlen = recs[0]['qlen']
        
        for i in range(1, len(recs)):
            row = recs[i]
            # 如果当前记录的 start 在 上一个 end + gap 范围内
            if row['g_start'] <= curr_end + gap:
                curr_end = max(curr_end, row['g_end'])
            else:
                merged_hits.append(GenomicInterval(chrom, curr_start, curr_end, qid, curr_qlen))
                curr_start = row['g_start']
                curr_end = row['g_end']
        # 添加最后一个
        merged_hits.append(GenomicInterval(chrom, curr_start, curr_end, qid, curr_qlen))

        # Filtering logic
        for interval in merged_hits:
            if full_ratio > 0:
                ratio = interval.length / interval.qlen
                if ratio < full_ratio:
                    continue
            final_intervals[chrom].append(interval)

    # 对每个染色体内的区间按位置排序，方便后续比较
    for chrom in final_intervals:
        final_intervals[chrom].sort(key=lambda x: x.start)
        
    return final_intervals

def calculate_stats(ref_dict, pred_dict, min_ov_ref, min_ov_pred):
    """
    比较 Reference 和 Prediction 的区间
    """
    tp_list = []
    fp_list = []
    fn_list = []

    all_chroms = set(ref_dict.keys()) | set(pred_dict.keys())

    for chrom in all_chroms:
        refs = ref_dict.get(chrom, [])
        preds = pred_dict.get(chrom, [])

        # 简单的双重循环 (鉴于每个染色体上的SINE数量通常在几千级别，是可以接受的)
        # 也可以使用 intervaltree 优化，但为了减少依赖这里用纯逻辑
        
        # 1. 遍历 Pred 寻找匹配的 Ref (TP 和 FP)
        for p in preds:
            best_match = None
            max_ov_len = 0
            
            for r in refs:
                # 快速跳过不重叠的
                if r.end < p.start: continue
                if r.start > p.end: break # 因为已排序，后面都不可能重叠

                # 计算重叠
                ov_start = max(p.start, r.start)
                ov_end = min(p.end, r.end)
                ov_len = max(0, ov_end - ov_start + 1)

                if ov_len > 0:
                    cov_r = ov_len / r.length
                    cov_p = ov_len / p.length
                    
                    if cov_r >= min_ov_ref and cov_p >= min_ov_pred:
                        if ov_len > max_ov_len:
                            max_ov_len = ov_len
                            best_match = r

            if best_match:
                # TP
                p.matched = True
                p.match_id = best_match.qid
                # 注意：一个 Ref 可能被多个 Pred 匹配（例如模型把一个大SINE切成了两个），
                # 或者多个 Ref 匹配一个 Pred。
                # 这里我们简单标记 Ref 也被匹配了
                best_match.matched = True 
                if best_match.match_id == "-":
                    best_match.match_id = p.qid
                else:
                    best_match.match_id += f";{p.qid}"

                tp_list.append({
                    'type': 'TP',
                    'chrom': chrom,
                    'p_start': p.start, 'p_end': p.end, 'p_id': p.qid,
                    'r_start': best_match.start, 'r_end': best_match.end, 'r_id': best_match.qid,
                    'overlap_len': max_ov_len,
                    'coverage_pred': max_ov_len / p.length,
                    'coverage_ref': max_ov_len / best_match.length
                })
            else:
                # FP
                fp_list.append({
                    'type': 'FP',
                    'chrom': chrom,
                    'p_start': p.start, 'p_end': p.end, 'p_id': p.qid,
                    'r_start': '-', 'r_end': '-', 'r_id': '-',
                    'overlap_len': 0, 'coverage_pred': 0, 'coverage_ref': 0
                })

        # 2. 遍历 Ref 寻找未被匹配的 (FN)
        # 注意：TP已经在上面处理了，这里只找 matched == False
        for r in refs:
            if not r.matched:
                fn_list.append({
                    'type': 'FN',
                    'chrom': chrom,
                    'p_start': '-', 'p_end': '-', 'p_id': '-',
                    'r_start': r.start, 'r_end': r.end, 'r_id': r.qid,
                    'overlap_len': 0, 'coverage_pred': 0, 'coverage_ref': 0
                })
    
    return tp_list, fp_list, fn_list

def generate_library_summary(intervals_dict, is_pred_lib=True):
    """
    生成库的汇总信息: 序列名 -> 基因组拷贝数 -> 匹配情况
    """
    summary = defaultdict(lambda: {'copies': 0, 'matched_copies': 0, 'matched_ids': []})
    
    for chrom, interval_list in intervals_dict.items():
        for interval in interval_list:
            sid = interval.qid
            summary[sid]['copies'] += 1
            if interval.matched:
                summary[sid]['matched_copies'] += 1
                summary[sid]['matched_ids'].append(interval.match_id)
            else:
                summary[sid]['matched_ids'].append("-")
    
    rows = []
    for sid, data in summary.items():
        # 统计最常见的匹配ID
        from collections import Counter
        matched_ids_clean = [x for x in data['matched_ids'] if x != '-']
        most_common_match = Counter(matched_ids_clean).most_common(1)
        top_match = most_common_match[0][0] if most_common_match else "-"
        
        rows.append({
            'seq_id': sid,
            'genome_copies': data['copies'],
            'matched_copies': data['matched_copies'],
            'match_ratio': data['matched_copies'] / data['copies'] if data['copies'] > 0 else 0,
            'top_match_id': top_match
        })
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    
    print("1. Parsing Reference BLAST (Ground Truth)...")
    ref_intervals = load_and_process_blast(args.ref_out, args.gap, args.full_ratio)
    
    print("2. Parsing Prediction BLAST (Model Output)...")
    pred_intervals = load_and_process_blast(args.pred_out, args.gap, args.full_ratio)
    
    print("3. Comparing Intervals...")
    tp, fp, fn = calculate_stats(ref_intervals, pred_intervals, args.min_overlap_ref, args.min_overlap_pred)
    
    # --- 输出 1: 详细列表 (TP/FP/FN) ---
    all_details = tp + fp + fn
    df_details = pd.DataFrame(all_details)
    df_details = df_details[['type', 'chrom', 'p_id', 'p_start', 'p_end', 
                             'r_id', 'r_start', 'r_end', 'overlap_len', 'coverage_pred', 'coverage_ref']]
    df_details.to_csv(f"{args.output_prefix}_details.csv", sep="\t", index=False)
    
    # --- 输出 2 & 3: 库汇总 ---
    print("4. Generating Summaries...")
    pred_lib_summary = generate_library_summary(pred_intervals, is_pred_lib=True)
    pred_lib_summary.to_csv(f"{args.output_prefix}_pred_lib_summary.csv", sep="\t", index=False)
    
    ref_lib_summary = generate_library_summary(ref_intervals, is_pred_lib=False)
    ref_lib_summary.to_csv(f"{args.output_prefix}_ref_lib_summary.csv", sep="\t", index=False)
    
    # --- 输出 4: 全局统计 ---
    # 计算碱基数
    total_ref_bases = sum(r.length for chrom in ref_intervals for r in ref_intervals[chrom])
    tp_bases = sum(x['overlap_len'] for x in tp) # TP碱基按重叠长度算
    # 或者 TP碱基按预测的长度算？通常 Precision = TP_len / (TP_len + FP_len)
    # 这里我们分别统计
    
    pred_tp_bases = sum(x['p_end'] - x['p_start'] + 1 for x in tp)
    pred_fp_bases = sum(x['p_end'] - x['p_start'] + 1 for x in fp)
    ref_fn_bases = sum(x['r_end'] - x['r_start'] + 1 for x in fn)
    
    with open(f"{args.output_prefix}_global_stats.txt", "w") as f:
        f.write("Global Evaluation Statistics\n")
        f.write("============================\n")
        f.write(f"Ref (Ground Truth) Total Bases: {total_ref_bases}\n")
        f.write(f"Ref Total Copies: {sum(len(v) for v in ref_intervals.values())}\n\n")
        
        f.write(f"TP (True Positive) Count: {len(tp)}\n")
        f.write(f"TP Bases (Overlap): {tp_bases}\n")
        f.write(f"TP Bases (From Prediction): {pred_tp_bases}\n\n")
        
        f.write(f"FP (False Positive) Count: {len(fp)}\n")
        f.write(f"FP Bases: {pred_fp_bases}\n\n")
        
        f.write(f"FN (False Negative) Count: {len(fn)}\n")
        f.write(f"FN Bases: {ref_fn_bases}\n\n")
        
        # Calculate Precision / Recall / F1 (Base level)
        prec = tp_bases / (pred_tp_bases + pred_fp_bases) if (pred_tp_bases + pred_fp_bases) > 0 else 0
        rec = tp_bases / total_ref_bases if total_ref_bases > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        f.write(f"Base-level Precision: {prec:.4f}\n")
        f.write(f"Base-level Recall:    {rec:.4f}\n")
        f.write(f"Base-level F1-Score:  {f1:.4f}\n")

    print(f"Done! Results saved with prefix: {args.output_prefix}")

if __name__ == "__main__":
    main()