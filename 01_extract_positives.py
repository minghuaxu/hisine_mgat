#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_positives.py (RM-Validated Version)
==============================================

逻辑更新：
1. Minimap2 提取候选序列 (保留 TSD/PolyA 边界)。
2. (可选) 运行 RepeatMasker 进行验证。
3. 利用 RepeatMasker 的 .out 文件清洗数据：
   仅保留那些在 RM 结果中也标记为 SINE 的序列。

"""
import argparse
import pandas as pd
import subprocess
import os
import shutil
from pathlib import Path
from collections import defaultdict

# 从我们重构的本地包中导入所需模块
from tools.utils import run_command
from tools.sam_parser import parse_sam_and_extract_seqs


def run_repeatmasker(fasta_path, threads=8, species=None, lib=None):
    """
    对提取的 FASTA 文件运行 RepeatMasker。
    返回生成的 .out 文件路径。
    """
    print(f"\n[RM] 正在对 {fasta_path} 运行 RepeatMasker...")
    
    cmd = ["RepeatMasker", "-pa", str(threads), "-nolow", "-no_is", "-norna"]
    
    if lib:
        cmd.extend(["-lib", lib])
    elif species:
        cmd.extend(["-species", species])
    else:
        # 如果都没提供，默认使用自带库或假设用户已配置好环境
        # 这里建议用户至少提供一个
        print("[RM 警告] 未提供 -species 或 -lib，RepeatMasker 将使用默认配置。")

    cmd.append(str(fasta_path))
    
    try:
        run_command(cmd)
        out_file = str(fasta_path) + ".out"
        if Path(out_file).exists():
            return out_file
        else:
            raise FileNotFoundError(f"RepeatMasker 完成但未生成输出文件: {out_file}")
    except Exception as e:
        print(f"[RM 错误] RepeatMasker 运行失败: {e}")
        return None


def parse_rm_out_file(rm_out_path):
    """
    解析 RepeatMasker .out 文件，提取所有 SINE 的坐标区间。
    由于 RM 是针对 Minimap2 提取的片段跑的，这里的坐标是相对于片段的 (local coordinates)。
    
    返回: dict { sequence_id: [(start, end), (start, end)...] }
    注意: RM 坐标是 1-based inclusive, 这里转换为 0-based [start, end)
    """
    print(f"[信息] 正在解析 RepeatMasker 输出文件: {rm_out_path}")
    rm_intervals = defaultdict(list)
    
    with open(rm_out_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("SW") or line.startswith("score"):
                continue
            
            parts = line.split()
            if len(parts) < 15:
                continue
                
            # 解析列 (标准 .out 格式)
            # score, div, del, ins, query_seq, q_begin, q_end, (left), strand, repeat, class/fam ...
            try:
                q_name = parts[4]
                q_start = int(parts[5])
                q_end = int(parts[6])
                repeat_name = parts[9]   # 第 10 列是 Repeat Name (0-based index 9)
                repeat_class = parts[10] # 第 11 列是 Class/Family (0-based index 10)
                # 核心过滤：只认 RepeatMasker 说是 SINE 的区域
                # 你也可以根据需要放宽，比如 "SINE" 或 "DNA/ transposon" (如果是 MITEs)
                if "SINE" in repeat_class.upper() or "SINE" in repeat_name.upper():
                    # 转换为 0-based
                    rm_intervals[q_name].append((q_start - 1, q_end))
                    
            except ValueError:
                continue
                
    return rm_intervals


def filter_hits_by_rm(hits, rm_intervals, min_overlap_ratio=0.5):
    """
    根据 RM 的覆盖度过滤 Minimap2 的 hits。
    
    逻辑：
    Minimap2 提取了一个片段 (Sequence A)。
    我们在 Sequence A 上跑了 RM。
    如果 RM 在 Sequence A 上标注的 SINE 区域总长度 < Sequence A 长度 * 50%，则丢弃。
    这意味着这个序列虽然长得像 SINE (Minimap2)，但 RM 并不认可它是 SINE。
    """
    kept_hits = []
    stats = {"total": len(hits), "pass": 0, "fail": 0}
    
    print(f"[验证] 正在利用 RM 结果验证 {len(hits)} 个候选序列 (阈值: >{min_overlap_ratio*100}% 覆盖)...")
    
    for hit in hits:
        # 构造 ID，必须与 write_output_files 中生成 FASTA header 的逻辑一致
        # Minimap2 提取后的 FASTA ID 格式通常是: >chrom:start-end(strand)
        # 但我们在跑 RM 时，输入的是生成的 temp fasta。
        # 这里我们需要一种方式把 hits 和 RM 结果对应起来。
        # 最简单的方式：在跑 RM 之前，先把 hits 写成临时 FASTA，用简单的 ID (0, 1, 2...) 或者完整 ID。
        
        # 为了稳健，我们使用完整 ID 匹配
        seq_id = f"{hit['chrom']}:{hit['start']}-{hit['end']}({hit['strand']})"
        
        seq_len = len(hit['seq']) # 这是核心序列长度
        # 注意：这里有个复杂的点。
        # 如果我们把 (FlankL + Core + FlankR) 丢给 RM 跑，RM 的坐标是基于全长的。
        # 如果我们只把 Core 丢给 RM 跑，RM 坐标是基于 Core 的。
        # 通常为了验证 SINE 本体，建议只验证 Core 部分，或者看 Core 部分是否被 RM 覆盖。
        
        # *假定*：我们在 main 函数里生成的临时 FASTA 是包含 Flank 的 (为了完整性)。
        # 那么 RM 的坐标里，SINE 应该出现在中间位置。
        # 简化策略：计算 RM SINE 区间 与 整个序列 的重叠长度。
        # 但这样有一个 Bug：如果 Flank 里恰好有一个 SINE，怎么办？
        # 改进策略：计算 RM SINE 区间与 "预期 Core 区间" 的重叠。
        
        # 我们的 sam_parser 输出包含 'seq' (Core), 'flank_left', 'flank_right'
        flank_l_len = len(hit['flank_left'])
        core_len = len(hit['seq'])
        
        # 预期的 Core 在临时序列中的区间
        expected_core_start = flank_l_len
        expected_core_end = flank_l_len + core_len
        
        intervals = rm_intervals.get(seq_id, [])
        
        # 计算交集长度
        overlap_bases = 0
        # 创建一个布尔数组或集合来处理重叠区间的并集（防止 RM 输出重叠片段导致重复计算）
        covered_indices = set()
        
        for (r_start, r_end) in intervals:
            # 计算 RM 区间 (r_start, r_end) 与 预期核心区间 (expected_core_start, expected_core_end) 的交集
            intersect_start = max(r_start, expected_core_start)
            intersect_end = min(r_end, expected_core_end)
            
            if intersect_end > intersect_start:
                for i in range(intersect_start, intersect_end):
                    covered_indices.add(i)
        
        overlap_len = len(covered_indices)
        ratio = overlap_len / core_len if core_len > 0 else 0
        
        if ratio >= min_overlap_ratio:
            kept_hits.append(hit)
            stats["pass"] += 1
        else:
            stats["fail"] += 1
            # 调试：打印几个失败的例子
            if stats["fail"] <= 5:
                print(f"   [剔除] {seq_id}: RM 覆盖度 {ratio:.2f} (长度 {core_len}, RM重叠 {overlap_len})")
    
    print(f"[验证完成] 保留: {stats['pass']}, 剔除: {stats['fail']}")
    return kept_hits


def write_temp_fasta(hits, out_path):
    """写入临时 FASTA 用于运行 RM"""
    with open(out_path, 'w') as f:
        for hit in hits:
            # ID 必须唯一且无特殊字符干扰 shell
            header = f"{hit['chrom']}:{hit['start']}-{hit['end']}({hit['strand']})"
            # 写入全序列 (Flank + Core + Flank)
            full_seq = f"{hit['flank_left']}{hit['seq']}{hit['flank_right']}"
            f.write(f">{header}\n{full_seq}\n")


def write_output_files(records: list, prefix: str):
    """
    将提取出的记录列表写入 TSV 和 FASTA 文件。
    """
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    tsv_path = f"{prefix}.tsv"
    fa_path = f"{prefix}.fa"

    if not records:
        print("[警告] 没有记录可供写入。")
        return

    df = pd.DataFrame(records)
    df.to_csv(tsv_path, sep='\t', index=False)
    
    with open(fa_path, 'w') as f:
        for _, row in df.iterrows():
            header = f">{row['chrom']}:{row['start']}-{row['end']}({row['strand']})"
            full_seq = f"{row['flank_left']}{row['seq']}{row['flank_right']}"
            f.write(f">{header}\n{full_seq}\n")
            
    print(f"✅ 最终结果已保存: {tsv_path} ({len(df)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Minimap2 提取 + RepeatMasker 验证流程")
    parser.add_argument("--genome", required=True, help="参考基因组 FASTA")
    parser.add_argument("--sine_ref", required=True, help="SINE 参考库 (用于 Minimap2)")
    parser.add_argument("--out_prefix", default="sine_pos", help="输出前缀")
    parser.add_argument("--threads", type=int, default=8)
    
    # Minimap2 参数
    parser.add_argument("--flank", type=int, default=150)
    parser.add_argument("--cov_thr", type=float, default=0.8)
    parser.add_argument("--min_as_score", type=int, default=100)
    parser.add_argument("--max_de_divergence", type=float, default=0.3)
    
    # RM 验证参数
    parser.add_argument("--run_rm", action="store_true", help="是否对提取的候选序列运行 RepeatMasker")
    parser.add_argument("--rm_species", help="RepeatMasker -species 参数 (如 'Oryza sativa')")
    parser.add_argument("--rm_lib", help="RepeatMasker -lib 参数 (自定义 TE 库)")
    parser.add_argument("--existing_rm_out", help="如果已有 RM 输出文件 (.out)，直接提供路径以跳过运行")
    parser.add_argument("--rm_overlap_thr", type=float, default=0.5, help="核心序列被 RM 标记为 SINE 的最小长度比例 (0.0-1.0)")

    args = parser.parse_args()

    genome_path = Path(args.genome)
    mm2_idx_path = str(genome_path) + ".mmi"

    # 1. 先删除旧索引（如果存在），再用 -k12 -w5 重建索引
    if Path(mm2_idx_path).exists():
        print(f"[信息] 检测到已有 minimap2 索引: {mm2_idx_path}，将先删除以便用 (-k12,-w5) 重建...")
        Path(mm2_idx_path).unlink()

    print(f"[信息] 正在使用 -k12 -w5 重建 minimap2 索引: {mm2_idx_path}")
    build_index_cmd = [
        "minimap2",
        "-d", mm2_idx_path,
        "-k", "12",
        "-w", "5",
        str(genome_path)
    ]
    # 建索引不需要 stdout 重定向
    run_command(build_index_cmd)

    # 2. 运行 minimap2 进行比对（这里不再写 -k/-w，避免覆盖索引参数）
    sam_filepath = f"{args.out_prefix}.sam"
    # 使用你之前定义的优化参数
    minimap2_cmd = [
        "minimap2",
        "-a",                    # 输出 SAM（必须）
        "--for-only",               # 只比对正链（参考是正链）
        "--end-bonus=10",        # 强烈推荐！救回大量 5'/3' 端截断的真实插入
        "-A", "2",               # match bonus
        "-B", "4",               # mismatch penalty（适中）
        "-O", "6",               # gap open
        "-E", "1",               # gap extension
        "-p", "0.1",            # 次级比对至少 85% 主得分
        "-N", "50000",             # 只保留最好的 100 个（足够了，配合 -p 0.85 后基本都是好比对）
        "--secondary=yes",
        "--score-N=0",
        "-t", str(args.threads),
        mm2_idx_path,
        args.sine_ref
    ]
    with open(sam_filepath, "w") as out_sam:
        run_command(minimap2_cmd, stdout=out_sam)
        
    print(f"[信息] 解析 SAM 文件...")
    hits = parse_sam_and_extract_seqs(
        sam_filepath=sam_filepath,
        genome_fa=args.genome,
        flank_size=args.flank,
        min_coverage_ratio=args.cov_thr,
        min_as_score=args.min_as_score,
        max_de_divergence=args.max_de_divergence
    )
    print(f"[信息] Minimap2 提取候选数: {len(hits)}")

    if not hits:
        print("未提取到任何候选序列，退出。")
        return

    # 2. 准备 RM 验证 (New Steps)
    rm_out_file = args.existing_rm_out
    
    print(args.run_rm)
    # 如果没有现成的 RM 文件，且要求运行 RM
    if not rm_out_file and args.run_rm:
        # 先把候选序列写成临时 FASTA
        temp_fa = f"{args.out_prefix}_temp_candidates.fa"
        write_temp_fasta(hits, temp_fa)
        
        # 运行 RM
        rm_out_file = run_repeatmasker(temp_fa, threads=args.threads, species=args.rm_species, lib=args.rm_lib)
    
    # 3. 执行过滤
    if rm_out_file:
        if not Path(rm_out_file).exists():
            print(f"[错误] 找不到 RM 输出文件: {rm_out_file}")
            return
            
        rm_intervals = parse_rm_out_file(rm_out_file)
        
        # 核心过滤
        final_hits = filter_hits_by_rm(hits, rm_intervals, min_overlap_ratio=args.rm_overlap_thr)
        
        # 写入最终结果
        write_output_files(final_hits, args.out_prefix + "_raw_rm_validated")
        
        # 清理临时文件
        if args.run_rm and Path(temp_fa).exists():
            os.remove(temp_fa)
    else:
        print("[警告] 未提供 RM 结果且未开启 RM 运行，仅输出 Minimap2 原始结果 (含潜在假阳性)。")
        write_output_files(hits, args.out_prefix)

if __name__ == "__main__":
    main()