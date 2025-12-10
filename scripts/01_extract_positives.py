#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_positives.py
=======================

本脚本通过使用 minimap2 工具，将一个 SINE 参考序列文库与目标基因组进行比对，
从而从基因组中提取出潜在的 SINE 正样本。

本脚本使用了一个重构过的 SAM 文件解析器，该解析器会将所有输出的序列
标准化为正链方向。

输出文件:
  - <out_prefix>.tsv: 一个表格文件，包含所有提取出的区域信息。
  - <out_prefix>.fa: 一个 FASTA 格式文件，包含完整的序列 (左侧翼-核心序列-右侧翼)。
  - <out_prefix>.sam: 中间步骤生成的比对文件。
"""
import argparse
import pandas as pd
from pathlib import Path

# 从我们重构的本地包中导入所需模块
from sine_classifier.utils import run_command  # ensure_minimap2_index 不再使用
from sine_classifier.sam_parser import parse_sam_and_extract_seqs


def write_output_files(records: list, prefix: str):
    """
    将提取出的记录列表写入 TSV 和 FASTA 文件。
    如果记录列表为空，则会创建带有正确表头的空文件。
    """
    # 定义输出文件的路径
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    tsv_path = f"{prefix}.tsv"
    fa_path = f"{prefix}.fa"

    if not records:
        print("[警告] 没有记录可供写入。正在创建空的输出文件。")
        # 创建一个带表头的空 TSV 文件
        header = ["chrom", "start", "end", "strand", "seq", "flank_left", "flank_right"]
        with open(tsv_path, 'w') as f:
            f.write('\t'.join(header) + '\n')
        # 创建一个空的 FASTA 文件
        open(fa_path, 'w').close()

        print(f"✅ 已写入空文件: {tsv_path}, {fa_path}")
        return

    df = pd.DataFrame(records)

    # 写入 TSV 文件
    df.to_csv(tsv_path, sep='\t', index=False)
    print(f"✅ 已将 {len(df)} 条记录写入 {tsv_path}")

    # 写入 FASTA 文件
    with open(fa_path, 'w') as f:
        for _, row in df.iterrows():
            header = f">{row['chrom']}:{row['start']}-{row['end']}({row['strand']})"
            full_seq = f"{row['flank_left']}{row['seq']}{row['flank_right']}"
            f.write(f"{header}\n{full_seq}\n")
    print(f"✅ 已将 {len(df)} 条序列写入 {fa_path}")


def main():
    parser = argparse.ArgumentParser(
        description="通过将 SINE 参考文库比对到基因组，来提取 SINE 正样本候选项。"
    )
    parser.add_argument("--genome", required=True, help="参考基因组 FASTA 文件的路径。")
    parser.add_argument("--sine_ref", required=True, help="SINE 参考序列 FASTA 文件的路径。")
    parser.add_argument("--out_prefix", default="sine_pos", help="输出文件的前缀。")
    parser.add_argument("--threads", type=int, default=8, help="minimap2 使用的线程数。")
    parser.add_argument("--min_mapq", type=int, default=0, help="一个比对结果所需的最低作图质量值 (mapping quality)。")
    parser.add_argument("--flank", type=int, default=150, help="需要提取的侧翼区域的大小 (单位: bp)。")
    parser.add_argument("--cov_thr", type=float, default=0.8, help="SINE 参考序列的最小覆盖度比例阈值。")
    parser.add_argument("--min_as_score", type=int, default=100, help="Minimum alignment score (AS:i tag) to keep an alignment.")
    parser.add_argument("--max_de_divergence", type=float, default=0.1, help="Maximum sequence divergence (de:f tag) to keep an alignment.")
    args = parser.parse_args()

    genome_path = Path(args.genome)
    # minimap2 索引路径：<genome>.mmi（与你之前日志里的路径保持一致）
    mm2_idx_path = str(genome_path) + ".mmi"

    # 1. 先删除旧索引（如果存在），再用 -k12 -w5 重建索引
    if Path(mm2_idx_path).exists():
        print(f"[信息] 检测到已有 minimap2 索引: {mm2_idx_path}，将先删除以便用 (-k12,-w5) 重建...")
        Path(mm2_idx_path).unlink()

    print(f"[信息] 正在使用 -k12 -w5 重建 minimap2 索引: {mm2_idx_path}")
    build_index_cmd = [
        "minimap2",
        "-d", mm2_idx_path,
        "-k", "19",
        "-w", "5",
        str(genome_path)
    ]
    # 建索引不需要 stdout 重定向
    run_command(build_index_cmd)

    # 2. 运行 minimap2 进行比对（这里不再写 -k/-w，避免覆盖索引参数）
    sam_filepath = f"{args.out_prefix}.sam"
    print(f"[信息] 正在运行 minimap2，将 {args.sine_ref} 比对到 {genome_path}...")
    # minimap2_cmd = [
    #     "minimap2",
    #     "-a",
    #     # -k/-w 已经固化在索引里，这里不要再写
    #     "-p", "0.5",           # 次要比对的得分阈值。保留得分至少为主比对 50% 的结果
    #     "-N", "50000",         # 最多保留 50000 个比对（高拷贝家族）
    #     "--secondary=yes",     # 输出次要比对记录
    #     "--score-N=0",         # 对 N 碱基不惩罚
    #     "-t", str(args.threads),
    #     mm2_idx_path,
    #     args.sine_ref
    # ]
    minimap2_cmd = [
        "minimap2",
        "-a",                    # 输出 SAM（必须）
        "--for-only",               # 只比对正链（参考是正链）
        "--end-bonus=10",        # 强烈推荐！救回大量 5'/3' 端截断的真实插入
        "-A", "2",               # match bonus
        "-B", "4",               # mismatch penalty（适中）
        "-O", "6",               # gap open
        "-E", "1",               # gap extension
        "-p", "0.85",            # 次级比对至少 85% 主得分
        "-N", "100",             # 只保留最好的 100 个（足够了，配合 -p 0.85 后基本都是好比对）
        "--secondary=yes",
        "--score-N=0",
        "-t", str(args.threads),
        mm2_idx_path,
        args.sine_ref
    ]
    with open(sam_filepath, "w") as out_sam:
        run_command(minimap2_cmd, stdout=out_sam)

    # 3. 解析 SAM 文件并提取序列
    print(f"[信息] 正在解析 SAM 文件并提取序列...")
    hits = parse_sam_and_extract_seqs(
        sam_filepath=sam_filepath,
        genome_fa=args.genome,
        min_mapq=None,                     # TE 场景下默认不按 MAPQ 过滤
        flank_size=args.flank,
        min_coverage_ratio=args.cov_thr,
        min_as_score=args.min_as_score,
        max_de_divergence=args.max_de_divergence
    )
    print(f"[信息] 共收集到 {len(hits)} 个高质量的正样本候选项。")

    # 4. 将结果写入输出文件
    write_output_files(hits, args.out_prefix)

    print("\n[成功] 正样本提取完成。")


# 当该脚本被直接执行时，运行 main 函数
if __name__ == "__main__":
    main()
