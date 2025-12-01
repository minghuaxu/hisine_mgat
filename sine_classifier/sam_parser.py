#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam_parser.py (TE-friendly Version)
===================================
- 修复 Minimap2 标签兼容性 (dv vs de)
- 添加详细过滤统计信息
- 针对高拷贝 TE / SINE 的场景：默认不按 MAPQ 过滤，
  主要依赖 Coverage / Divergence / AS 控制比对质量
"""

import re
from typing import List, Dict, Any, Optional

from pyfaidx import Fasta
from sine_classifier.utils import revcomp


def _calculate_aln_len_from_cigar(cigar: str) -> int:
    """
    根据 CIGAR 计算在参考序列上的比对长度。

    规则：
    - 统计 M, =, X, D 这些 op 的长度（消耗 reference 的 op）
    - 忽略 I, S, H, P, N 等不消耗 reference 或特殊标记
    """
    matches = re.findall(r'(\d+)([MIDNSHP=X])', cigar)
    aln_len = 0
    for length, op in matches:
        if op in ('M', '=', 'X', 'D'):
            aln_len += int(length)
    return aln_len


def _parse_optional_tags(tags_list: List[str]) -> Dict[str, Any]:
    """
    解析 SAM 记录中第 12 列以后的可选标签字段，返回字典。

    每个 tag 一般形如： "AS:i:123" / "dv:f:0.123"
    - i: int
    - f: float
    - 其他类型统一当作 str
    """
    tags_dict: Dict[str, Any] = {}
    for tag in tags_list:
        try:
            parts = tag.split(':')
            if len(parts) < 3:
                continue
            key = parts[0]
            type_code = parts[1]
            val_str = parts[2]

            if type_code == 'i':
                tags_dict[key] = int(val_str)
            elif type_code == 'f':
                tags_dict[key] = float(val_str)
            else:
                tags_dict[key] = val_str
        except ValueError:
            # 解析失败就跳过该 tag
            continue
    return tags_dict


def parse_sam_and_extract_seqs(
    sam_filepath: str,
    genome_fa: str,
    min_mapq: Optional[int] = None,       # ★ 默认不按 MAPQ 过滤（TE 场景下 MAPQ 通常很低）
    flank_size: int = 150,
    min_coverage_ratio: float = 0.5,      # 默认覆盖度阈值 0.5，可视情况调高
    min_as_score: Optional[int] = None,   # 可选：按 AS 分数过滤
    max_de_divergence: Optional[float] = None  # 可选：按 dv/de 过滤（如 0.25）
) -> List[Dict[str, Any]]:
    """
    解析 SAM，比对到基因组，抽取比对的 core + 左右侧翼序列。

    返回 records 列表，每个元素是:
    {
        "chrom": rname,
        "start": start,
        "end": end,
        "strand": "+" / "-",
        "seq": core_seq(已经按 5'->3' 方向规范化),
        "flank_left":  左侧翼（上游，5' 方向）,
        "flank_right": 右侧翼（下游，3' 方向）
    }

    注意：
    - 对负链比对会做 revcomp，并且把侧翼重新映射到“真实 5'/3'”方向
    """
    genome = Fasta(genome_fa, sequence_always_upper=True)
    records: List[Dict[str, Any]] = []

    # --- 统计计数器 ---
    stats = {
        "total": 0,
        "kept": 0,
        "failed_mapq": 0,
        "failed_cov": 0,
        "failed_as": 0,
        "failed_div": 0,
        "unmapped": 0,
    }

    print("[INFO] Starting SAM parsing...")

    with open(sam_filepath, "r") as f:
        for line in f:
            if line.startswith("@"):
                # header 直接跳过
                continue

            stats["total"] += 1
            cols = line.strip().split("\t")
            if len(cols) < 11:
                # 非法行
                continue

            qname, flag_str, rname, pos_str, mapq_str, cigar, _, _, _, seq, *opt_tags = cols

            # 1. 过滤未比对
            if rname == "*":
                stats["unmapped"] += 1
                continue

            # --- 基础量计算 ---
            try:
                mapq = int(mapq_str)
            except ValueError:
                mapq = 0

            query_len = len(seq)
            aln_len = _calculate_aln_len_from_cigar(cigar)

            if query_len == 0:
                stats["failed_cov"] += 1
                continue

            coverage = aln_len / query_len

            # 2. 覆盖度过滤（对 TE 拷贝完整性更关键）
            if coverage < min_coverage_ratio:
                stats["failed_cov"] += 1
                continue

            # 3. 解析可选标签（AS/dv/de 等）
            tags = _parse_optional_tags(opt_tags)

            # 4. Alignment Score 过滤（可选）
            if min_as_score is not None:
                as_score = tags.get("AS")
                # 没有 AS 标签时，可以选择视为通过；这里只在有 AS 且低于阈值时过滤
                if as_score is not None and as_score < min_as_score:
                    stats["failed_as"] += 1
                    continue

            # 5. Divergence 过滤（dv / de，可选）
            if max_de_divergence is not None:
                divergence = tags.get("dv")
                if divergence is None:
                    divergence = tags.get("de")
                if divergence is not None and divergence > max_de_divergence:
                    stats["failed_div"] += 1
                    continue

            # 6. MAPQ 过滤（高拷贝 TE 一般建议关闭或非常宽松）
            if min_mapq is not None and mapq < min_mapq:
                stats["failed_mapq"] += 1
                continue

            # --- 通过所有过滤，提取序列 ---
            try:
                pos = int(pos_str) - 1  # SAM 是 1-based，内部用 0-based
                flag = int(flag_str)
                strand = "-" if (flag & 16) else "+"

                if rname not in genome:
                    # 染色体名称不匹配
                    continue

                chrom_seq_obj = genome[rname]
                chrom_len = len(chrom_seq_obj)

                start = pos
                end = pos + aln_len

                # 侧翼提取边界
                extract_start = max(0, start - flank_size)
                extract_end = min(chrom_len, end + flank_size)

                core_seq = chrom_seq_obj[start:end].seq
                left_flank = chrom_seq_obj[extract_start:start].seq
                right_flank = chrom_seq_obj[end:extract_end].seq

                if strand == "-":
                    # 负链：比对方向为反向互补，这里要规范到 SINE 的 5'->3' 方向
                    core_seq = revcomp(core_seq)

                    # 负链时，基因组右侧是 5' 上游，基因组左侧是 3' 下游
                    real_upstream = revcomp(right_flank)
                    real_downstream = revcomp(left_flank)

                    final_left = real_upstream   # 5' 上游
                    final_right = real_downstream  # 3' 下游
                    final_core = core_seq
                else:
                    # 正链：基因组左为上游，右为下游，直接使用即可
                    final_left = left_flank
                    final_right = right_flank
                    final_core = core_seq

                records.append({
                    "chrom": rname,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "seq": str(final_core),
                    "flank_left": str(final_left),
                    "flank_right": str(final_right),
                })
                stats["kept"] += 1

            except Exception as e:
                print(f"[WARN] Error processing read {qname}: {e}")
                continue

    # 打印详细统计信息
    print("\n" + "=" * 40)
    print("FILTERING STATISTICS")
    print("=" * 40)
    print(f"Total alignments in SAM:   {stats['total']}")
    print(f"Unmapped (*):              {stats['unmapped']}")

    if min_mapq is not None:
        print(f"Failed MAPQ (<{min_mapq}):         {stats['failed_mapq']}")
    else:
        print(f"Failed MAPQ:                {stats['failed_mapq']} (filter disabled)")

    print(f"Failed Coverage (<{min_coverage_ratio:.2f}): {stats['failed_cov']}")

    if min_as_score is not None:
        print(f"Failed AS Score (<{min_as_score}):      {stats['failed_as']}")
    else:
        print(f"Failed AS Score:             {stats['failed_as']} (filter disabled)")

    if max_de_divergence is not None:
        print(f"Failed Divergence (>{max_de_divergence}): {stats['failed_div']}")
    else:
        print(f"Failed Divergence:           {stats['failed_div']} (filter disabled)")

    print("-" * 40)
    print(f"FINAL KEPT:                {stats['kept']}")
    print("=" * 40 + "\n")

    return records
