#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_training_data.py
=========================
合并多个物种的正负样本FASTA为训练数据

输入:
- 多个正样本FASTA文件
- 多个负样本FASTA文件

输出:
- 合并的FASTA文件，header格式: >unique_id_LABEL
  其中LABEL为 "SINE" 或 "nonSINE"
"""

import argparse
from pathlib import Path
from Bio import SeqIO


def combine_fastas(pos_fastas: list, neg_fastas: list, output_fasta: str):
    """
    合并正负样本FASTA文件
    
    参数:
        pos_fastas: 正样本FASTA文件列表
        neg_fastas: 负样本FASTA文件列表
        output_fasta: 输出文件路径
    """
    total_pos = 0
    total_neg = 0
    
    print("[INFO] 开始合并FASTA文件...")
    
    with open(output_fasta, 'w') as outfile:
        # 处理正样本
        print("\n处理正样本:")
        for i, fasta_file in enumerate(pos_fastas, 1):
            if not Path(fasta_file).exists():
                print(f"  [WARN] 文件不存在: {fasta_file}")
                continue
            
            count = 0
            for record in SeqIO.parse(fasta_file, "fasta"):
                # 修改ID: >originalID_SINE
                record.id = f"{record.id}_SINE"
                record.description = ""
                SeqIO.write(record, outfile, "fasta")
                count += 1
            
            print(f"  [{i}/{len(pos_fastas)}] {Path(fasta_file).name}: {count} 序列")
            total_pos += count
        
        # 处理负样本
        print("\n处理负样本:")
        for i, fasta_file in enumerate(neg_fastas, 1):
            if not Path(fasta_file).exists():
                print(f"  [WARN] 文件不存在: {fasta_file}")
                continue
            
            count = 0
            for record in SeqIO.parse(fasta_file, "fasta"):
                # 修改ID: >originalID_nonSINE
                record.id = f"{record.id}_nonSINE"
                record.description = ""
                SeqIO.write(record, outfile, "fasta")
                count += 1
            
            print(f"  [{i}/{len(neg_fastas)}] {Path(fasta_file).name}: {count} 序列")
            total_neg += count
    
    # 统计
    print("\n" + "="*60)
    print("合并完成")
    print("="*60)
    print(f"正样本 (SINE):    {total_pos}")
    print(f"负样本 (nonSINE): {total_neg}")
    print(f"总计:             {total_pos + total_neg}")
    print(f"正负比例:         {total_pos / (total_neg + 1e-8):.2f}")
    print(f"输出文件:         {output_fasta}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="合并多个物种的正负样本FASTA为训练数据"
    )
    parser.add_argument(
        "--pos_fastas",
        nargs='+',
        required=True,
        help="正样本FASTA文件列表"
    )
    parser.add_argument(
        "--neg_fastas",
        nargs='+',
        required=True,
        help="负样本FASTA文件列表"
    )
    parser.add_argument(
        "--output_fasta",
        required=True,
        help="输出FASTA文件路径"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_fasta).parent.mkdir(parents=True, exist_ok=True)
    
    # 合并
    combine_fastas(args.pos_fastas, args.neg_fastas, args.output_fasta)


if __name__ == "__main__":
    main()