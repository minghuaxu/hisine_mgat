#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_predict.py
=============
使用训练好的模型进行SINE预测 (修复版)

修复点:
1. 导入正确的 Dataset 类 (SINEDatasetE2E)
2. 增加 collate_fn 支持
3. 修复模型权重加载 (去除 DDP 的 module. 前缀)
4. 修正 Dataset 参数名称
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
# [修复1] 导入 SINEDatasetE2E 和 collate_fn
from sine_classifier.data import SINEDatasetE2E, collate_fn
from sine_classifier.model import MotifGuidedSINEClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_preprocessed_data(tsv_file: str):
    """
    加载预处理的TSV文件
    """
    df = pd.read_csv(tsv_file, sep='\t')
    
    # 确保有unique_id列
    if 'unique_id' not in df.columns:
        df['unique_id'] = df.apply(
            lambda row: f"{row['chrom']}:{row['start']}-{row['end']}({row['strand']})",
            axis=1
        )
    
    return df


def main():
    parser = argparse.ArgumentParser(description="SINE预测")
    
    # 模型参数
    parser.add_argument("--model_path", required=True, help="训练好的模型路径")
    parser.add_argument("--backbone_path", required=True, help="Backbone路径")
    parser.add_argument("--hidden_dim", type=int, required=True, help="隐藏层维度")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout率")
    
    # 数据参数
    parser.add_argument("--input_fasta", required=True, help="输入FASTA（含侧翼）")
    parser.add_argument("--motif_coords", required=True, help="Motif坐标TSV")
    parser.add_argument("--preprocessed_tsv", required=True, help="预处理TSV（含原始seq）")
    parser.add_argument("--output_prefix", required=True, help="输出前缀")
    
    # 预测参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--threshold", type=float, default=0.99, help="SINE判定阈值")

    # 指定显卡
    parser.add_argument("--gpu", type=int, default=3, help="使用的GPU编号")
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 1. 加载模型
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)
    
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout
    ).to(device)
    
    # [修复2] 鲁棒的权重加载 (处理 DDP module. 前缀)
    try:
        state_dict = torch.load(args.model_path, map_location=device)

        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict'] # 处理 checkpoint 包含多个部分的情况
        
        # 检查是否包含 module. 前缀
        if any(k.startswith('module.') for k in state_dict.keys()):
            logger.info("检测到DDP权重，正在移除 'module.' 前缀...")
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    model.eval()
    logger.info("模型加载完成")
    
    # 2. 加载数据
    logger.info("加载数据...")
    
    # Motif坐标
    motif_df = pd.read_csv(args.motif_coords, sep='\t')
    if 'unique_id' not in motif_df.columns:
        # 注意：这里必须与 detect_motifs.py 的输出一致
        motif_df['unique_id'] = motif_df.apply(
            lambda row: f"{row['chrom']}:{row['original_start']}-{row['original_end']}({row['strand']})",
            axis=1
        )
    # 去重
    motif_df.drop_duplicates(subset=['unique_id'], keep='first', inplace=True)
    
    # 完整序列（用于预测）
    sequences_with_ids = [
        (record.id, str(record.seq).upper()) 
        for record in SeqIO.parse(args.input_fasta, "fasta")
    ]
    
    # 原始数据（用于输出FASTA，保留原始核心序列）
    original_data = load_preprocessed_data(args.preprocessed_tsv).set_index('unique_id')
    
    # [修复3] 使用 SINEDatasetE2E 并修正参数名称
    pred_dataset = SINEDatasetE2E(
        sequences_with_ids=sequences_with_ids,
        labels=None,  # 预测模式，无标签
        motif_df=motif_df,
        tokenizer=tokenizer,
        max_token_length=args.max_length  # 注意：data.py 中参数名为 max_token_length
    )
    
    # [修复4] 增加 collate_fn
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn  # 必须使用自定义整理函数
    )
    
    # 3. 执行预测
    logger.info(f"开始预测 {len(pred_dataset)} 条序列...")
    all_ids = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="预测中"):
            # 数据移至GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            motif_mask = batch['motif_mask'].to(device)
            # logger.warning("!!! TESTING MODE: Forcing flat mask !!!")
            # motif_mask = torch.full_like(batch['motif_mask'], 0.1).to(device) 
            # # 注意：要把 padding 部分恢复为 0.0，否则长度信息会乱
            # is_padding = (batch['input_ids'] == tokenizer.pad_token_id).to(device)
            # motif_mask[is_padding] = 0.0
            
            # 前向传播
            logits = model(input_ids, attention_mask, motif_mask)
            probs = torch.softmax(logits, dim=1)
            
            # batch['unique_ids'] 是列表，直接扩展
            all_ids.extend(batch['unique_ids']) 
            all_probs.append(probs.cpu().numpy())
    
    if len(all_probs) == 0:
        logger.warning("没有生成任何预测结果，请检查输入数据。")
        return

    all_probs = np.vstack(all_probs)
    
    # 4. 整理结果
    results_df = pd.DataFrame({
        'id': all_ids,
        'prob_nonSINE': all_probs[:, 0],
        'prob_SINE': all_probs[:, 1],
    })
    results_df['prediction'] = np.where(
        results_df['prob_SINE'] >= args.threshold, 
        'SINE', 
        'non-SINE'
    )
    
    # 保存TSV
    output_tsv = f"{args.output_prefix}_predictions.tsv"
    results_df.to_csv(output_tsv, sep='\t', index=False)
    logger.info(f"✅ 预测结果已保存: {output_tsv}")
    
    # 保存SINE FASTA (只保存预测为 SINE 的序列)
    positive_results = results_df[results_df['prediction'] == 'SINE']
    
    fasta_records = []
    for _, row in positive_results.iterrows():
        seq_id = row['id']
        prob = row['prob_SINE']
        
        # 从原始数据获取 core 序列 (不含侧翼)
        if seq_id in original_data.index:
            core_seq = original_data.loc[seq_id, 'seq']
            # Header格式: ID#SINE#概率
            new_header = f"{seq_id}#SINE#{prob:.4f}"
            record = SeqRecord(Seq(core_seq), id=new_header, description="")
            fasta_records.append(record)
    
    output_fasta = f"{args.output_prefix}_predicted_SINEs.fa"
    if fasta_records:
        with open(output_fasta, "w") as f:
            SeqIO.write(fasta_records, f, "fasta")
        logger.info(f"✅ 预测为SINE的FASTA已保存: {output_fasta} ({len(fasta_records)}条)")
    else:
        logger.info("⚠️ 没有预测出任何 SINE 序列，FASTA文件未生成。")
    
    # 统计信息
    logger.info("\n预测统计:")
    logger.info(f"  总序列数: {len(results_df)}")
    logger.info(f"  预测 SINE: {len(positive_results)}")
    logger.info(f"  预测 non-SINE: {len(results_df) - len(positive_results)}")
    logger.info(f"  平均 SINE 概率: {results_df['prob_SINE'].mean():.4f}")


if __name__ == "__main__":
    main()