#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_e2e_classifier.py
==========================
真正的端到端SINE分类器训练

关键修复:
1. ❌ 不使用预计算的embedding
2. ✅ 每个batch动态提取特征
3. ✅ Backbone参与训练，权重会更新
4. ✅ 梯度可以反向传播到Backbone
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from Bio import SeqIO

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sine_classifier.data import SINEDatasetE2E, collate_fn
from sine_classifier.model import MotifGuidedSINEClassifier, FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_ddp():
    """初始化DDP"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ["WORLD_SIZE"])


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def load_sine_data(fasta_file: str):
    """
    从FASTA加载数据
    
    格式: >unique_id_LABEL
    """
    sequences_with_ids = []
    labels = []
    label_mapping = {'SINE': 1, 'nonSINE': 0}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        try:
            unique_id, label_name = record.id.rsplit('_', 1)
            if label_name in label_mapping:
                sequences_with_ids.append((unique_id, str(record.seq).upper()))
                labels.append(label_mapping[label_name])
        except ValueError:
            logger.warning(f"无法解析标签: {record.id}")
    
    return sequences_with_ids, labels


def train_epoch(
    model, 
    loader, 
    criterion, 
    optimizer, 
    device, 
    epoch, 
    rank, 
    world_size
):
    """
    训练一个epoch
    
    关键: 每个batch中，序列会被送入backbone动态提取特征
    """
    model.train()
    
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for batch_idx, batch in enumerate(iterator):
        # 将数据移到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播（关键：backbone会在这里动态提取特征）
        logits = model(input_ids, attention_mask, motif_mask)
        loss = criterion(logits, labels)
        
        # 反向传播（梯度会传到backbone）
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新参数（包括backbone的参数）
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if rank == 0:
            iterator.set_postfix({'loss': f'{loss.item():.12f}'})
    
    # 同步指标
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    
    if world_size > 1:
        metrics = torch.tensor([avg_loss, acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
        avg_loss, acc = metrics.cpu().numpy()
    
    return float(avg_loss), float(acc)


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank):
    """评估模型"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    iterator = tqdm(loader, desc="Evaluating", disable=(rank != 0))
    
    for batch in iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播（backbone仍会提取特征，但不计算梯度）
        logits = model(input_ids, attention_mask, motif_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    
    # 计算指标
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError:
        auc = 0.0
    
    return avg_loss, acc, f1, auc


def main():
    parser = argparse.ArgumentParser(
        description="端到端SINE分类器训练（正确实现）"
    )
    
    # 数据参数
    parser.add_argument("--backbone_path", required=True, help="Plant-NT backbone路径")
    parser.add_argument("--sine_data_path", required=True, help="训练数据FASTA")
    parser.add_argument("--motif_data_path", required=True, help="Motif坐标TSV")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Backbone学习率")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="分类头学习率")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--freeze_backbone", action='store_true', help="冻结backbone")
    
    parser.add_argument("--split_dir", type=str, default=None, 
                        help="包含 train_ids.txt 和 val_ids.txt 的目录。如果未提供，则使用随机划分（不推荐）。")
    args = parser.parse_args()
    
    # DDP初始化
    rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        logger.info("="*80)
        logger.info("端到端SINE分类器训练")
        logger.info("="*80)
        logger.info(f"使用{world_size}个GPU进行DDP训练")
        logger.info(f"Backbone学习率: {args.backbone_lr}")
        logger.info(f"分类头学习率: {args.head_lr}")
        logger.info(f"Backbone冻结: {args.freeze_backbone}")
        logger.info("="*80)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer和backbone
    if rank == 0:
        logger.info("\n加载模型和tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone_path, 
        trust_remote_code=True
    )
    backbone = AutoModelForMaskedLM.from_pretrained(
        args.backbone_path, 
        trust_remote_code=True
    )
    
    # 创建分类器
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone  # 根据参数决定是否冻结
    ).to(device)
    
    # DDP包装
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\n模型参数:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # 加载数据
    if rank == 0:
        logger.info("\n加载训练数据...")
    
    sequences_with_ids, labels = load_sine_data(args.sine_data_path)
    
    if rank == 0:
        logger.info(f"  总样本数: {len(sequences_with_ids)}")
        logger.info(f"  SINE样本: {sum(labels)}")
        logger.info(f"  nonSINE样本: {len(labels) - sum(labels)}")
    
    motif_df = pd.read_csv(args.motif_data_path, sep='\t')
    
    # 创建unique_id列（如果没有）
    if 'unique_id' not in motif_df.columns:
        motif_df['unique_id'] = motif_df.apply(
            lambda row: f"{row['chrom']}:{row['original_start']}-{row['original_end']}({row['strand']})",
            axis=1
        )
    motif_df.drop_duplicates(subset=['unique_id'], keep='first', inplace=True)
    
    # 数据集划分
    # X_train, X_val, y_train, y_val = train_test_split(
    #     sequences_with_ids, labels,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=labels
    # )

    X_train, X_val, y_train, y_val = [], [], [], []

    if args.split_dir and os.path.exists(args.split_dir):
        if rank == 0:
            logger.info(f"使用预定义的簇划分 (CD-HIT): {args.split_dir}")
        
        train_id_file = os.path.join(args.split_dir, "train_ids.txt")
        val_id_file = os.path.join(args.split_dir, "val_ids.txt")
        
        # 读取 ID 集合
        def load_ids_set(file_path):
            ids = set()
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # 关键修复: 需要与 load_sine_data 的逻辑保持一致
                    # 将 'ID_LABEL' 格式分割为 'ID'，去掉后缀
                    try:
                        uid, _ = line.rsplit('_', 1)
                        ids.add(uid)
                    except ValueError:
                        # 如果没有下划线，则假设整行就是 ID
                        ids.add(line)
            return ids
            
        # 读取 ID 集合
        train_ids_set = load_ids_set(train_id_file)
        val_ids_set = load_ids_set(val_id_file)
            
        # 构建查找字典
        # 为了加快速度，先把 data 转换成 dict: uid -> (seq, label)
        # 注意 load_sine_data 返回的是 [(uid, seq), ...], labels 是 [label, ...]
        data_map = {}
        for (uid, seq), lbl in zip(sequences_with_ids, labels):
            data_map[uid] = (uid, seq, lbl)
            
        # 填充训练集
        for uid in train_ids_set:
            if uid in data_map:
                u, s, l = data_map[uid]
                X_train.append((u, s))
                y_train.append(l)
        
        # 填充验证集
        for uid in val_ids_set:
            if uid in data_map:
                u, s, l = data_map[uid]
                X_val.append((u, s))
                y_val.append(l)
                
    else:
        if rank == 0:
            logger.warning("未找到划分文件，使用随机划分 (警告: 可能存在数据泄露!)")
        # 回退到随机划分
        X_train, X_val, y_train, y_val = train_test_split(
            sequences_with_ids, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

    if rank == 0:
        logger.info(f"\n数据集划分完成:")
        logger.info(f"  训练集: {len(X_train)}")
        logger.info(f"  验证集: {len(X_val)}")
        if len(X_train) == 0 or len(X_val) == 0:
            logger.error("错误: 训练集或验证集为空！请检查 ID 匹配情况。")
            sys.exit(1)
    
    # 创建Dataset（不预计算embedding）
    train_dataset = SINEDatasetE2E(
        X_train, y_train, motif_df, tokenizer, args.max_length
    )
    val_dataset = SINEDatasetE2E(
        X_val, y_val, motif_df, tokenizer, args.max_length
    )
    
    # DataLoader with DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 优化器（根据是否冻结backbone调整）
    if args.freeze_backbone:
        # 只优化分类头
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.head_lr,
            weight_decay=0.01
        )
        if rank == 0:
            logger.info("\n优化器: 仅训练分类头")
    else:
        # 分别设置backbone和分类头的学习率
        optimizer = torch.optim.AdamW([
            {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr},
            {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr},
            {'params': model.module.classifier.parameters(), 'lr': args.head_lr}
        ], weight_decay=0.01)
        if rank == 0:
            logger.info("\n优化器: Backbone + 分类头联合训练")
            logger.info(f"  Backbone LR: {args.backbone_lr}")
            logger.info(f"  Head LR: {args.head_lr}")
    
    # criterion = nn.CrossEntropyLoss()
    class_weights = torch.tensor([3.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 训练循环
    best_val_f1 = 0.0
    
    if rank == 0:
        logger.info("\n开始训练...")
        logger.info("="*80)
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, rank, world_size
        )
        
        # 验证
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model.module, val_loader, criterion, device, rank
        )
        
        if rank == 0:
            logger.info(
                f"Epoch {epoch:2d}/{args.epochs} | "
                f"Train: Loss={train_loss:.20f} Acc={train_acc:.4f} | "
                f"Val: Loss={val_loss:.20f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}"
            )
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_path = output_dir / "best_model.pt"
                torch.save(model.module.state_dict(), save_path)
                logger.info(f"  ✅ 保存最佳模型 (F1: {val_f1:.4f})")

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_f1': val_f1,
            }

            history_ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, history_ckpt_path)

            # 4. 每轮都覆盖保存最新的检查点 (latest.pt) - 方便Snakemake失败后手动恢复
            latest_ckpt_path = output_dir / "latest.pt"
            torch.save(checkpoint_data, latest_ckpt_path)

            logger.info(f"  💾 已保存检查点: {history_ckpt_path.name}")
    
    if rank == 0:
        logger.info("="*80)
        logger.info(f"训练完成! 最佳F1: {best_val_f1:.4f}")
        logger.info("="*80)
    
    cleanup_ddp()


if __name__ == "__main__":
    main()