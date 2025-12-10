#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_e2e_classifier.py
==========================
【终极无敌版 - 修复ID匹配】SINE 端到端分类器训练脚本

修复日志:
1. 修复 train_ids.txt 读取时未去除后缀导致的匹配失败问题 (0样本bug)。
2. 增加 FileHandler 确保日志写入文件。
3. 增加数据加载的 Debug 打印。
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from sine_classifier.data import SINEDatasetE2E, collate_fn
from sine_classifier.model import MotifGuidedSINEClassifier, FocalLoss

logger = logging.getLogger(__name__)

def setup_logging(output_dir, rank):
    """配置日志：同时输出到控制台和文件"""
    handlers = [logging.StreamHandler()]
    if rank == 0:
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(output_dir) / "training.log"
            handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

def setup_ddp():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group(backend='nccl')
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def load_sine_data(fasta_file: str):
    sequences_with_ids = []
    labels = []
    label_mapping = {'SINE': 1, 'nonSINE': 0}

    # 读取 FASTA，提取纯净 ID (去除 _SINE 后缀)
    for record in SeqIO.parse(fasta_file, "fasta"):
        try:
            if '_' in record.id:
                unique_id, label_name = record.id.rsplit('_', 1)
                if label_name in label_mapping:
                    sequences_with_ids.append((unique_id, str(record.seq).upper()))
                    labels.append(label_mapping[label_name])
        except ValueError:
            pass
    return sequences_with_ids, labels

def set_backbone_freeze(model, freeze: bool):
    raw_model = model.module if hasattr(model, "module") else model
    for param in raw_model.backbone.parameters():
        param.requires_grad = not freeze

@torch.no_grad()
def evaluate(model, loader, criterion, device, rank, world_size):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    # 防止除以零：如果 loader 为空（例如单卡没有数据分配），直接返回
    if len(loader) == 0:
        return 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="Evaluating", disable=(rank != 0)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask, motif_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    if len(all_probs) > 0:
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
    else:
        all_probs = torch.empty((0, 2))
        all_preds = torch.empty((0,), dtype=torch.long)
        all_labels = torch.empty((0,), dtype=torch.long)

    if world_size > 1:
        local_size = torch.tensor([all_labels.size(0)], dtype=torch.long, device=device)
        size_list = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        
        sizes = [s.item() for s in size_list]
        max_size = max(sizes)
        total_samples = sum(sizes)
        
        # 防止所有卡都没有数据
        if total_samples == 0:
            return 0.0, 0.0, 0.0, 0.0

        prob_buffer = torch.zeros((max_size, 2), dtype=all_probs.dtype, device=device)
        prob_buffer[:local_size] = all_probs.to(device)
        gathered_probs = [torch.zeros_like(prob_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_probs, prob_buffer)
        
        pred_buffer = torch.zeros((max_size,), dtype=all_preds.dtype, device=device)
        pred_buffer[:local_size] = all_preds.to(device)
        gathered_preds = [torch.zeros_like(pred_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_preds, pred_buffer)
        
        label_buffer = torch.zeros((max_size,), dtype=all_labels.dtype, device=device)
        label_buffer[:local_size] = all_labels.to(device)
        gathered_labels = [torch.zeros_like(label_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_labels, label_buffer)
        
        loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        # 注意：这里分母应该是总 batch 数，近似处理为本地 len(loader) * world_size
        # 更严谨是 gather 步数，但只要 loss 趋势对即可
        total_loss_global = loss_tensor.item()
        batch_count_tensor = torch.tensor([len(loader)], device=device)
        dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_global / batch_count_tensor.item()

        if rank == 0:
            final_probs = []
            final_preds = []
            final_labels = []
            for i, size in enumerate(sizes):
                final_probs.append(gathered_probs[i][:size].cpu())
                final_preds.append(gathered_preds[i][:size].cpu())
                final_labels.append(gathered_labels[i][:size].cpu())
            
            all_probs = torch.cat(final_probs).numpy()
            all_preds = torch.cat(final_preds).numpy()
            all_labels = torch.cat(final_labels).numpy()
        else:
            return 0, 0, 0, 0
    else:
        avg_loss = total_loss / len(loader)
        all_probs = all_probs.numpy()
        all_preds = all_preds.numpy()
        all_labels = all_labels.numpy()

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.0

    return avg_loss, acc, f1, auc

def main():
    parser = argparse.ArgumentParser(description="【终极版】SINE 端到端分类器训练")
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--sine_data_path", required=True)
    parser.add_argument("--motif_data_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--split_dir", type=str, default=None)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # 初始化日志
    setup_logging(args.output_dir, rank)

    if rank == 0:
        logger.info("="*60)
        logger.info(f"启动训练: Epochs={args.epochs}, Freeze={args.freeze_epochs}")
        logger.info(f"GPUs: {world_size}")
        logger.info("="*60)

    # 加载模型 & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)

    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        freeze_backbone=True 
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 加载数据
    if rank == 0: logger.info("加载数据...")
    sequences_with_ids, labels = load_sine_data(args.sine_data_path)
    
    motif_df = pd.read_csv(args.motif_data_path, sep='\t')
    if 'unique_id' not in motif_df.columns:
        motif_df['unique_id'] = motif_df.apply(
            lambda row: f"{row['chrom']}:{row['original_start']}-{row['original_end']}({row['strand']})", axis=1)
    motif_df = motif_df.drop_duplicates(subset=['unique_id']).set_index('unique_id', drop=False)

    # === 关键修复：ID 匹配逻辑 ===
    if args.split_dir and os.path.exists(args.split_dir):
        def load_ids(p):
            ids = set()
            with open(p) as f:
                for line in f:
                    l = line.strip()
                    if not l: continue
                    # 关键：尝试去除后缀，使其与 sequences_with_ids 中的 ID 格式一致
                    try: 
                        uid, suffix = l.rsplit('_', 1)
                        if suffix in ('SINE', 'nonSINE'):
                            ids.add(uid)
                        else:
                            ids.add(l)
                    except ValueError:
                        ids.add(l.strip())
            return ids
        
        t_ids = load_ids(os.path.join(args.split_dir, "train_ids.txt"))
        v_ids = load_ids(os.path.join(args.split_dir, "val_ids.txt"))
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        # Debug: 打印前几个 ID 帮助排查
        if rank == 0:
            logger.info(f"Example Data ID: {sequences_with_ids[0][0] if sequences_with_ids else 'None'}")
            logger.info(f"Example Split ID: {list(t_ids)[0] if t_ids else 'None'}")
        
        for (uid, seq), lbl in zip(sequences_with_ids, labels):
            if uid in t_ids:
                X_train.append((uid, seq)); y_train.append(lbl)
            elif uid in v_ids:
                X_val.append((uid, seq)); y_val.append(lbl)
                
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            sequences_with_ids, labels, test_size=0.2, stratify=labels, random_state=42
        )

    if rank == 0:
        logger.info(f"训练集: {len(X_train)} | 验证集: {len(X_val)}")
        if len(X_train) == 0:
            logger.error("❌ 严重错误：训练集为空！请检查 ID 匹配逻辑。程序退出。")
            sys.exit(1)

    train_ds = SINEDatasetE2E(X_train, y_train, motif_df, tokenizer, args.max_length)
    val_ds = SINEDatasetE2E(X_val, y_val, motif_df, tokenizer, args.max_length)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Focal Loss
    if len(y_train) > 0:
        # pos_ratio = sum(y_train) / len(y_train) if len(y_train) > 0 else 0.5
        # alpha = 1 - pos_ratio  # 少数类获得更高权重
        # 或者更稳健的做法：
        alpha = 0.25  # 经典配置，适用于正样本较少的情况
        criterion = FocalLoss(gamma=2.0, alpha=alpha).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    best_val_f1 = 0.0
    optimizer = None

    # 在进入 epoch 循环前，先创建 optimizer（默认全解冻的情况）
    optimizer = torch.optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr, 'weight_decay': 0.01},
        {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
        {'params': model.module.classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
        {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 
        'lr': args.head_lr, 'weight_decay': 0.1}
    ])

    current_phase = "unfreeze"  # 标记当前阶段

    for epoch in range(1, args.epochs + 1):
        # 只有在真正切换阶段时才重新设置冻结和优化器
        if epoch == 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=True)
            # 只优化 head
            optimizer = torch.optim.AdamW([
                {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr},
                {'params': model.module.classifier.parameters(), 'lr': args.head_lr},
                {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 'lr': args.head_lr},
            ], weight_decay=0.1)
            current_phase = "freeze"
            if rank == 0: logger.info("Phase 1: 冻结 Backbone")

        elif epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=False)
            # 重新创建包含 backbone 的优化器
            optimizer = torch.optim.AdamW([
                {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr, 'weight_decay': 0.01},
                {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
                {'params': model.module.classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
                {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 'lr': args.head_lr},
            ])
            current_phase = "unfreeze"
            if rank == 0: logger.info("Phase 2: 解冻 Backbone")

        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=(rank != 0))
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            motif_mask = batch['motif_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, motif_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if rank == 0: pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, rank, world_size)

        if rank == 0:
            avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            phase = "Freeze" if epoch <= args.freeze_epochs else "Unfreeze"
            
            logger.info(f"Epoch {epoch} [{phase}] | Train Loss: {avg_train_loss:.4f} | "
                        f"Val F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.module.state_dict(), Path(args.output_dir) / "best_model.pt")
                logger.info(f"  ✅ New Best F1! Saved.")
            
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_val_f1
            }, Path(args.output_dir) / "latest.pt")

    if rank == 0:
        logger.info("训练完成！")
    cleanup_ddp()

if __name__ == "__main__":
    main()