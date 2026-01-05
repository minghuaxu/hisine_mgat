#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_e2e_classifier.py (Fixed Multi-Task)
=============================================
修复内容：
1. 添加 --train_motif_tsv 参数以适配新 Data.py
2. 修复 Evaluate 函数签名，传入两个 Loss
3. 修复 Optimizer，加入 token_classifier 参数
4. 修复 Loss 定义和 Batch 解包
"""
import argparse
import logging
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
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
from tqdm import tqdm
from Bio.Seq import Seq
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from sine_classifier.data import SINEDatasetE2E, collate_fn
from sine_classifier.model import MotifGuidedSINEClassifier, FocalLoss

logger = logging.getLogger(__name__)

def revcomp(s):
    if pd.isna(s) or s == "": return ""
    return str(Seq(s).reverse_complement())

def setup_logging(output_dir, rank):
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

def set_backbone_freeze(model, freeze: bool):
    raw_model = model.module if hasattr(model, "module") else model
    for param in raw_model.backbone.parameters():
        param.requires_grad = not freeze

def process_csv_to_data_list(csv_path, rank):
    if rank == 0: logger.info(f"Processing data from: {csv_path}")
    df = pd.read_csv(csv_path)
    sequences_with_ids = []
    labels = []
    
    for _, row in df.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        strand = row['strand']
        label = int(row['label'])
        
        uid = f"{chrom}:{start}-{end}({strand})"
        
        if strand == '-':
            seq_concat = revcomp(row['flank_right']) + revcomp(row['seq']) + revcomp(row['flank_left'])
        else:
            seq_concat = row['flank_left'] + row['seq'] + row['flank_right']
            
        sequences_with_ids.append((uid, seq_concat))
        labels.append(label)
        
    return sequences_with_ids, labels

@torch.no_grad()
def evaluate(model, loader, criterion_cls, criterion_seg, device, rank, world_size):
    """
    [修复] 接收两个 Loss 函数，计算总 Loss
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    if len(loader) == 0:
        return 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="Evaluating", disable=(rank != 0)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)
        token_labels = batch['token_labels'].to(device) # [修复] 解包 Token Labels

        # [修复] 接收两个输出
        global_logits, token_logits = model(input_ids, attention_mask, motif_mask)

        # Loss 1: 全局分类
        loss_cls = criterion_cls(global_logits, labels)

        # Loss 2: 序列分割 (Flatten计算)
        loss_seg = criterion_seg(token_logits.view(-1, 5), token_labels.view(-1))

        # 组合 Loss
        loss = loss_cls + 0.5 * loss_seg 
        total_loss += loss.item()

        probs = torch.softmax(global_logits, dim=1)
        preds = torch.argmax(global_logits, dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    # 聚合逻辑保持不变...
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
        
        if sum(sizes) == 0: return 0.0, 0.0, 0.0, 0.0

        # Gather Probs
        prob_buffer = torch.zeros((max_size, 2), dtype=all_probs.dtype, device=device)
        prob_buffer[:local_size] = all_probs.to(device)
        gathered_probs = [torch.zeros_like(prob_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_probs, prob_buffer)
        
        # Gather Preds
        pred_buffer = torch.zeros((max_size,), dtype=all_preds.dtype, device=device)
        pred_buffer[:local_size] = all_preds.to(device)
        gathered_preds = [torch.zeros_like(pred_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_preds, pred_buffer)
        
        # Gather Labels
        label_buffer = torch.zeros((max_size,), dtype=all_labels.dtype, device=device)
        label_buffer[:local_size] = all_labels.to(device)
        gathered_labels = [torch.zeros_like(label_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_labels, label_buffer)
        
        # Reduce Loss
        loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        batch_count_tensor = torch.tensor([len(loader)], device=device)
        dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / batch_count_tensor.item()

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
    parser = argparse.ArgumentParser(description="SINE Train (Multi-Task)")
    parser.add_argument("--backbone_path", required=True)
    
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--train_mask", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_mask", required=True)
    
    # [新增] 必须传入 TSV 路径，否则 Dataset 会报错
    parser.add_argument("--train_motif_tsv", required=True, help="Path to train_motif_pos.tsv")
    parser.add_argument("--val_motif_tsv", required=True, help="Path to val_motif_pos.tsv")
    
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    setup_logging(args.output_dir, rank)

    if rank == 0:
        logger.info("="*60)
        logger.info(f"启动训练 (Multi-Task: Cls + Seg): Epochs={args.epochs}")
        logger.info("="*60)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True)

    # 加载 CSV 数据
    X_train, y_train = process_csv_to_data_list(args.train_csv, rank)
    X_val, y_val = process_csv_to_data_list(args.val_csv, rank)

    if rank == 0:
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")

    # [修复] 初始化 Dataset (传入 motif_tsv_path)
    if rank == 0: logger.info("Initializing Datasets...")
    
    train_ds = SINEDatasetE2E(
        sequences_with_ids=X_train,
        labels=y_train,
        mask_path=args.train_mask,
        motif_tsv_path=args.train_motif_tsv, # 传入 TSV
        tokenizer=tokenizer,
        max_token_length=args.max_length,
        is_training=True
    )
    
    val_ds = SINEDatasetE2E(
        sequences_with_ids=X_val,
        labels=y_val,
        mask_path=args.val_mask,
        motif_tsv_path=args.val_motif_tsv,   # 传入 TSV
        tokenizer=tokenizer,
        max_token_length=args.max_length,
        is_training=False
    )

    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        freeze_backbone=True
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False) # 建议设为 False 除非有未用参数

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # [新增] 定义两个 Loss
    # Loss 1: 分类 Loss
    if len(y_train) > 0:
        alpha = 0.25
        criterion_cls = FocalLoss(gamma=2.0, alpha=alpha).to(device)
    else:
        criterion_cls = nn.CrossEntropyLoss().to(device)

    # Loss 2: 分割 Loss (忽略 Padding=4)
    # 类别: 0=Bg, 1=TSD, 2=Body, 3=PolyA, 4=Pad
    criterion_seg = nn.CrossEntropyLoss(ignore_index=4).to(device)

    best_val_f1 = 0.0
    
    # [修复] Optimizer: 必须包含 token_classifier
    optimizer = torch.optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr, 'weight_decay': 0.01},
        {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
        {'params': model.module.classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
        {'params': model.module.token_classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1}, # 新增
        {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 
        'lr': args.head_lr, 'weight_decay': 0.1}
    ])

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=True)
            optimizer = torch.optim.AdamW([
                {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr},
                {'params': model.module.classifier.parameters(), 'lr': args.head_lr},
                {'params': model.module.token_classifier.parameters(), 'lr': args.head_lr}, # 新增
                {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 'lr': args.head_lr},
            ], weight_decay=0.1)
            if rank == 0: logger.info("Phase 1: 冻结 Backbone")

        elif epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=False)
            optimizer = torch.optim.AdamW([
                {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr, 'weight_decay': 0.01},
                {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
                {'params': model.module.classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
                {'params': model.module.token_classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1}, # 新增
                {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 'lr': args.head_lr},
            ])
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
            token_labels = batch['token_labels'].to(device) # [修复] 解包

            # [修复] 接收两个输出
            global_logits, token_logits = model(input_ids, attention_mask, motif_mask)
            
            # [修复] 计算两个 Loss
            loss_cls = criterion_cls(global_logits, labels)
            
            # (Batch, Len, 5) -> (Batch*Len, 5) vs (Batch*Len)
            loss_seg = criterion_seg(token_logits.view(-1, 5), token_labels.view(-1))
            
            # 联合 Loss
            loss = loss_cls + 0.5 * loss_seg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            
            if rank == 0: pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_cls.item():.3f}',
                'seg': f'{loss_seg.item():.3f}'
            })

        # [修复] Evaluate 传参
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model, val_loader, criterion_cls, criterion_seg, device, rank, world_size
        )

        if rank == 0:
            avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            phase = "Freeze" if epoch <= args.freeze_epochs else "Unfreeze"
            logger.info(f"Epoch {epoch} [{phase}] | Train Loss: {avg_train_loss:.4f} | "
                        f"Val F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

            # 保存 Checkpoint
            epoch_save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': val_f1
            }, epoch_save_path)

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