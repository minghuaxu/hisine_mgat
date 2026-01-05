#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
# 屏蔽 Transformers 的警告
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
    """DNA 反向互补"""
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
    """辅助函数：统一处理 CSV 读取逻辑"""
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
        
        # 必须与 build_masks.py 逻辑一致
        uid = f"{chrom}:{start}-{end}({strand})"
        
        if strand == '-':
            seq_concat = revcomp(row['flank_right']) + revcomp(row['seq']) + revcomp(row['flank_left'])
        else:
            seq_concat = row['flank_left'] + row['seq'] + row['flank_right']
            
        sequences_with_ids.append((uid, seq_concat))
        labels.append(label)
        
    return sequences_with_ids, labels

@torch.no_grad()
def evaluate(model, loader, criterion, device, rank, world_size):
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
    parser = argparse.ArgumentParser(description="SINE Train (Pre-split)")
    parser.add_argument("--backbone_path", required=True)
    
    # === 修改处: 明确指定 Train 和 Val 的文件路径 ===
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--train_mask", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_mask", required=True)
    
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
        logger.info(f"启动训练 (Pre-split Mode): Epochs={args.epochs}, Freeze={args.freeze_epochs}")
        logger.info("="*60)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True)

    # 2. 分别加载 Train 和 Val 数据
    X_train, y_train = process_csv_to_data_list(args.train_csv, rank)
    X_val, y_val = process_csv_to_data_list(args.val_csv, rank)

    if rank == 0:
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")

    # 3. 初始化 Dataset (使用各自的 mask 文件)
    if rank == 0: logger.info("Initializing Datasets...")
    
    train_ds = SINEDatasetE2E(
        sequences_with_ids=X_train,
        labels=y_train,
        mask_path=args.train_mask,  # <--- 使用 train_masks.pt
        tokenizer=tokenizer,
        max_token_length=args.max_length,
        is_training=True
    )
    
    val_ds = SINEDatasetE2E(
        sequences_with_ids=X_val,
        labels=y_val,
        mask_path=args.val_mask,
        tokenizer=tokenizer,
        max_token_length=args.max_length,
        is_training=False
    )

    # 4. 加载模型
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        freeze_backbone=True
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 5. Dataloaders
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 6. Loss & Optimizer
    if len(y_train) > 0:
        alpha = 0.25
        criterion = FocalLoss(gamma=2.0, alpha=alpha).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    best_val_f1 = 0.0
    optimizer = torch.optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr, 'weight_decay': 0.01},
        {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
        {'params': model.module.classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
        {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 
        'lr': args.head_lr, 'weight_decay': 0.1}
    ])

    scaler = torch.cuda.amp.GradScaler()

    # 7. 训练循环
    for epoch in range(1, args.epochs + 1):
        # 冻结/解冻逻辑
        if epoch == 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=True)
            optimizer = torch.optim.AdamW([
                {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr},
                {'params': model.module.classifier.parameters(), 'lr': args.head_lr},
                {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 'lr': args.head_lr},
            ], weight_decay=0.1)
            if rank == 0: logger.info("Phase 1: 冻结 Backbone")

        elif epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=False)
            optimizer = torch.optim.AdamW([
                {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr, 'weight_decay': 0.01},
                {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
                {'params': model.module.classifier.parameters(), 'lr': args.head_lr, 'weight_decay': 0.1},
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

            # # --- 2. 使用 autocast 开启混合精度 ---
            # # device_type 建议显式指定为 'cuda'
            # with torch.cuda.amp.autocast(device_type='cuda'):
            #     logits = model(input_ids, attention_mask, motif_mask)
            #     loss = criterion(logits, labels)

            # # --- 3. FP16 的反向传播与更新 ---
            # # 必须使用 scaler 来缩放 loss，防止梯度下溢
            # scaler.scale(loss).backward()
            
            # # --- 4. 重要：如果你想做梯度裁剪 (Clip Grad Norm) ---
            # # 在 FP16 下，必须先 unscale 梯度，然后再裁剪
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # # --- 5. 最后通过 scaler 更新参数 ---
            # scaler.step(optimizer)
            # scaler.update()

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

            # 每一轮都保存一个独立的 checkpoint
            epoch_save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auc': val_auc,
                'train_loss': avg_train_loss
            }, epoch_save_path)
            logger.info(f"  💾 Saved epoch checkpoint to {epoch_save_path}")

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