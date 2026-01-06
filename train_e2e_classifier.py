#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2e_classifier.py (OBIE 4-Class)
======================================
1. 计算 Core Start/End 索引并传给 Dataset
2. 模型使用 4 分类 (O, B, I, E)
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from sine_classifier.data import SINEDatasetE2E, collate_fn
from sine_classifier.model import MotifGuidedSINEClassifier, FocalLoss

logger = logging.getLogger(__name__)

def revcomp(s):
    if pd.isna(s) or s == "": return ""
    return str(Seq(s).reverse_complement())

def setup_logging(output_dir, rank):
    # (保持不变)
    handlers = [logging.StreamHandler()]
    if rank == 0:
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(output_dir) / "training.log"
            handlers.append(logging.FileHandler(log_file, mode='w'))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers, force=True)

def setup_ddp():
    # (保持不变)
    if "RANK" not in os.environ:
        os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = "0", "0", "1"
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
    dist.init_process_group(backend='nccl')
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

def cleanup_ddp(): dist.destroy_process_group()
def set_backbone_freeze(model, freeze: bool):
    raw = model.module if hasattr(model, "module") else model
    for p in raw.backbone.parameters(): p.requires_grad = not freeze

def process_csv_to_samples(csv_path, rank):
    """
    读取 CSV 并计算 OBIE 边界
    """
    if rank == 0: logger.info(f"Processing data from: {csv_path}")
    df = pd.read_csv(csv_path)
    samples = []
    
    for _, row in df.iterrows():
        chrom = row['chrom']
        s, e = row['start'], row['end']
        strand = row['strand']
        label = int(row['label'])
        uid = f"{chrom}:{s}-{e}({strand})"
        
        # 提取各部分序列 (处理空值)
        fl = row['flank_left'] if pd.notna(row['flank_left']) else ""
        core = row['seq'] if pd.notna(row['seq']) else ""
        fr = row['flank_right'] if pd.notna(row['flank_right']) else ""
        
        if strand == '-':
            # 负链: 物理上的右侧翼变成了 5'端 (Flank Left)
            # 拼接顺序: RC(Right) + RC(Core) + RC(Left)
            fl_rc = revcomp(fr)
            core_rc = revcomp(core)
            fr_rc = revcomp(fl)
            
            full_seq = fl_rc + core_rc + fr_rc
            core_start = len(fl_rc)
            core_end = len(fl_rc) + len(core_rc)
        else:
            # 正链: Flank Left + Core + Flank Right
            full_seq = fl + core + fr
            core_start = len(fl)
            core_end = len(fl) + len(core)
            
        samples.append({
            'uid': uid,
            'seq': full_seq,
            'label': label,
            'core_start': core_start,
            'core_end': core_end
        })
    return samples

def calculate_iou(pred_flat, label_flat):
    # 0=O, 1=B, 2=I, 3=E.  Foreground = {1, 2, 3}
    valid_mask = (label_flat != -100)
    pred = pred_flat[valid_mask]
    label = label_flat[valid_mask]
    if len(label) == 0: return 0.0
    
    pred_bin = (pred > 0).long()
    label_bin = (label > 0).long()
    
    intersection = (pred_bin & label_bin).sum().float()
    union = (pred_bin | label_bin).sum().float()
    if union == 0: return 1.0
    return (intersection / union).item()

@torch.no_grad()
def evaluate(model, loader, criterion_cls, criterion_seg, device, rank, world_size):
    model.eval()
    total_loss, total_iou, iou_count = 0.0, 0.0, 0
    all_g_probs, all_g_preds, all_g_labels = [], [], []

    if len(loader) == 0: return 0.0, 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="Eval", disable=(rank!=0)):
        input_ids = batch['input_ids'].to(device)
        att_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)
        token_labels = batch['token_labels'].to(device) 

        g_logits, t_logits = model(input_ids, att_mask, motif_mask)

        loss_cls = criterion_cls(g_logits, labels)
        loss_seg = criterion_seg(t_logits.view(-1, 4), token_labels.view(-1)) # 4 classes
        loss = loss_cls + 0.2 * loss_seg 
        total_loss += loss.item()

        probs = torch.softmax(g_logits, dim=1)
        preds = torch.argmax(g_logits, dim=1)
        all_g_probs.append(probs.cpu())
        all_g_preds.append(preds.cpu())
        all_g_labels.append(labels.cpu())

        t_preds = torch.argmax(t_logits, dim=-1)
        for i in range(len(input_ids)):
            total_iou += calculate_iou(t_preds[i].cpu(), token_labels[i].cpu())
            iou_count += 1

    # (Global Metrics Gathering Omitted for Brevity - kept logic same as before)
    # ... 简单起见，这里假设单卡或已处理好 ...
    # 为了完整性，请保留之前的 DDP gather 代码块
    
    # 这里只写单机逻辑示意，请保留原文件DDP部分
    if len(all_g_probs) > 0:
        all_g_probs = torch.cat(all_g_probs).numpy()
        all_g_preds = torch.cat(all_g_preds).numpy()
        all_g_labels = torch.cat(all_g_labels).numpy()
        
    acc = accuracy_score(all_g_labels, all_g_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_g_labels, all_g_preds, average='weighted', zero_division=0)
    try: auc = roc_auc_score(all_g_labels, all_g_probs[:, 1])
    except: auc = 0.0
    
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    
    return avg_loss, acc, f1, auc, avg_iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--train_mask", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_mask", required=True)
    parser.add_argument("--output_dir", required=True)
    
    # 仍需传入但实际上不怎么用的参数，为了兼容
    parser.add_argument("--train_motif_tsv", default="", help="Optional")
    parser.add_argument("--val_motif_tsv", default="", help="Optional")
    
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
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True)

    # 加载数据 (使用新函数)
    train_samples = process_csv_to_samples(args.train_csv, rank)
    val_samples = process_csv_to_samples(args.val_csv, rank)

    train_ds = SINEDatasetE2E(train_samples, args.train_mask, tokenizer, args.max_length, True)
    val_ds = SINEDatasetE2E(val_samples, args.val_mask, tokenizer, args.max_length, False)

    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)
    
    # 4分类: O, B, I, E
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_token_labels=4, # OBIE
        dropout=args.dropout,
        freeze_backbone=True
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    train_dl = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=4, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=4, collate_fn=collate_fn)

    if len(train_samples) > 0: criterion_cls = FocalLoss(gamma=2.0, alpha=0.25).to(device)
    else: criterion_cls = nn.CrossEntropyLoss().to(device)
    
    criterion_seg = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    best_val_f1 = 0.0
    optimizer = torch.optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': args.backbone_lr},
        {'params': model.module.motif_attention.parameters(), 'lr': args.head_lr},
        {'params': model.module.cls_adapter.parameters(), 'lr': args.head_lr},
        {'params': model.module.seg_adapter.parameters(), 'lr': args.head_lr},
        {'params': model.module.classifier.parameters(), 'lr': args.head_lr},
        {'params': model.module.token_classifier.parameters(), 'lr': args.head_lr},
        {'params': getattr(model.module, 'confidence_module', nn.ModuleList()).parameters(), 'lr': args.head_lr}
    ])

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_epochs > 0:
            set_backbone_freeze(model, freeze=True)
        elif epoch == args.freeze_epochs + 1:
            set_backbone_freeze(model, freeze=False)
            
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", disable=(rank!=0))
        
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            m_mask = batch['motif_mask'].to(device)
            labels = batch['label'].to(device)
            t_labels = batch['token_labels'].to(device)

            g_out, t_out = model(ids, mask, m_mask)
            
            l_cls = criterion_cls(g_out, labels)
            l_seg = criterion_seg(t_out.view(-1, 4), t_labels.view(-1))
            loss = l_cls + 0.2 * l_seg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if rank==0: pbar.set_postfix({'loss': f"{loss.item():.4f}", 'seg': f"{l_seg.item():.4f}"})

        # Evaluate
        # (Please verify imports for evaluate function DDP if copy-pasting partially)
        v_loss, v_acc, v_f1, v_auc, v_iou = evaluate(model, val_dl, criterion_cls, criterion_seg, device, rank, world_size)
        
        if rank == 0:
            logger.info(f"Epoch {epoch} | Val F1: {v_f1:.4f} | mIoU: {v_iou:.4f}")
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(model.module.state_dict(), Path(args.output_dir)/"best_model.pt")
                
    cleanup_ddp()

if __name__ == "__main__":
    main()