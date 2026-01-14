#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2e_classifier.py
=======================
Motif-Guided SINE Classifier 的端到端训练脚本。

核心技术特性 (Key Technologies):
1. LoRA (Low-Rank Adaptation): 
   - 仅微调 Backbone 的少量参数 (Rank=8)，大幅降低显存占用，同时保留预训练模型的泛化能力。
   
2. Auxiliary Task Strategy (辅助任务策略):
   - 主任务：序列分类 (Binary Classification)。
   - 辅助任务：序列分割 (CRF Segmentation)。
   - 策略：CRF Loss 仅作为辅助正则项 (Weight=0.1)，用于引导模型关注正确的生物学边界，但不主导梯度方向。

3. Dynamic Loss Weighting (动态损失权重):
   - CRF Loss 引入 Warmup 机制，在训练初期权重为 0，防止初始阶段不准确的边界预测干扰分类头收敛。

4. Focal Loss:
   - 解决正负样本不平衡问题 (Pos:Neg ≈ 1:10+)，专注于难分样本的挖掘。
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tf_logging
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from Bio.Seq import Seq

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from data import SINEDatasetE2E, collate_fn
from model import MotifGuidedSINEClassifier, FocalLoss

# 设置 Transformers 日志级别
tf_logging.set_verbosity_error()
logger = logging.getLogger(__name__)

# ==========================================
# 基础工具函数
# ==========================================

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # 确保每个子进程的种子既是确定的，又在不同进程间不同
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = "0", "0", "1"
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
    dist.init_process_group(backend='nccl')
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

def cleanup_ddp(): 
    dist.destroy_process_group()

def set_backbone_freeze(model, freeze: bool):
    """控制 Backbone 是否参与梯度计算"""
    raw = model.module if hasattr(model, "module") else model
    for p in raw.backbone.parameters(): 
        # 如果使用了 LoRA，这里主要控制非 LoRA 部分
        # 通常 LoRA 开启后，base model 自动 freeze，这里是双重保险
        p.requires_grad = not freeze

def process_csv_to_samples(csv_path, rank):
    """读取 CSV 并计算 OBIE 边界，构建样本列表"""
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
    """计算分割任务的 Intersection over Union"""
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

def uniform_subsample(samples, target_size):
    """
    分层等间隔采样(Debug 模式专用)：保证正负样本比例不变，且覆盖整个数据集范围
    """
    if len(samples) <= target_size:
        return samples

    # 1. 按标签分组
    pos_samples = [s for s in samples if s['label'] == 1]
    neg_samples = [s for s in samples if s['label'] == 0]
    
    # 2. 计算比例
    pos_ratio = len(pos_samples) / len(samples)
    num_pos = max(1, int(target_size * pos_ratio)) # 至少取1个正样本
    num_neg = target_size - num_pos
    
    def get_indices(data_len, count):
        # 计算步长，实现等间隔采样
        if count <= 0: return []
        if count == 1: return [data_len // 2]
        return np.linspace(0, data_len - 1, count, dtype=int).tolist()

    # 3. 执行采样
    selected_pos = [pos_samples[i] for i in get_indices(len(pos_samples), num_pos)]
    selected_neg = [neg_samples[i] for i in get_indices(len(neg_samples), num_neg)]

    return selected_pos + selected_neg

# ==========================================
# 评估循环
# ==========================================

@torch.no_grad()
def evaluate(model, loader, criterion_cls, device, rank, world_size):
    model.eval()
    total_loss, total_iou, iou_count = 0.0, 0.0, 0
    all_g_probs, all_g_preds, all_g_labels = [], [], []

    # DDP处理：获取原始模型以调用 decode 方法
    raw_model = model.module if hasattr(model, "module") else model

    # 检查数据量 确保所有进程都知道 loader 是否为空
    local_count = torch.tensor([len(loader)], device=device)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    if local_count.item() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="Evaluate", disable=(rank != 0)):
        input_ids = batch['input_ids'].to(device)
        att_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)
        token_labels = batch['token_labels'].to(device) 

        # 传入 labels 以计算 CRF Loss (虽然 eval 时 loss 仅作参考)
        g_logits, emissions, crf_loss = model(input_ids, att_mask, motif_mask, token_labels, labels=labels)

        # Loss 计算
        loss_cls = criterion_cls(g_logits, labels)
        loss = loss_cls + 0.1 * crf_loss

        total_loss += loss.item()

        # Global Metrics 收集
        probs = torch.softmax(g_logits, dim=1)
        preds = torch.argmax(g_logits, dim=1)
        all_g_probs.append(probs) # 保持在 GPU 上以便 gather，或者 cpu() 后再处理
        all_g_preds.append(preds)
        all_g_labels.append(labels)

        # Viterbi 解码计算 IoU
        decoded_tags_list = raw_model.decode(emissions, att_mask)
        for i, pred_tags in enumerate(decoded_tags_list):
            # pred_tags: List[int], 长度为该样本的真实有效长度
            # 转换预测为 tensor
            p_tensor = torch.tensor(pred_tags, device=device)
            # 获取对应的真实标签 (需截取有效长度，忽略原本的 padding -100)
            valid_len = len(pred_tags)
            l_tensor = token_labels[i, :valid_len]
            
            # 计算 IoU
            iou = calculate_iou(p_tensor, l_tensor)
            total_iou += iou
            iou_count += 1

    # --- DDP 数据聚合 ---
    if len(all_g_probs) > 0:
        all_g_probs = torch.cat(all_g_probs)
        all_g_preds = torch.cat(all_g_preds)
        all_g_labels = torch.cat(all_g_labels)
    else:
        # 处理空数据情况
        all_g_probs = torch.empty((0, 2), device=device)
        all_g_preds = torch.empty((0,), dtype=torch.long, device=device)
        all_g_labels = torch.empty((0,), dtype=torch.long, device=device)

    if world_size > 1:
        # 聚合标量指标 (Loss, IoU)
        metrics_tensor = torch.tensor([total_loss, total_iou, iou_count, len(loader)], device=device, dtype=torch.float64)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = metrics_tensor[0].item() / metrics_tensor[3].item()
        avg_iou = metrics_tensor[1].item() / metrics_tensor[2].item() if metrics_tensor[2].item() > 0 else 0.0

        # 聚合向量指标 (Probs, Preds, Labels)
        # 获取各卡数据大小
        local_size = torch.tensor([all_g_labels.size(0)], dtype=torch.long, device=device)
        size_list = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = [s.item() for s in size_list]
        max_size = max(sizes)
        
        # 动态获取类别数 (SINE 任务通常是 2 分类)
        num_classes_prob = all_g_probs.size(1) if all_g_probs.size(0) > 0 else 2
        
        # 准备 Buffer (带 Padding)
        prob_buffer = torch.zeros((max_size, num_classes_prob), dtype=all_g_probs.dtype, device=device)
        prob_buffer[:local_size] = all_g_probs
        gathered_probs = [torch.zeros_like(prob_buffer) for _ in range(world_size)]
        
        pred_buffer = torch.zeros((max_size,), dtype=all_g_preds.dtype, device=device)
        pred_buffer[:local_size] = all_g_preds
        gathered_preds = [torch.zeros_like(pred_buffer) for _ in range(world_size)]
        
        label_buffer = torch.zeros((max_size,), dtype=all_g_labels.dtype, device=device)
        label_buffer[:local_size] = all_g_labels
        gathered_labels = [torch.zeros_like(label_buffer) for _ in range(world_size)]

        # 执行 Gather
        dist.all_gather(gathered_probs, prob_buffer)
        dist.all_gather(gathered_preds, pred_buffer)
        dist.all_gather(gathered_labels, label_buffer)

        if rank == 0:
            final_probs = []
            final_preds = []
            final_labels = []
            # 去除 Padding
            for i, size in enumerate(sizes):
                final_probs.append(gathered_probs[i][:size].cpu())
                final_preds.append(gathered_preds[i][:size].cpu())
                final_labels.append(gathered_labels[i][:size].cpu())
            
            all_g_probs = torch.cat(final_probs).numpy()
            all_g_preds = torch.cat(final_preds).numpy()
            all_g_labels = torch.cat(final_labels).numpy()
        else:
            # 非 Rank 0 返回占位符
            return 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        # 单机模式
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
        all_g_probs = all_g_probs.cpu().numpy()
        all_g_preds = all_g_preds.cpu().numpy()
        all_g_labels = all_g_labels.cpu().numpy()

    # 3. 计算最终指标 (仅 Rank 0)
    acc = accuracy_score(all_g_labels, all_g_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_g_labels, all_g_preds, average='weighted', zero_division=0
    )
    
    try:
        # 二分类取 index 1, 多分类通常需要 label_binarize 或 macro/weighted 处理
        if all_g_probs.shape[1] == 2:
            auc = roc_auc_score(all_g_labels, all_g_probs[:, 1])
        else:
            # 如果是多分类，这里简单置0或计算 Macro AUC
            auc = 0.0 
    except Exception:
        auc = 0.0

    return avg_loss, acc, f1, auc, avg_iou

# ==========================================
# 主训练流程
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--train_mask", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_mask", required=True)
    parser.add_argument("--output_dir", required=True)
    
    # 兼容性参数
    parser.add_argument("--train_motif_tsv", default="", help="Optional")
    parser.add_argument("--val_motif_tsv", default="", help="Optional")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=5e-6)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="使用 1/1000 的数据快速跑通流程")

    # [技术 1] LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_lora", action="store_true", default=True, help="默认开启 LoRA")

    args = parser.parse_args()
    rank, local_rank, world_size = setup_ddp()
    set_seed(0) 

    device = torch.device(f'cuda:{local_rank}')
    setup_logging(args.output_dir, rank)
    torch.cuda.set_device(local_rank)

    # 1. 数据准备
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True, use_fast=True)
    train_samples = process_csv_to_samples(args.train_csv, rank)
    val_samples = process_csv_to_samples(args.val_csv, rank)

    if rank == 0: logger.info(f"Max Seq Length: {args.max_length}")

    if args.debug:
        target_train = max(args.batch_size * world_size * 2, int(len(train_samples) * ratio))
        target_val = max(args.batch_size * world_size * 1, int(len(val_samples) * ratio))
        if rank == 0: logger.info(f"⚠️ DEBUG MODE: Subsampling to ~{target_train} samples")
        train_samples = uniform_subsample(train_samples, target_train)
        val_samples = uniform_subsample(val_samples, target_val)

    train_ds = SINEDatasetE2E(train_samples, args.train_mask, tokenizer, args.max_length, True)
    val_ds = SINEDatasetE2E(val_samples, args.val_mask, tokenizer, args.max_length, False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    # 保证 DataLoader 的随机性在多卡间一致
    g = torch.Generator()
    g.manual_seed(0)
    train_dl = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=4, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
    val_dl = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=4, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
    
    # 2. 模型构建与 LoRA 应用
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)

    if args.use_lora:
        if rank == 0: logger.info(f"🚀 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, # 或者 None，取决于你的 Backbone 类型
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"] 
        )
        backbone = get_peft_model(backbone, peft_config)
        if rank == 0:
            backbone.print_trainable_parameters()

    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_token_labels=4, # OBIE
        dropout=args.dropout,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    
    # 3. 优化器配置
    # 策略：区分 CRF 与主体网络的学习率
    crf_params = list(map(id, model.module.crf.parameters()))
    base_params = filter(lambda p: id(p) not in crf_params and p.requires_grad, model.parameters()) 

    crf_lr = 1e-3
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.head_lr, 'weight_decay': 0.05},          # LoRA + Head
        {'params': model.module.crf.parameters(), 'lr': crf_lr, 'weight_decay': 0.0} # CRF (需要较大 LR)
    ])

    #  添加 Scheduler (Cosine Decay 是目前最常用的)
    # 计算总步数
    num_training_steps = len(train_dl) * args.epochs
    num_warmup_steps = int(0.2 * num_training_steps) # 10% Warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    if len(train_samples) > 0: 
        criterion_cls = FocalLoss(gamma=2.0, alpha=0.75).to(device)
    else: 
        criterion_cls = nn.CrossEntropyLoss().to(device)

    
    best_val_f1 = 0.0
    start_epoch = 1

    # 4. 恢复训练逻辑
    if args.resume:
        if rank == 0: logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        # 优化器状态选择性恢复 (这里选择不恢复，以免旧的 Loss 权重干扰)
        if rank == 0: logger.info(f"  ✅ Weights loaded. Resuming from Epoch {start_epoch}")
        if start_epoch > args.freeze_epochs + 1:
            set_backbone_freeze(model, freeze=False)
        else:
            set_backbone_freeze(model, freeze=True)

    # [技术 3] Loss 动态权重配置
    # 目标：CRF 权重从 0 缓慢升至 0.05 (辅助地位)
    TARGET_CRF_WEIGHT = 0.05
    CRF_WARMUP_EPOCHS = 10  # 前10个epoch逐渐增加权重

    # =======================================================
    # 打印训练参数 & 初始化日志工具 (WandB)
    # =======================================================
    if rank == 0: 
        logger.info("="*60)
        logger.info("🚀 TRAINING CONFIGURATION SUMMARY")
        logger.info("="*60)
        logger.info(f"  > Backbone      : {args.backbone_path}")
        logger.info(f"  > LoRA Config   : r={args.lora_r}, alpha={args.lora_alpha}")
        logger.info(f"  > Batch Size    : {args.batch_size} (Per GPU)")
        logger.info(f"  > Learning Rate : Head={args.head_lr}, CRF={crf_lr}")
        logger.info("-" * 60)
        logger.info(f"  > [Dynamic Weight Strategy]")
        logger.info(f"  > Target CRF Weight : {TARGET_CRF_WEIGHT}")
        logger.info(f"  > Warmup Epochs     : {CRF_WARMUP_EPOCHS}")
        logger.info("="*60)

        # 初始化 WandB (主流可视化方案)
        try:
            import wandb
            os.environ["WANDB_MODE"] = "offline"
            # 你可以在这里修改 project 名
            wandb.init(
                project="SINE-Classifier-E2E", 
                name=f"LoRA-r{args.lora_r}-wCRF{TARGET_CRF_WEIGHT}",
                config=vars(args)
            )
            # 补充记录硬编码的参数
            wandb.config.update({
                "target_crf_weight": TARGET_CRF_WEIGHT,
                "crf_warmup_epochs": CRF_WARMUP_EPOCHS
            })
            logger.info("✅ WandB initialized successfully.")
        except ImportError:
            logger.warning("⚠️ WandB not installed. Please run 'pip install wandb' for best visualization.")
            wandb = None

    # 定义全局步数，用于画连续的曲线
    global_step = 0


    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        # 计算 CRF 权重
        if epoch <= CRF_WARMUP_EPOCHS:
            current_crf_weight = (epoch / CRF_WARMUP_EPOCHS) * TARGET_CRF_WEIGHT
        else:
            current_crf_weight = TARGET_CRF_WEIGHT
            
        if rank == 0:
            logger.info(f"Epoch {epoch} | CRF Loss Weight: {current_crf_weight:.4f}")

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

            # 显式传入 labels=labels，触发 model.py 内部的 CRF 计算
            # g_out: Global Logits
            # l_crf: CRF Loss (仅在 Positive Samples 上非零)
            g_out, emissions, l_crf = model(ids, mask, m_mask, t_labels, labels=labels)

            l_cls = criterion_cls(g_out, labels)

            # [关键策略] 辅助任务加权组合
            loss = l_cls + current_crf_weight * l_crf
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # 更新全局步数
            global_step += 1

            if rank==0: 
                # 使用科学计数法，并处理 value 极小的情况
                # 记录一下当前的 crf loss 值，如果是 0 说明是负样本 batch
                crf_val = l_crf.item()
                cls_val = l_cls.item()
                loss_val = loss.item()

                # 只有当 crf_val > 0 时才认为是有意义的 CRF 训练
                pbar.set_postfix({
                    'ls': f"{loss_val:.2e}", 
                    'cls': f"{cls_val:.2e}",
                    'crf': f"{crf_val:.2e}"
                })

                # 每 100 个 Batch 打印一次详细 Log 到控制台
                # 这样 log 文件里会有记录，方便事后 grep
                if global_step % 100 == 0:
                    logger.info(
                        f"[Step {global_step}] "
                        f"Loss: {loss_val:.4f} | "
                        f"CLS: {cls_val:.4f} | "
                        f"CRF: {crf_val:.4f} | "
                        f"CRF_W: {current_crf_weight:.4f}"
                    )

                # 实时上报 WandB (记录每一个 step，画出来最平滑)
                if wandb:
                    wandb.log({
                        "train/total_loss": loss_val,
                        "train/cls_loss": cls_val,
                        "train/crf_loss": crf_val,
                        "train/crf_weight_step": current_crf_weight,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "epoch": epoch
                    }, step=global_step)
        
        # --- Validation ---
        v_loss, v_acc, v_f1, v_auc, v_iou = evaluate(
            model, val_dl, criterion_cls, device, rank, world_size
        )

        if rank == 0:
            logger.info(f"Epoch {epoch} | Val F1: {v_f1:.4f} | mIoU: {v_iou:.4f}")

            # WandB 记录验证集指标
            if wandb:
                wandb.log({
                    "val/f1": v_f1,
                    "val/iou": v_iou,
                    "val/acc": v_acc,
                    "val/loss": v_loss
                }, step=global_step)

            # 保存 Checkpoint
            state = {
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': v_f1,
                'val_iou': v_iou,
            }

            # 1. 保存每一轮的独立 Checkpoint
            logger.info(f" 正在写入模型")
            torch.save(state, Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt")
            logger.info(f"  💾 Saved epoch checkpoint to checkpoint_epoch_{epoch}.pt")

            # Best model
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(model.module.state_dict(), Path(args.output_dir)/"best_model.pt")
                logger.info(f"  ✅ New Best F1! Saved best_model.pt")

            # Latest (用于断点续训)
            torch.save(state, Path(args.output_dir) / "latest.pt")
    
    if rank == 0 and wandb:
        wandb.finish() # 结束 logging

    cleanup_ddp()

if __name__ == "__main__":
    main()