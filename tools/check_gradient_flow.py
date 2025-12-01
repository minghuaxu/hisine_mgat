#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_gradient_flow.py
======================
验证梯度能否正确传播到backbone

这个脚本会:
1. 创建一个简单的训练场景
2. 执行前向和反向传播
3. 检查backbone参数是否有梯度
4. 验证embedding在训练后是否变化
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from sine_classifier.model import MotifGuidedSINEClassifier


def check_gradient_flow(backbone_path: str):
    """检查梯度流"""
    
    print("="*80)
    print("检查梯度流")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 1. 加载模型
    print("\n步骤1: 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(backbone_path, trust_remote_code=True)
    
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=128,
        freeze_backbone=False  # 不冻结
    ).to(device)
    
    print(f"✅ 模型加载完成")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    if trainable_params < 1_000_000:
        print("\n⚠️  警告: 可训练参数过少，backbone可能被冻结了")
        return False
    
    # 2. 创建测试数据
    print("\n步骤2: 创建测试数据...")
    batch_size = 2
    seq_len = 50
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    motif_mask = torch.rand(batch_size, seq_len, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    
    print(f"✅ 测试数据创建完成")
    
    # 3. 保存初始embedding
    print("\n步骤3: 提取初始embedding...")
    model.eval()
    with torch.no_grad():
        outputs = model.backbone(input_ids, attention_mask, output_hidden_states=True)
        initial_embedding = outputs.hidden_states[-1].clone()
    
    print(f"  初始embedding形状: {initial_embedding.shape}")
    print(f"  初始embedding均值: {initial_embedding.mean().item():.6f}")
    
    # 4. 训练一步
    print("\n步骤4: 执行一次训练迭代...")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 前向传播
    logits = model(input_ids, attention_mask, motif_mask)
    loss = criterion(logits, labels)
    
    print(f"  Loss: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    print(f"✅ 反向传播完成")
    
    # 5. 检查backbone参数的梯度
    print("\n步骤5: 检查backbone梯度...")
    
    backbone_params_with_grad = []
    backbone_params_without_grad = []
    
    for name, param in model.backbone.named_parameters():
        if param.grad is not None:
            backbone_params_with_grad.append(name)
        else:
            backbone_params_without_grad.append(name)
    
    print(f"  有梯度的参数: {len(backbone_params_with_grad)}")
    print(f"  无梯度的参数: {len(backbone_params_without_grad)}")
    
    if len(backbone_params_with_grad) > 0:
        print(f"\n  ✅ Backbone有梯度（前5个）:")
        for name in backbone_params_with_grad[:5]:
            param = dict(model.backbone.named_parameters())[name]
            grad_norm = param.grad.norm().item()
            print(f"    {name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"\n  ❌ Backbone没有梯度!")
        return False
    
    # 6. 更新参数
    print("\n步骤6: 更新参数...")
    optimizer.step()
    print(f"✅ 参数更新完成")
    
    # 7. 检查embedding是否变化
    print("\n步骤7: 检查embedding变化...")
    model.eval()
    with torch.no_grad():
        outputs = model.backbone(input_ids, attention_mask, output_hidden_states=True)
        updated_embedding = outputs.hidden_states[-1]
    
    embedding_diff = torch.abs(initial_embedding - updated_embedding).mean().item()
    
    print(f"  更新后embedding均值: {updated_embedding.mean().item():.6f}")
    print(f"  Embedding变化量: {embedding_diff:.8f}")
    
    if embedding_diff > 1e-6:
        print(f"\n  ✅ Embedding随训练变化（变化量: {embedding_diff:.8f}）")
    else:
        print(f"\n  ❌ Embedding没有变化（变化量: {embedding_diff:.8f}）")
        return False
    
    # 8. 总结
    print("\n" + "="*80)
    print("梯度流检查结果")
    print("="*80)
    
    checks = [
        ("可训练参数 > 1M", trainable_params > 1_000_000),
        ("Backbone有梯度", len(backbone_params_with_grad) > 0),
        ("Embedding随训练变化", embedding_diff > 1e-6),
    ]
    
    all_passed = all(passed for _, passed in checks)
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    print("="*80)
    
    if all_passed:
        print("\n✅ 所有检查通过! 梯度流正常，这是真正的端到端训练。")
        return True
    else:
        print("\n❌ 部分检查失败! 可能不是端到端训练。")
        return False


def main():
    parser = argparse.ArgumentParser(description="检查梯度流")
    parser.add_argument("--backbone_path", required=True, help="Backbone路径")
    args = parser.parse_args()
    
    try:
        success = check_gradient_flow(args.backbone_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()