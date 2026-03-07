#!/usr/bin/env python3
"""
诊断微调模型问题
"""

import torch
from src.model import QwenLoRAModel
from src.config import settings

print("=" * 60)
print("微调模型诊断")
print("=" * 60)

print(f"\n配置信息:")
print(f"  基础模型路径: {settings.base_model_path}")
print(f"  LoRA 模型路径: {settings.lora_model_path}")
print(f"  设备: {settings.device}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA 设备: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 60)
print("加载微调模型...")
print("=" * 60)

try:
    model = QwenLoRAModel()
    model.load()
    print("✅ 模型加载成功")
    
    print("\n" + "=" * 60)
    print("测试生成...")
    print("=" * 60)
    
    test_question = "推荐一个适合夏天旅游的地方"
    print(f"问题: {test_question}")
    
    import time
    start_time = time.time()
    
    response = model.generate(
        [{"role": "user", "content": test_question}],
        max_new_tokens=256,
        temperature=0.7
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"回答: {response}")
    print(f"用时: {elapsed_time:.2f}秒")
    print(f"生成长度: {len(response)} 字符")
    
    if len(response) < 50:
        print("\n⚠️  警告: 回答内容过短，可能存在问题")
    else:
        print("\n✅ 生成测试成功")
        
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()