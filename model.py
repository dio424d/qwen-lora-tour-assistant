import warnings
import numpy as np
import re
import os

# 核心：禁用所有不必要的依赖和警告
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 禁止PEFT加载bitsandbytes
os.environ["PEFT_NO_BITSANDBYTES"] = "1"

# 只导入必要库（完全避开bitsandbytes）
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
import time

# ---------------------- 1. 数据集类（逻辑不变） ----------------------
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在：{file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_texts = [line.strip() for line in f]
        
        self.texts = []
        valid_char_pattern = re.compile(r'^[\u4e00-\u9fff_a-zA-Z0-9，。！？；：""''（）【】、·\s]+$')
        
        for text in raw_texts:
            if not text or text.isspace():
                continue
            if isinstance(text, float) and (np.isnan(text) or np.isinf(text)):
                continue
            
            text = re.sub(r'[^\u4e00-\u9fff_a-zA-Z0-9，。！？；：""''（）【】、·\s]', '', text)
            
            if len(text) < 8:
                continue
            if len(text) > max_len:
                text = text[:max_len]
            
            if len(text) > 0:
                unique_chars = len(set(text))
                if unique_chars / len(text) >= 0.15:
                    self.texts.append(text)
        
        print(f"✅ 加载并深度清洗数据集 {file_path}，有效数据 {len(self.texts)} 条")
        
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        self.encodings["input_ids"] = torch.nan_to_num(self.encodings["input_ids"], nan=0)
        
        self.labels = self.encodings["input_ids"].clone()
        padding_mask = self.encodings["attention_mask"].bool()
        self.labels[~padding_mask] = -100

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# ---------------------- 2. 模型加载（适配PEFT 0.6.2） ----------------------
model_path = "D:/aka/AIstudy/lora/Qwen1.5-1.8B-Chat"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件夹不存在：{model_path}")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
    cache_dir=None
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ========== 核心：纯GPU加载，PEFT 0.6.2专用 ==========
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",                # GPU设备
    torch_dtype=torch.float16,        # 半精度（3.6GB显存）
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager",      # 兼容RTX4050
    # 完全关闭所有量化相关参数
    load_in_4bit=False,
    load_in_8bit=False,
    use_cache=False
)

# 验证GPU加载
if next(model.parameters()).device.type != "cuda":
    raise RuntimeError("❌ 模型未加载到GPU！请检查PyTorch安装")
print(f"✅ 模型成功加载到GPU：{torch.cuda.get_device_name(0)}（显存占用约3.6GB）")

# ========== LoRA配置（PEFT 0.6.2原生版本，无bitsandbytes） ==========
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # Qwen1.5的关键模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# PEFT 0.6.2的get_peft_model不会调用bitsandbytes
model = get_peft_model(model, lora_config)
# 打印可训练参数（确认LoRA生效）
model.print_trainable_parameters()

# ---------------------- 3. 数据集加载（GPU版） ----------------------
train_dataset = TextDataset("train.txt", tokenizer, max_len=128)
# BatchSize=4（6GB显存稳跑）
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4,  
    shuffle=True, 
    pin_memory=True,
    num_workers=0  # Windows下稳定
)

# ---------------------- 4. 训练参数（GPU优化） ----------------------
EPOCHS = 1
LEARNING_RATE = 1e-4
MAX_STEPS = None
LOG_STEPS = 10
SAVE_STEPS = 100
GRADIENT_CLIP_NORM = 1.0
WARMUP_STEPS = 50
GRADIENT_ACCUMULATION_STEPS = 1

# 纯PyTorch优化器（无bitsandbytes依赖）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.001,
    eps=1e-7,
    betas=(0.9, 0.999)
)

# 调度器
total_training_steps = (len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_training_steps
)

# ---------------------- 5. 训练循环（纯GPU） ----------------------
model.train()
total_steps = 0
start_time = time.time()
print(f"\n🚀 开始GPU训练（{torch.cuda.get_device_name(0)} | 学习率：{LEARNING_RATE} | BatchSize：4）...")

# 清空GPU缓存（避免显存碎片）
torch.cuda.empty_cache()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_steps = 0
    accumulated_loss = 0.0
    
    for step, batch in enumerate(train_dataloader):
        if MAX_STEPS and total_steps >= MAX_STEPS:
            break
        
        # 数据移到GPU
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        
        # 数据检查
        if torch.isnan(batch["input_ids"]).any() or torch.isinf(batch["input_ids"]).any():
            print(f"⚠️ Step {total_steps} 输入数据异常，跳过")
            continue
        if (batch["labels"] == -100).all():
            print(f"⚠️ Step {total_steps} 无有效label，跳过")
            continue
        
        # 前向计算
        try:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss = torch.clamp(loss, 0, 15)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
        except Exception as e:
            print(f"⚠️ Step {total_steps} 前向计算出错：{e}，跳过")
            continue
        
        # 反向传播
        loss.backward()
        accumulated_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        # 更新参数
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += accumulated_loss
            epoch_steps += 1
            total_steps += 1
            accumulated_loss = 0.0
            
            # 打印日志
            if total_steps % LOG_STEPS == 0 and epoch_steps > 0:
                elapsed_time = time.time() - start_time
                avg_loss = epoch_loss / epoch_steps
                print(f"📌 Epoch {epoch+1}/{EPOCHS} | Step {total_steps} | Avg Loss: {avg_loss:.4f} | Time: {elapsed_time//60:.0f}分钟")
            
            # 保存权重
            if total_steps % SAVE_STEPS == 0:
                save_dir = f"./qwen-lora-step-{total_steps}"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                print(f"💾 保存权重到 {save_dir}")
    
    # Epoch总结
    if epoch_steps > 0:
        avg_epoch_loss = epoch_loss / epoch_steps
        elapsed_time = time.time() - start_time
        print(f"\n✅ Epoch {epoch+1} 完成 | Avg Loss: {avg_epoch_loss:.4f} | 耗时: {elapsed_time//60:.0f}分钟")
    else:
        print(f"\n⚠️ Epoch {epoch+1} 无有效数据")

# 保存最终权重
if total_steps > 0:
    final_dir = "./qwen-lora-final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 GPU训练完成！总步数：{total_steps} | 总耗时：{total_time:.0f}分钟")
    print(f"🏁 最终权重保存至 {final_dir}")
else:
    print("\n❌ 无有效训练步骤，请检查train.txt")

# 训练结束后清空GPU缓存
torch.cuda.empty_cache()