import warnings
import os
import torch
import jieba
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ====================== 核心配置（完全对齐你的环境） ======================
BASE_MODEL_PATH = "D:/aka/AIstudy/lora/Qwen1.5-1.8B-Chat"
LORA_MODEL_PATH = "./qwen-lora-final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 生成配置（和训练时完全一致）
GEN_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "eos_token_id": None,
    "pad_token_id": None,
}

# ====================== 初始化配置（避坑） ======================
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PEFT_NO_BITSANDBYTES"] = "1"
torch.cuda.empty_cache()

# ====================== 辅助函数：判断是否为中文样本（核心筛选逻辑） ======================
def is_chinese_sample(input_text):
    """判断输入是否为中文样本（过滤英文输入）"""
    # 统计中文字符占比（超过30%视为中文样本）
    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', input_text))
    total_char_count = len(input_text.strip())
    if total_char_count == 0:
        return False
    chinese_ratio = chinese_char_count / total_char_count
    return chinese_ratio > 0.3

# ====================== 加载模型（不变） ======================
def load_model_and_tokenizer():
    print(f"📌 开始加载模型（设备：{DEVICE}）...")
    
    # 1. 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map=DEVICE,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_cache=True
    )
    
    # 3. 初始化LoRA（严格对齐训练参数）
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. 挂载LoRA权重
    model = get_peft_model(base_model, lora_config)
    model.load_adapter(LORA_MODEL_PATH, adapter_name="default")
    model.eval()
    
    print(f"✅ 模型加载完成！（已挂载LoRA权重：{LORA_MODEL_PATH}）")
    return tokenizer, model

# ====================== 核心功能：生成回答 + 量化评估（不变） ======================
def travel_qa(tokenizer, model, question):
    prompt = f"""### 指令：作为旅游行业客服，简洁、友好回复用户的旅游咨询，贴合旅游场景话术，保持真实回复风格。
### 输入：{question}
### 输出："""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_CONFIG)
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("### 输出：")[-1].strip()
    return answer if answer else full_text.strip()

def keyword_match_score(pred_answer, ref_answer):
    # 分词 + 过滤无效字符
    pred_words = set(jieba.lcut(pred_answer.strip()))
    ref_words = set(jieba.lcut(ref_answer.strip()))
    # 过滤标点、空格、单字
    filter_chars = ['，', '。', '：', '、', ' ', '\n', '！', '？', '；', '“', '”', '（', '）']
    pred_words = {w for w in pred_words if w not in filter_chars and len(w) > 1}
    ref_words = {w for w in ref_words if w not in filter_chars and len(w) > 1}
    
    if len(ref_words) == 0:
        return 0.0
    common_words = pred_words & ref_words
    score = len(common_words) / len(ref_words)
    return round(score, 4)

def semantic_similarity_score(tokenizer, model, pred_answer, ref_answer):
    def get_embedding(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.base_model.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        return embedding
    
    pred_emb = get_embedding(pred_answer)
    ref_emb = get_embedding(ref_answer)
    similarity = cosine_similarity(pred_emb, ref_emb)[0][0]
    return round(similarity, 4)

# ====================== 核心修改：解析test.txt并筛选中文样本 ======================
def parse_test_txt(file_path="test.txt"):
    """解析test.txt，只保留中文输入样本（过滤英文）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 未找到测试文件：{file_path}")
    
    test_dataset = []  # 存储所有样本（问题，标准答案）
    chinese_dataset = []  # 只存储中文样本
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # 按 "### 指令：" 分割样本（每个样本以指令开头）
        samples = content.split("### 指令：")
    
    # 解析每个样本
    for sample in samples[1:]:  # 跳过第一个空元素
        if "### 输入：" in sample and "### 输出：" in sample:
            # 提取问题（输入）和标准答案（输出）
            input_part = sample.split("### 输入：")[1].split("### 输出：")[0].strip()
            output_part = sample.split("### 输出：")[1].strip()
            if input_part and output_part:
                test_dataset.append((input_part, output_part))
                # 筛选中文样本（核心：判断输入是否为中文）
                if is_chinese_sample(input_part):
                    chinese_dataset.append((input_part, output_part))
    
    print(f"📊 解析结果：")
    print(f"   - test.txt 总样本数：{len(test_dataset)}")
    print(f"   - 中文样本数（待测试）：{len(chinese_dataset)}")
    print(f"   - 英文样本数（已过滤）：{len(test_dataset) - len(chinese_dataset)}")
    
    if not chinese_dataset:
        raise ValueError("❌ 未解析到有效中文样本，请检查test.txt格式！")
    return chinese_dataset

# ====================== 批量评估中文样本（全量跑） ======================
def evaluate_all_chinese_samples(tokenizer, model):
    """批量评估所有中文样本"""
    # 1. 解析并筛选中文样本
    chinese_dataset = parse_test_txt("test.txt")
    total_samples = len(chinese_dataset)
    
    print("\n" + "="*120)
    print(f"📋 中文样本批量评估报告（共 {total_samples} 个中文案例）")
    print("="*120)
    
    # 2. 初始化评估指标
    total_keyword_score = 0.0
    total_semantic_score = 0.0
    failed_samples = []  # 记录生成失败的样本（可选）
    
    # 3. 逐一样本评估（批量跑所有中文案例）
    for idx, (question, ref_answer) in enumerate(chinese_dataset, 1):
        try:
            # 生成模型回答
            pred_answer = travel_qa(tokenizer, model, question)
            
            # 计算评估指标
            kw_score = keyword_match_score(pred_answer, ref_answer)
            sem_score = semantic_similarity_score(tokenizer, model, pred_answer, ref_answer)
            
            # 累加总分
            total_keyword_score += kw_score
            total_semantic_score += sem_score
            
            # 打印单个样本结果（精简输出，避免刷屏）
            print(f"\n【中文样本 {idx}/{total_samples}】")
            print(f"🎯 用户问题：{question[:50]}..." if len(question) > 50 else f"🎯 用户问题：{question}")
            print(f"📖 标准答案：{ref_answer[:80]}..." if len(ref_answer) > 80 else f"📖 标准答案：{ref_answer}")
            print(f"🤖 模型回答：{pred_answer[:80]}..." if len(pred_answer) > 80 else f"🤖 模型回答：{pred_answer}")
            print(f"📈 关键词匹配度：{kw_score*100:6.2f}%  |  语义相似度：{sem_score*100:6.2f}%")
            print("-"*120)
        except Exception as e:
            print(f"\n【中文样本 {idx}/{total_samples}】- 评估失败！")
            print(f"❌ 问题：{question[:50]}...")
            print(f"❌ 错误信息：{str(e)[:100]}")
            print("-"*120)
            failed_samples.append(idx)
    
    # 4. 计算整体平均指标
    avg_keyword_score = round(total_keyword_score / total_samples, 4) if total_samples > 0 else 0.0
    avg_semantic_score = round(total_semantic_score / total_samples, 4) if total_samples > 0 else 0.0
    
    # 5. 输出整体报告
    print("\n" + "="*120)
    print(f"📊 中文样本整体评估结果")
    print("="*120)
    print(f"✅ 测试中文样本总数：{total_samples}")
    print(f"✅ 评估失败样本数：{len(failed_samples)}")
    print(f"✅ 平均关键词匹配度：{avg_keyword_score*100:6.2f}%")
    print(f"✅ 平均语义相似度：{avg_semantic_score*100:6.2f}%")
    print("="*120)
    
    return {
        "中文样本数": total_samples,
        "失败样本数": len(failed_samples),
        "平均关键词匹配度": avg_keyword_score,
        "平均语义相似度": avg_semantic_score
    }

# ====================== 交互式问答（保留） ======================
def interactive_qa(tokenizer, model):
    print("\n" + "="*80)
    print("💬 进入交互式问答模式（输入「退出」结束）")
    print("="*80)
    while True:
        user_input = input("\n请输入你的旅游咨询问题：")
        if user_input.strip().lower() in ["退出", "exit", "quit"]:
            print("👋 问答结束，感谢使用！")
            break
        if not user_input.strip():
            print("⚠️ 请输入有效问题！")
            continue
        answer = travel_qa(tokenizer, model, user_input)
        print(f"🎯 模型回答：{answer}")

# ====================== 主函数（批量评估中文样本 + 交互） ======================
if __name__ == "__main__":
    # 1. 加载模型（只加载一次）
    tokenizer, model = load_model_and_tokenizer()
    
    # 2. 批量评估所有中文样本（核心功能）
    evaluate_all_chinese_samples(tokenizer, model)
    
    # 3. 保留交互式测试（可选，不想用可以注释掉）
    interactive_qa(tokenizer, model)
    
    # 4. 清理GPU显存
    if DEVICE == "cuda":
        torch.cuda.empty_cache()