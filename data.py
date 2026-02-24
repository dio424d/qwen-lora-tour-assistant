import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# ---------------------- 1. 适配旅游数据集（TouInd）核心配置 ----------------------
# 数据集文件夹名称（改为你克隆的旅游数据集文件夹名）
dataset_dir = "TouInd"  
# 旅游数据集核心文件（JSONL格式，替换原有的train.csv）
dataset_file = os.path.join(dataset_dir, "sft_dataset_train_processed_rewritev2_with_example.jsonl")  

# ---------------------- 2. 数据集检查（适配手动克隆的旅游数据集）----------------------
print("="*60)
print("✅ 适配旅游数据集（TouInd），开始检查本地文件...")
print("="*60)

# 检查文件夹和核心文件是否存在
if not os.path.exists(dataset_dir) or not os.path.exists(dataset_file):
    raise FileNotFoundError(
        f"\n❌ 未找到旅游数据集文件！请确认：\n"
        f"1. 已通过git clone下载TouInd文件夹，路径：{os.getcwd()}\\{dataset_dir}\n"
        f"2. 文件夹内有核心文件：sft_dataset_train_processed_rewritev2_with_qa.jsonl\n"
        f"3. 若文件名不同，请修改代码中「dataset_file」变量为实际文件名\n"
        f"   当前代码路径：{os.getcwd()}\n"
        f"   需满足路径：{dataset_file}"
    )
else:
    print("✅ 本地旅游数据集已找到，开始读取处理...")

# ---------------------- 3. 读取旅游数据集（JSONL格式）----------------------
# 读取JSONL格式的旅游数据集
train_data = []
with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            train_data.append(json.loads(line))

# 转换为DataFrame，适配后续处理
df = pd.DataFrame(train_data)
# 确保数据有「问题/回复」核心列（适配旅游数据集字段）
# 若数据集字段不同，可根据实际字段名修改（如"query"/"response"等）
if "question" not in df.columns or "answer" not in df.columns:
    # 兼容旅游数据集常见字段名
    if "query" in df.columns and "response" in df.columns:
        df.rename(columns={"query": "question", "response": "answer"}, inplace=True)
    elif "input" in df.columns and "output" in df.columns:
        df.rename(columns={"input": "question", "output": "answer"}, inplace=True)
    else:
        raise KeyError("❌ 数据集缺少核心字段！请确认字段名（如question/answer、query/response等）")

print(f"✅ 旅游数据集读取完成，初始数据量：{len(df)}条")

# ---------------------- 4. 数据清洗（适配旅游场景，轻量化处理）----------------------
df = df.drop_duplicates(subset=["question"])  # 去重
df = df.dropna(subset=["question", "answer"])  # 去空值
# 过滤过短无效数据（旅游咨询通常更长，调整阈值）
df = df[(df["question"].str.len() >= 8) & (df["answer"].str.len() >= 15)]
# 限制数据量（适配6GB显存，旅游数据更复杂，调整为1000条）
if len(df) > 1000:
    df = df.sample(n=1000, random_state=42)
print(f"✅ 数据清洗完成，剩余有效旅游数据量：{len(df)}条（适配6GB显存）")

# ---------------------- 5. 兜底补充（旅游场景专属备用数据）----------------------
if len(df) < 800:
    supplement_num = 800 - len(df)
    # 旅游场景真实客服话术（贴合你的数据集风格）
    backup_questions = [
        "去云南旅游最佳时间是什么时候？", "景区门票可以提前几天预订？",
        "自由行和跟团游哪个更适合去西藏？", "酒店取消预订需要扣费吗？",
        "旅游意外险包含哪些保障范围？", "高铁票改签后还能退票吗？",
        "境外旅游需要提前办理签证吗？", "景区内有代步车可以租赁吗？",
        "儿童门票的身高限制是多少？", "旅游团的用餐标准是什么？"
    ]
    backup_answers = [
        "您好，云南最佳旅游时间是3-5月和9-11月，气候宜人，避开雨季和旺季人流。",
        "您好，大部分景区门票可提前1-7天预订，热门景区建议提前3天以上预订。",
        "您好，西藏旅游若首次前往，跟团游更省心（含交通/住宿/高反保障）；有经验可选择自由行。",
        "您好，酒店取消预订需看预订规则，提前24小时取消通常免费，临时取消可能扣除首晚房费。",
        "您好，旅游意外险包含意外身故/伤残、医疗费用、紧急救援，部分含行程延误/行李丢失保障。",
        "您好，高铁票改签后仍可退票，退票手续费按改签后的退票时间计算，以12306规则为准。",
        "您好，多数境外国家需要提前办理签证，部分国家支持落地签/免签，需提前确认目的地政策。",
        "您好，大部分景区内提供代步车租赁服务，费用按小时/次收取，可现场咨询景区工作人员。",
        "您好，儿童门票通常以1.2米为界，1.2米以下免票，1.2-1.5米半价，具体以景区公示为准。",
        "您好，旅游团用餐标准通常为八菜一汤，十人一桌，含当地特色菜品，特殊饮食可提前告知导游。"
    ]
    # 补充数据
    for j in range(supplement_num):
        q = backup_questions[j % len(backup_questions)]
        a = backup_answers[j % len(backup_answers)]
        df = pd.concat([df, pd.DataFrame({"question": [q], "answer": [a]})], ignore_index=True)
    print(f"✅ 数据量不足，自动补充{supplement_num}条旅游客服数据，最终数据量：{len(df)}条")

# ---------------------- 6. 格式转换（适配Qwen-1.8B-int4，旅游场景专属指令）----------------------
def format_data(row):
    # 旅游客服专属指令，贴合你的数据集场景
    instruction = "作为旅游行业客服，简洁、友好回复用户的旅游咨询，贴合旅游场景话术，保持真实回复风格。"
    input_text = row["question"]
    output_text = row["answer"]
    # Qwen专属格式，轻量化，适配6GB显存
    return f"### 指令：{instruction}\n### 输入：{input_text}\n### 输出：{output_text}"

df["formatted_data"] = df.apply(format_data, axis=1)

# ---------------------- 7. 划分数据集（8:1:1，适配旅游数据）----------------------
train_data, temp_data = train_test_split(df["formatted_data"], test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存为txt，供后续训练使用
train_data.to_csv("train.txt", index=False, header=False, encoding="utf-8")
val_data.to_csv("val.txt", index=False, header=False, encoding="utf-8")
test_data.to_csv("test.txt", index=False, header=False, encoding="utf-8")

# ---------------------- 8. 结果输出（适配旅游数据集）----------------------
print("\n🎉 旅游数据集处理完成！")
print(f"📊 训练集：{len(train_data)}条，验证集：{len(val_data)}条，测试集：{len(test_data)}条")
print(f"💡 显存占用提示：当前数据处理完成，未占用GPU显存，后续训练将控制在5GB以内")
print(f"📁 生成文件：train.txt / val.txt / test.txt（已保存到当前目录）")