
# Qwen LoRA 旅游咨询助手

基于 Qwen1.5-1.8B-Chat 的旅游行业智能客服，使用 LoRA 微调技术，提供兼容 OpenAI 格式的 API 接口，可直接接入 OpenWebUI。

## 项目特点

- 🔧 使用 LoRA 高效微调技术
- 📱 提供兼容 OpenAI 格式的 API
- 🖥️ 支持 OpenWebUI 直接接入
- 🚀 快速部署，易于使用
- 📚 包含完整训练代码和数据集

## 项目结构

```
.
├── Qwen1.5-1.8B-Chat/          # 基础模型
├── qwen-lora-final/             # 训练好的 LoRA 权重
├── train.txt                     # 训练数据集
├── model.py                      # 训练代码
├── api_server.py                 # API 服务
├── requirements.txt              # 依赖包
└── README.md                     # 项目文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
python api_server.py
```

服务将在 `http://localhost:8000` 启动。

### 3. 接入 OpenWebUI

1. 打开 OpenWebUI
2. 进入设置 -&gt; 模型 -&gt; 添加 OpenAI 兼容模型
3. 配置如下：
   - API Base: `http://localhost:8000/v1`
   - API Key: 任意字符串（留空或填任意值）
   - Model Name: `qwen-lora`

### 4. API 接口示例

#### 聊天接口

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [
      {"role": "user", "content": "推荐一个适合夏天旅游的地方"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### 模型列表

```bash
curl http://localhost:8000/v1/models
```

#### 健康检查

```bash
curl http://localhost:8000/health
```

## 模型训练

### 数据集

数据集位于 `train.txt`，包含中英文旅游咨询问答对。可以使用 `filter_chinese_data.py` 过滤出仅中文数据：

```bash
python filter_chinese_data.py
```

### 训练模型

```bash
python model.py
```

训练参数可以在 `model.py` 中调整：
- `EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率
- `MAX_LEN`: 最大序列长度
- `GRADIENT_CLIP_NORM`: 梯度裁剪

## 技术栈

- FastAPI: API 服务框架
- Transformers: 模型加载和推理
- PEFT (LoRA): 参数高效微调
- PyTorch: 深度学习框架
- Uvicorn: ASGI 服务器

## 性能说明

- 基础模型: Qwen1.5-1.8B-Chat
- LoRA 参数量: 约 0.1%
- 推理显存需求: 约 4GB (FP16) 或 2GB (INT4)
- CPU 推理: 支持，但速度较慢

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 作者

AI Study Project

