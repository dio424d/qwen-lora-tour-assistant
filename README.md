
# Qwen LoRA 旅游咨询助手

&gt; 基于 Qwen1.5-1.8B-Chat 的旅游行业智能客服，使用 LoRA 高效微调技术，提供企业级 API 服务

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 项目简介

这是一个完整的大语言模型微调与部署项目，展示了从模型训练到生产部署的完整流程。

## 🎯 项目亮点

### 1. 技术栈
- **后端框架**: FastAPI + Pydantic（类型安全、自动文档）
- **深度学习**: PyTorch + Transformers + PEFT
- **工程化**: Docker + docker-compose + 日志系统 + 配置管理

### 2. 代码质量
- 模块化设计（`src/` 目录）
- 类型注解完整
- 完善的错误处理
- 清晰的代码注释

### 3. 工程实践
- 单例模式管理模型生命周期
- 环境变量配置（12-Factor App）
- 结构化日志记录
- API 版本管理

## 📁 项目结构

```
.
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── config.py            # 配置管理（Pydantic Settings）
│   ├── logger.py            # 日志系统
│   └── model.py             # 模型封装类（单例模式）
├── app.py                    # FastAPI 主应用
├── model.py                  # LoRA 训练代码
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 镜像构建
├── docker-compose.yml        # Docker Compose 编排
├── .env.example              # 环境变量示例
└── README.md                 # 项目文档
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/dio424d/qwen-lora-tour-assistant.git
cd qwen-lora-tour-assistant

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 修改配置
```

### 2. 启动服务

```bash
# 方式1：直接运行
python app.py

# 方式2：使用 uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 方式3：Docker
docker-compose up -d
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 3. 接入 OpenWebUI

1. 打开 OpenWebUI
2. 设置 → 模型 → 添加 OpenAI 兼容模型
3. 配置：
   - **API Base**: `http://localhost:8000/v1`
   - **API Key**: `sk-任意字符串`
   - **Model Name**: `qwen-lora`

## 📡 API 接口文档

### 聊天完成

**POST** `/v1/chat/completions`

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

**响应示例：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen-lora",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "推荐您去青岛..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

### 其他接口

- `GET /health` - 健康检查
- `GET /v1/models` - 模型列表

## 🎓 模型训练

### 训练代码

```bash
python model.py
```

### 核心训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EPOCHS` | 训练轮数 | 1 |
| `LEARNING_RATE` | 学习率 | 2e-5 |
| `MAX_LEN` | 最大序列长度 | 128 |
| `GRADIENT_CLIP_NORM` | 梯度裁剪 | 1.0 |
| `LORA_R` | LoRA 秩 | 2 |
| `LORA_ALPHA` | LoRA alpha | 4 |

### LoRA 配置亮点

```python
LoraConfig(
    r=2,                    # 低秩矩阵维度
    lora_alpha=4,           # 缩放因子
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t qwen-lora-api .
```

### 使用 Docker Compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 基础模型 | Qwen1.5-1.8B-Chat |
| LoRA 参数量 | ~0.1% |
| 显存需求 (FP16) | ~4GB |
| 显存需求 (INT4) | ~2GB |
| CPU 推理 | 支持（较慢） |

## 🛠️ 技术栈详解

### 后端框架
- **FastAPI**: 高性能异步 Web 框架
- **Pydantic**: 数据验证和设置管理
- **Uvicorn**: ASGI 服务器

### 深度学习
- **PyTorch**: 深度学习框架
- **Transformers**: 预训练模型库
- **PEFT**: 参数高效微调（LoRA）

### 工程化
- **Docker**: 容器化部署
- **结构化日志**: 日志追踪和调试
- **环境变量**: 12-Factor App 最佳实践

## 📝 代码设计模式

### 1. 单例模式（模型管理）
```python
# src/model.py
_model_instance: Optional[QwenLoRAModel] = None

def get_model() -&gt; QwenLoRAModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = QwenLoRAModel()
    return _model_instance
```

### 2. 依赖注入（配置管理）
```python
# src/config.py
class Settings(BaseSettings):
    base_model_path: str = "./Qwen1.5-1.8B-Chat"
    # ...
```

### 3. 生命周期管理
```python
@app.on_event("startup")
async def startup_event():
    model.load()
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 👨‍💻 作者

JIASHAOYUN


