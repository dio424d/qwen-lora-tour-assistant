
# Qwen LoRA 旅游咨询助手

&gt; 基于 Qwen1.5-1.8B-Chat 的旅游行业智能客服，使用 LoRA 高效微调技术，提供企业级 API 服务

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 项目简介

这是一个完整的大语言模型微调与部署项目，展示了从模型训练到生产部署的完整流程。

**核心特性：**
- ✅ **LoRA 参数高效微调** - 仅训练约 0.1% 的参数，大幅降低显存需求
- ✅ **OpenAI 兼容 API** - 可直接对接 OpenWebUI、LangChain 等生态
- ✅ **对话历史记忆** - 基于 SQLite 的多会话管理，支持上下文记忆
- ✅ **技能系统集成** - 集成高德地图API，支持天气查询、酒店搜索、路线规划等
- ✅ **语音会话功能** - 支持语音输入和语音合成，提供更自然的交互体验
- ✅ **企业级架构** - 模块化设计、日志系统、配置管理、Docker 支持
- ✅ **生产就绪** - 错误处理、健康检查、性能监控

## 🎯 项目亮点

### 1. 技术栈
- **后端框架**: FastAPI + Pydantic（类型安全、自动文档）
- **深度学习**: PyTorch + Transformers + PEFT
- **语音处理**: Edge TTS + gTTS（语音合成）
- **前端**: HTML5 + JavaScript + Web Speech API（语音识别）
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
│   ├── model.py             # 模型封装类（单例模式）
│   ├── database.py          # 对话数据库管理（SQLite）
│   ├── amap.py              # 高德地图API客户端
│   ├── skill.py             # 技能系统基类
│   └── skills.py            # 具体技能实现（天气/酒店/路线等）
├── app.py                    # FastAPI 主应用（推荐使用，支持对话记忆）
├── init_db.py               # 数据库初始化脚本
├── example_with_memory.py   # 对话记忆功能示例
├── simple_api.py             # 简化版 API
├── api_server.py             # 旧版 API
├── model.py                  # LoRA 训练代码
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 镜像构建
├── docker-compose.yml        # Docker Compose 编排
├── .env.example              # 环境变量示例
├── .gitignore                # Git 忽略规则
├── GITHUB_GUIDE.md           # GitHub 上传指南
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

# 安装语音功能依赖
pip install edge-tts gTTS

# 安装FunASR语音识别依赖（可选，用于提升识别准确率）
pip install modelscope onnxruntime

# 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 修改配置
```

### 2. 初始化数据库（对话记忆功能）

```bash
# 初始化对话数据库
python init_db.py
```

数据库文件 `conversations.db` 会自动创建，用于存储对话历史。

### 3. 启动服务

```bash
# 方式1：直接运行
python app.py

# 方式2：使用 uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 方式3：Docker（推荐生产环境）
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

### 聊天完成（支持对话记忆）

**POST** `/v1/chat/completions`

#### 基础用法（无对话记忆）
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

#### 高级用法（带对话记忆）
```bash
# 第一次对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [
      {"role": "user", "content": "我想去云南旅游"}
    ],
    "session_id": "user-123",
    "temperature": 0.7
  }'

# 第二次对话（会记住之前的上下文）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [
      {"role": "user", "content": "大概需要多少钱？"}
    ],
    "session_id": "user-123",
    "temperature": 0.7
  }'
```

**请求参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名称 |
| `messages` | array | 是 | 对话消息列表 |
| `temperature` | number | 否 | 温度参数 (0-2) |
| `max_tokens` | number | 否 | 最大生成 tokens |
| `session_id` | string | 否 | 会话ID，用于对话记忆 |
| `user_id` | string | 否 | 用户ID |

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

#### 获取所有会话

**GET** `/v1/sessions`

```bash
curl http://localhost:8000/v1/sessions
```

#### 获取会话历史

**GET** `/v1/sessions/{session_id}/history`

```bash
curl http://localhost:8000/v1/sessions/user-123/history?limit=10
```

#### 删除会话

**DELETE** `/v1/sessions/{session_id}`

```bash
curl -X DELETE http://localhost:8000/v1/sessions/user-123
```

- `GET /health` - 健康检查
- `GET /v1/models` - 模型列表
- `POST /v1/tts` - 文本转语音
- `POST /v1/voice/recognize` - 语音识别
- `GET /v1/voice/hotwords` - 获取热词表

## 🎤 语音功能

### 1. 文本转语音接口

**POST** `/v1/tts`

**请求参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 要转换的文本 |

**响应：**
- 成功：返回 MP3 音频数据（Content-Type: audio/mpeg）
- 失败：返回错误信息

**使用示例：**
```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "您好，欢迎使用旅游咨询助手"}' \
  -o output.mp3
```

### 2. 语音识别接口

**POST** `/v1/voice/recognize`

**请求参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_data` | string | 是 | Base64编码的音频数据 |
| `use_hotwords` | boolean | 否 | 是否使用热词表（默认true） |

**响应：**
```json
{
  "text": "识别到的文本",
  "confidence": 0.95,
  "engine": "FunASR",
  "hotwords_matched": ["大唐不夜城", "兵马俑"]
}
```

**使用示例：**
```bash
# 获取热词表
curl http://localhost:8000/v1/voice/hotwords

# 语音识别
curl -X POST http://localhost:8000/v1/voice/recognize \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_encoded_audio", "use_hotwords": true}'
```

### 3. 前端语音功能

项目提供了完整的前端界面，支持：
- 🎤 **语音输入** - 点击麦克风按钮进行语音输入
- 🔊 **语音播放** - 每条回复后可点击播放按钮听取语音

**技术实现：**
- **语音识别**：FunASR（优先）+ Web Speech API（保底）
- **语音合成**：Edge TTS + gTTS
- **音频格式**：MP3
- **热词识别**：内置文旅专属热词表

**语音识别策略：**
1. 优先使用FunASR（需要安装modelscope）
2. 如果FunASR不可用，自动切换到Web Speech API
3. 支持文旅热词识别，提升专有名词识别率

### 4. 语音服务配置

**支持的语音引擎：**

**语音合成：**
| 引擎 | 特点 | 配置 |
|------|------|------|
| Edge TTS | 微软语音服务，自然度高 | 默认使用 |
| gTTS | Google 语音服务，兼容性好 | 备用方案 |

**语音识别：**
| 引擎 | 特点 | 配置 |
|------|------|------|
| FunASR | 阿里达摩院语音识别，支持热词 | 优先使用（需安装modelscope） |
| Web Speech API | 浏览器内置语音识别 | 保底方案 |

**安装依赖：**
```bash
# 语音合成依赖
pip install edge-tts gTTS

# 语音识别依赖（可选，用于FunASR）
pip install modelscope onnxruntime
```

### 5. 文旅专属热词表

项目内置了丰富的文旅专属热词，包括：
- 景点：大唐不夜城、兵马俑、大雁塔、华清池、西安城墙等
- 美食：肉夹馍、回民街、永兴坊等
- 文化：长安十二时辰、大唐芙蓉园、陕西历史博物馆等
- 其他：文旅一卡通、钟鼓楼、碑林等

热词表会自动匹配语音识别结果，提升专有名词识别准确率。
```

## 🤖 技能系统

### 功能列表

项目集成了高德地图API，提供以下智能技能：

| 技能名称 | 触发词 | 功能说明 | 示例 |
|---------|--------|---------|------|
| 天气查询 | 天气、气温 | 查询实时天气和未来预报 | "北京天气"、"上海天气怎么样" |
| 酒店搜索 | 酒店、住宿 | 搜索目的地酒店 | "在北京找酒店" |
| 景点查询 | 景点、旅游 | 搜索旅游景点 | "北京有什么景点" |
| 餐厅搜索 | 餐厅、美食 | 搜索餐厅美食 | "北京有什么好吃的" |
| 路线规划 | 路线、交通 | 规划驾车路线 | "从北京到上海怎么走" |

### 配置方法

1. 申请高德地图API密钥：https://lbs.amap.com/dev/key/app
2. 编辑 `.env` 文件：
```bash
QWEN_AMAP_API_KEY=您的API密钥
```

### 使用示例

```bash
# 天气查询
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [{"role": "user", "content": "北京天气"}]
  }'

# 酒店搜索
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [{"role": "user", "content": "在上海找酒店"}]
  }'
```

### 技能API

#### 获取技能列表

**GET** `/v1/skills`

```bash
curl http://localhost:8000/v1/skills
```

#### 执行技能

**POST** `/v1/skills/execute`

```bash
curl -X POST http://localhost:8000/v1/skills/execute \
  -H "Content-Type: application/json" \
  -d '{
    "skill_name": "weather",
    "parameters": {"city": "北京"}
  }'
```

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

def get_model() -> QwenLoRAModel:
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

AI Study Project

---

**如果这个项目对你有帮助，请给个 Star ⭐**

