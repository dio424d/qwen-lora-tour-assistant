# 对话记忆功能设置指南

本项目已集成 SQLite 数据库，支持对话历史记忆功能。

## 📋 功能特性

- ✅ **多会话管理** - 每个用户可以有多个独立会话
- ✅ **对话历史** - 自动保存和检索对话记录
- ✅ **上下文记忆** - AI 能记住之前的对话内容
- ✅ **会话统计** - 查看每个会话的消息数量
- ✅ **会话管理** - 支持查询和删除会话

## 🚀 快速开始

### 1. 初始化数据库

```bash
python init_db.py
```

执行后会自动创建 `conversations.db` 数据库文件。

### 2. 启动服务

```bash
python app.py
```

服务启动后会自动连接数据库。

## 📡 API 使用

### 基础对话（无记忆）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

### 带对话记忆的对话

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
    "user_id": "user-001"
  }'

# 第二次对话（会记住第一次的内容）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-lora",
    "messages": [
      {"role": "user", "content": "大概需要多少钱？"}
    ],
    "session_id": "user-123",
    "user_id": "user-001"
  }'
```

**关键参数：**
- `session_id` - 会话ID，相同 ID 的对话会共享历史
- `user_id` - 用户ID，用于区分不同用户（可选）

### 查看会话历史

```bash
# 获取所有会话
curl http://localhost:8000/v1/sessions

# 获取指定会话的历史
curl http://localhost:8000/v1/sessions/user-123/history

# 删除会话
curl -X DELETE http://localhost:8000/v1/sessions/user-123
```

## 🔧 配置说明

### 环境变量配置

在 `.env` 文件中添加：

```bash
# 数据库路径
QWEN_DB_PATH=conversations.db
```

### 数据库表结构

```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,      -- 会话ID
    user_id TEXT,                   -- 用户ID
    role TEXT NOT NULL,              -- 角色 (user/assistant/system)
    content TEXT NOT NULL,           -- 消息内容
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 创建时间
)
```

## 📊 数据库管理

### 查看数据库内容

```bash
# 使用 SQLite 命令行
sqlite3 conversations.db

# 查看所有会话
SELECT DISTINCT session_id FROM conversations;

# 查看某个会话的历史
SELECT role, content, created_at 
FROM conversations 
WHERE session_id = 'user-123' 
ORDER BY created_at ASC;

# 查看会话统计
SELECT 
    session_id,
    COUNT(*) as total_messages,
    COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages
FROM conversations
GROUP BY session_id;
```

### 清空数据库

```bash
# 删除数据库文件
rm conversations.db

# 重新初始化
python init_db.py
```

## 💡 使用示例

### Python 示例

```python
import requests

BASE_URL = "http://localhost:8000/v1"
session_id = "my-session"

# 第一次对话
response = requests.post(f"{BASE_URL}/chat/completions", json={
    "model": "qwen-lora",
    "messages": [{"role": "user", "content": "推荐一个旅游目的地"}],
    "session_id": session_id
})

# 第二次对话（带上下文）
response = requests.post(f"{BASE_URL}/chat/completions", json={
    "model": "qwen-lora",
    "messages": [{"role": "user", "content": "那里有什么好玩的？"}],
    "session_id": session_id
})
```

### JavaScript 示例

```javascript
const BASE_URL = "http://localhost:8000/v1";
const sessionId = "my-session";

// 第一次对话
fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        model: "qwen-lora",
        messages: [{ role: "user", content: "推荐一个旅游目的地" }],
        session_id: sessionId
    })
});

// 第二次对话（带上下文）
fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        model: "qwen-lora",
        messages: [{ role: "user", content: "那里有什么好玩的？" }],
        session_id: sessionId
    })
});
```

## 🎯 最佳实践

1. **Session ID 生成**
   - 使用 UUID 或用户 ID + 时间戳
   - 示例：`user-{user_id}-{timestamp}`

2. **对话历史限制**
   - 默认保留最近 10 条消息
   - 可通过 `limit` 参数调整

3. **会话清理**
   - 定期清理过期会话
   - 使用 DELETE 接口删除不需要的会话

4. **用户隔离**
   - 使用 `user_id` 区分不同用户
   - 同一用户可以有多个 `session_id`

## 🔒 安全建议

1. **数据库权限**
   - 确保 `conversations.db` 文件权限正确
   - 避免上传到公共仓库（已在 .gitignore 中配置）

2. **敏感信息**
   - 不要在对话中存储敏感信息
   - 定期备份和清理数据库

3. **生产环境**
   - 考虑使用 PostgreSQL 或 MySQL 替代 SQLite
   - 添加数据库加密

## 📝 完整示例

运行 `example_with_memory.py` 查看完整使用示例：

```bash
python example_with_memory.py
```

