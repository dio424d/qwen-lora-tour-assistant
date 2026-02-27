
"""
Qwen LoRA 旅游咨询助手 API 服务

提供兼容 OpenAI 格式的 API，支持 OpenWebUI 接入
支持对话历史记忆功能

作者: AI Study Project
"""

import time
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import setup_logger
from src.model import get_model
from src.database import get_database

logger = setup_logger(__name__)
model = get_model()
db = get_database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理
    
    启动时加载模型，关闭时清理资源
    """
    logger.info("=" * 60)
    logger.info("Qwen LoRA 旅游咨询助手 API 服务启动中...")
    logger.info("=" * 60)
    
    try:
        model.load()
        logger.info("服务启动成功！")
        yield
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        raise
    finally:
        logger.info("服务正在关闭...")


app = FastAPI(
    title="Qwen LoRA 旅游咨询助手 API",
    description="基于 Qwen1.5-1.8B 的旅游咨询智能客服，支持 OpenWebUI 接入，支持对话历史记忆",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="消息角色: system/user/assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求（兼容 OpenAI 格式）"""
    model: str = Field(default="qwen-lora", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成 tokens")
    stream: bool = Field(default=False, description="是否流式输出")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于对话记忆")
    user_id: Optional[str] = Field(default=None, description="用户ID")


class ChatCompletionResponseChoice(BaseModel):
    """聊天完成响应选项"""
    index: int = Field(default=0)
    message: ChatMessage
    finish_reason: str = Field(default="stop")


class ChatCompletionResponseUsage(BaseModel):
    """Token 使用统计"""
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class ChatCompletionResponse(BaseModel):
    """聊天完成响应（兼容 OpenAI 格式）"""
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = Field(default="model")
    created: int
    owned_by: str = Field(default="lora-tour-assistant")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    object: str = Field(default="list")
    data: List[ModelInfo]


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: List[SessionInfo]


@app.get("/health", summary="健康检查")
async def health_check():
    """
    检查服务健康状态
    
    Returns:
        服务状态信息
    """
    return {
        "status": "ok",
        "model_loaded": model.model is not None,
        "device": model.device if model.model else None,
        "database": "connected"
    }


@app.get("/v1/models", response_model=ModelListResponse, summary="获取模型列表")
async def list_models():
    """
    获取可用模型列表（兼容 OpenAI 格式）
    
    Returns:
        模型列表
    """
    return ModelListResponse(
        data=[
            ModelInfo(
                id="qwen-lora",
                created=int(time.time())
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, summary="聊天完成")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天完成接口（兼容 OpenAI 格式）
    
    支持多轮对话，可直接接入 OpenWebUI
    支持对话历史记忆（通过 session_id）
    
    Args:
        request: 聊天请求
    
    Returns:
        聊天响应
    """
    if not model.model or not model.tokenizer:
        raise HTTPException(status_code=503, detail="Model not ready, please try again later")
    
    session_id = request.session_id or str(uuid.uuid4())
    logger.debug(f"收到聊天请求: session_id={session_id}, {len(request.messages)} 条消息")
    
    try:
        if request.session_id:
            history = db.get_conversation_history(session_id, limit=10)
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]
        else:
            messages = []
        
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
            db.save_message(session_id, msg.role, msg.content, request.user_id)
        
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": settings.system_prompt})
        
        response_text = model.generate(
            messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        db.save_message(session_id, "assistant", response_text, request.user_id)
        
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        logger.debug(f"生成回复: session_id={session_id}, response={response_text[:50]}...")
        
        return ChatCompletionResponse(
            id=chat_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionResponseUsage()
        )
        
    except Exception as e:
        logger.error(f"生成回复失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/v1/sessions", response_model=SessionListResponse, summary="获取所有会话")
async def list_sessions():
    """
    获取所有会话列表
    
    Returns:
        会话列表
    """
    session_ids = db.get_all_sessions()
    
    sessions = []
    for session_id in session_ids:
        stats = db.get_session_stats(session_id)
        sessions.append(SessionInfo(
            session_id=session_id,
            total_messages=stats["total_messages"],
            user_messages=stats["user_messages"],
            assistant_messages=stats["assistant_messages"]
        ))
    
    return SessionListResponse(sessions=sessions)


@app.get("/v1/sessions/{session_id}/history", summary="获取会话历史")
async def get_session_history(session_id: str, limit: int = 20):
    """
    获取指定会话的对话历史
    
    Args:
        session_id: 会话ID
        limit: 最多返回多少条消息
    
    Returns:
        对话历史
    """
    history = db.get_conversation_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": history}


@app.delete("/v1/sessions/{session_id}", summary="删除会话")
async def delete_session(session_id: str):
    """
    删除指定会话
    
    Args:
        session_id: 会话ID
    
    Returns:
        删除结果
    """
    deleted_count = db.clear_session(session_id)
    return {"session_id": session_id, "deleted": deleted_count}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"启动 API 服务: http://{settings.host}:{settings.port}")
    logger.info(f"API 文档: http://{settings.host}:{settings.port}/docs")
    logger.info(f"对话记忆功能: 已启用 (SQLite)")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )

