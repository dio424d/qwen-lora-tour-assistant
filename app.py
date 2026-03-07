
"""
Qwen LoRA 旅游咨询助手 API 服务

提供兼容 OpenAI 格式的 API，支持 OpenWebUI 接入
支持对话历史记忆功能

作者: AI Study Project
"""

import time
import uuid
import threading
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import setup_logger
from src.model import QwenLoRAModel, get_model
from src.database import ConversationDatabase, get_database
from src.skill import SkillManager
from src.skills import HotelSearchSkill, AttractionSearchSkill, RestaurantSearchSkill, RoutePlanningSkill, WeatherSkill
from src.amap import AmapAPI

logger = setup_logger(__name__)
model = get_model()
db = get_database()

# 初始化技能系统
skill_manager = SkillManager()

# 初始化高德地图 API（如果配置了 API 密钥）
amap_api = None
if settings.amap_api_key:
    amap_api = AmapAPI(settings.amap_api_key)
    # 注册技能
    skill_manager.register_skill(HotelSearchSkill(amap_api))
    skill_manager.register_skill(AttractionSearchSkill(amap_api))
    skill_manager.register_skill(RestaurantSearchSkill(amap_api))
    skill_manager.register_skill(RoutePlanningSkill(amap_api))
    skill_manager.register_skill(WeatherSkill(amap_api))
    logger.info("技能系统初始化完成")
else:
    logger.warning("未配置高德地图 API 密钥，技能系统未初始化")


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
    max_tokens: int = Field(default=1024, ge=1, le=2048, description="最大生成 tokens")
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


class SkillInfo(BaseModel):
    """技能信息"""
    name: str
    description: str


class SkillListResponse(BaseModel):
    """技能列表响应"""
    skills: List[SkillInfo]


class SkillExecuteRequest(BaseModel):
    """技能执行请求"""
    skill_name: str
    parameters: Dict[str, Any]


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
        "database": "connected",
        "skills_available": len(skill_manager.list_skills()) > 0
    }


@app.get("/v1/skills", response_model=SkillListResponse, summary="获取技能列表")
async def list_skills():
    """
    获取所有可用技能
    
    Returns:
        技能列表
    """
    skills = skill_manager.list_skills()
    return SkillListResponse(
        skills=[SkillInfo(name=skill["name"], description=skill["description"]) for skill in skills]
    )


@app.post("/v1/skills/execute", summary="执行技能")
async def execute_skill(request: SkillExecuteRequest) -> Dict[str, Any]:
    """
    执行指定技能
    
    Args:
        request: 技能执行请求
    
    Returns:
        技能执行结果
    """
    skill = skill_manager.get_skill(request.skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    try:
        result = skill.execute(**request.parameters)
        return {
            "skill": request.skill_name,
            "result": result
        }
    except Exception as e:
        logger.error(f"技能执行失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Skill execution failed")


@app.get("/", summary="聊天界面")
async def get_chat_interface():
    """
    返回聊天界面 HTML
    
    Returns:
        HTML 页面
    """
    import os
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_chat.html")
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


from fastapi.responses import HTMLResponse


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
    支持自动调用技能（如酒店查询、路线规划等）
    
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
        
        # 检查是否需要调用技能
        last_message = request.messages[-1].content if request.messages else ""
        skill = skill_manager.find_skill(last_message)
        
        if skill:
            # 提取参数
            params = {}
            # 简单的参数提取逻辑
            logger.debug(f"用户输入: '{last_message}'")
            
            if "酒店" in last_message:
                # 提取城市名称
                import re
                city_match = re.search(r'在(.*?)找酒店', last_message)
                if city_match:
                    params["city"] = city_match.group(1)
                    logger.debug(f"提取城市名称: {params['city']}")
            elif "路线" in last_message:
                # 提取起点和终点
                origin_match = re.search(r'从(.*?)到(.*?)', last_message)
                if origin_match:
                    params["origin"] = origin_match.group(1)
                    params["destination"] = origin_match.group(2)
                    logger.debug(f"提取路线: {params['origin']} -> {params['destination']}")
            elif "天气" in last_message:
                # 提取城市名称
                import re
                # 改进正则表达式，去除前后空格
                city_match = re.search(r'(.*?)的天气', last_message)
                if city_match:
                    params["city"] = city_match.group(1).strip()
                    logger.debug(f"提取城市名称: {params['city']}")
                else:
                    # 尝试其他格式
                    city_match = re.search(r'(.*?)天气', last_message)
                    if city_match:
                        params["city"] = city_match.group(1).strip()
                        logger.debug(f"提取城市名称（格式2）: {params['city']}")
                    else:
                        # 尝试只提取城市名
                        city_match = re.search(r'([^\s天气]+)', last_message)
                        if city_match:
                            params["city"] = city_match.group(1).strip()
                            logger.debug(f"提取城市名称（格式3）: {params['city']}")
            
            # 执行技能
            logger.debug(f"执行技能: {skill.name()}, 参数: {params}")
            skill_result = skill.execute(**params)
            logger.debug(f"技能执行结果: {skill_result}")
            
            # 生成回复
            response_text = f"根据查询结果：\n\n"
            
            if skill.name() == "hotel_search":
                hotels = skill_result.get("hotels", [])
                if hotels:
                    response_text += "找到以下酒店：\n"
                    for i, hotel in enumerate(hotels, 1):
                        response_text += f"{i}. {hotel['name']}\n"
                        response_text += f"   地址：{hotel['address']}\n"
                        if hotel['tel']:
                            response_text += f"   电话：{hotel['tel']}\n"
                        if hotel['distance']:
                            response_text += f"   距离：{hotel['distance']}米\n"
                else:
                    response_text += "未找到相关酒店"
            
            elif skill.name() == "attraction_search":
                attractions = skill_result.get("attractions", [])
                if attractions:
                    response_text += "找到以下景点：\n"
                    for i, attraction in enumerate(attractions, 1):
                        response_text += f"{i}. {attraction['name']}\n"
                        response_text += f"   地址：{attraction['address']}\n"
                        if attraction['tel']:
                            response_text += f"   电话：{attraction['tel']}\n"
                        if attraction['distance']:
                            response_text += f"   距离：{attraction['distance']}米\n"
                else:
                    response_text += "未找到相关景点"
            
            elif skill.name() == "restaurant_search":
                restaurants = skill_result.get("restaurants", [])
                if restaurants:
                    response_text += "找到以下餐厅：\n"
                    for i, restaurant in enumerate(restaurants, 1):
                        response_text += f"{i}. {restaurant['name']}\n"
                        response_text += f"   地址：{restaurant['address']}\n"
                        if restaurant['tel']:
                            response_text += f"   电话：{restaurant['tel']}\n"
                        if restaurant['distance']:
                            response_text += f"   距离：{restaurant['distance']}米\n"
                else:
                    response_text += "未找到相关餐厅"
            
            elif skill.name() == "route_planning":
                routes = skill_result.get("routes", [])
                if routes:
                    route = routes[0]
                    response_text += "路线规划：\n"
                    response_text += f"距离：{route['distance']}米\n"
                    response_text += f"预计时间：{route['duration']}秒\n"
                else:
                    response_text += "无法规划路线"
            
            elif skill.name() == "weather":
                weather = skill_result.get("weather", [])
                city = skill_result.get("city", "")
                if weather:
                    current_weather = weather[0]
                    response_text += f"📍 {current_weather.get('city', city)} 天气信息\n\n"
                    response_text += f"🌡️ 温度：{current_weather.get('temperature', 'N/A')}°C\n"
                    response_text += f"☁️ 天气：{current_weather.get('weather', 'N/A')}\n"
                    response_text += f"🌬️ 风向：{current_weather.get('winddirection', 'N/A')}\n"
                    response_text += f"💨 风力：{current_weather.get('windpower', 'N/A')}\n"
                    response_text += f"💧 湿度：{current_weather.get('humidity', 'N/A')}%\n"
                    response_text += f"🕐 更新时间：{current_weather.get('reporttime', 'N/A')}\n"
                    
                    # 显示未来几天预报
                    if len(weather) > 1:
                        response_text += f"\n📅 未来几天预报：\n"
                        for forecast in weather[1:4]:  # 显示未来3天
                            if 'date' in forecast:
                                response_text += f"\n{forecast.get('date', '')}:\n"
                                response_text += f"  白天：{forecast.get('dayweather', 'N/A')} {forecast.get('daytemp', 'N/A')}°C\n"
                                response_text += f"  夜间：{forecast.get('nightweather', 'N/A')} {forecast.get('nighttemp', 'N/A')}°C\n"
                else:
                    if not city:
                        response_text += "❌ 无法获取天气信息：请提供城市名称\n"
                        response_text += "💡 示例：北京天气、上海天气怎么样"
                    else:
                        response_text += f"❌ 无法获取 {city} 的天气信息\n"
                        response_text += "💡 可能原因：\n"
                        response_text += "1. 城市名称不正确\n"
                        response_text += "2. 高德地图 API 密钥未配置或已过期\n"
                        response_text += "3. 网络连接问题"
        else:
            # 正常生成回复
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
    import webbrowser
    import time
    
    logger.info(f"启动 API 服务: http://localhost:{settings.port}")
    logger.info(f"聊天界面: http://localhost:{settings.port}")
    logger.info(f"API 文档: http://localhost:{settings.port}/docs")
    logger.info(f"对话记忆功能: 已启用 (SQLite)")
    
    # 启动 API 服务器
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )
    
    # 延迟 2 秒后自动打开聊天界面
    def open_browser():
        time.sleep(2)
        # 打开浏览器访问聊天界面
        webbrowser.open(f"http://{settings.host}:{settings.port}")
    
    # 在新线程中打开浏览器，不阻塞主线程
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

