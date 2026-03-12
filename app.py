"""
Qwen LoRA 旅游咨询助手 API 服务

提供兼容 OpenAI 格式的 API，支持 OpenWebUI 接入
支持语音识别和语音合成功能

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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import Config
from src.voice_recognition import get_voice_service, AudioRecognitionRequest

# 全局变量
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

# 语音识别服务
voice_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, voice_service
    
    # 启动时加载模型
    print("正在加载模型...")
    
    base_model_path = config.BASE_MODEL_PATH
    lora_model_path = config.LORA_MODEL_PATH
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = model.merge_and_unload()
    model.eval()
    
    print("模型加载完成！")
    
    # 初始化语音识别服务
    try:
        voice_service = get_voice_service()
        print("语音识别服务初始化完成")
    except Exception as e:
        print(f"语音识别服务初始化失败: {e}")
        voice_service = None
    
    yield
    
    # 关闭时清理资源
    print("正在关闭服务...")

app = FastAPI(
    title="Qwen LoRA API",
    description="基于Qwen 1.5-1.8B的旅游咨询助手API服务",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_response(messages: List[Dict], max_new_tokens: int = 1024, temperature: float = 0.7):
    prompt = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen-lora"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class TTSRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Qwen LoRA API 服务运行中"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "qwen-lora"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-lora",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen-lora"
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天接口
    """
    try:
        messages = [msg.dict() for msg in request.messages]
        response_text = generate_response(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": sum(len(msg.content) for msg in request.messages),
                "completion_tokens": len(response_text),
                "total_tokens": sum(len(msg.content) for msg in request.messages) + len(response_text)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tts", summary="文本转语音")
async def text_to_speech(request: TTSRequest):
    """
    文本转语音接口
    """
    try:
        from fastapi.responses import Response
        
        # 使用 Edge TTS（免费）
        import edge_tts
        import asyncio
        
        tts = edge_tts.Communicate(request.text, "zh-CN-YunxiNeural")
        audio_data = b""
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return Response(content=audio_data, media_type="audio/mpeg")
    except ImportError:
        # 回退到gTTS
        from gtts import gTTS
        import io
        from fastapi.responses import Response
        
        tts = gTTS(text=request.text, lang='zh-cn')
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        return Response(content=audio_io.read(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.post("/v1/voice/recognize", summary="语音识别")
async def voice_recognize(request: AudioRecognitionRequest):
    """
    语音识别接口
    
    支持FunASR语音识别，如果FunASR不可用则返回错误让前端使用Web Speech API
    """
    if voice_service is None:
        raise HTTPException(
            status_code=503,
            detail="语音识别服务未初始化，请使用Web Speech API"
        )
    
    if request.audio_data:
        import base64
        try:
            audio_data = base64.b64decode(request.audio_data)
            result = await voice_service.recognize_audio(audio_data, request.use_hotwords)
            return result
        except Exception as e:
            print(f"语音识别失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="音频数据不能为空")

@app.get("/v1/voice/hotwords", summary="获取热词表")
async def get_hotwords():
    """
    获取文旅专属热词表
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="语音识别服务未初始化")
    
    return {"hotwords": voice_service.get_hotwords()}

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    
    print(f"启动 API 服务: http://localhost:{config.PORT}")
    print(f"聊天界面: http://localhost:{config.PORT}")
    print(f"API 文档: http://localhost:{config.PORT}/docs")
    
    # 启动 API 服务器
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
    
    # 延迟 2 秒后自动打开聊天界面
    def open_browser():
        time.sleep(2)
        webbrowser.open(f"http://{config.HOST}:{config.PORT}")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()