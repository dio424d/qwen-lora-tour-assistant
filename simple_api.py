import time
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig  # 新增：导入PeftConfig

app = FastAPI(title="Qwen LoRA API", description="旅游咨询助手API服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
# 强制指定CPU（适配你的场景，避免自动检测CUDA导致的问题）
device = "cpu"  

SYSTEM_PROMPT = "你是一个专业的旅游行业客服，请简洁、友好地回复用户的旅游咨询，贴合旅游场景话术，保持真实回复风格。"

def load_model():
    global model, tokenizer
    
    print("正在加载模型...")
    
    base_model_path = "./Qwen1.5-1.8B-Chat"
    lora_model_path = "./qwen-lora-final"
    
    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 加载基础模型（CPU专属配置，降低内存占用）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device,  # 强制CPU
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True  # 新增：仅加载本地文件，避免联网
    )
    
    # 3. 关键修复：加载训练时的LoRA配置（自动匹配r=2）
    peft_config = PeftConfig.from_pretrained(lora_model_path)
    peft_config.r = 2  # 强制指定和训练时一致的r值，双重保险
    
    # 4. 加载LoRA权重（使用匹配的配置）
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        config=peft_config  # 传入正确的LoRA配置
    )
    model.eval()
    
    print("模型加载完成！")

def generate_response(messages: List[Dict], max_new_tokens: int = 1024, temperature: float = 0.7):
    prompt = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"  # 修复转义符错误
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"  # 修复转义符错误
    
    # 适配CPU：限制输入长度，避免内存溢出
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=1024  # 新增：限制最大输入长度
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            # 新增：CPU生成加速配置
            num_beams=1,
            early_stopping=True
        )
    
    # 解码时跳过输入部分，只取生成的回复
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  # 新增：清理多余空格
    )
    return response.strip()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen-lora"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "lora-tour-assistant"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[Model]

# 修复：替换废弃的on_event为lifespan（可选，兼容旧版本FastAPI）
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[
            Model(id="qwen-lora", created=int(time.time()))
        ]
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # 自动添加系统提示（如果用户没传）
    if not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    try:
        response_text = generate_response(
            messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    
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

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": device}

if __name__ == "__main__":
    import uvicorn
    # 新增：添加日志配置，方便排查问题
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )