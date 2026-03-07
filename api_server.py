import os
import time
import json
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Qwen LoRA API", description="基于Qwen 1.5-1.8B的旅游咨询助手API服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = "你是一个专业的旅游行业客服，请简洁、友好地回复用户的旅游咨询，贴合旅游场景话术，保持真实回复风格。"

def load_model():
    global model, tokenizer
    
    print("正在加载模型...")
    
    base_model_path = "./Qwen1.5-1.8B-Chat"
    lora_model_path = "./qwen-lora-final"
    
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
    stream: bool = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

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
    created: int = int(time.time())
    owned_by: str = "lora-tour-assistant"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[Model]

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[
            Model(id="qwen-lora")
        ]
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    response_text = generate_response(
        messages,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
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
        usage=ChatCompletionResponseUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
