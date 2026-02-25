
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    应用配置类，支持环境变量覆盖
    
    环境变量示例:
        QWEN_BASE_MODEL_PATH=./Qwen1.5-1.8B-Chat
        QWEN_LORA_MODEL_PATH=./qwen-lora-final
        QWEN_HOST=0.0.0.0
        QWEN_PORT=8000
    """
    
    base_model_path: str = "./Qwen1.5-1.8B-Chat"
    lora_model_path: str = "./qwen-lora-final"
    
    host: str = "0.0.0.0"
    port: int = 8000
    
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    system_prompt: str = "你是一个专业的旅游行业客服，请简洁、友好地回复用户的旅游咨询，贴合旅游场景话术，保持真实回复风格。"
    
    device: str = "auto"
    
    class Config:
        env_prefix = "QWEN_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()

