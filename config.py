
import os
import torch

class Config:
    BASE_MODEL_PATH = "./Qwen1.5-1.8B-Chat"
    LORA_MODEL_PATH = "./qwen-lora-final"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HOST = "0.0.0.0"
    PORT = 8000
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.7
    SYSTEM_PROMPT = "你是一个专业的旅游行业客服，请简洁、友好地回复用户的旅游咨询，贴合旅游场景话术，保持真实回复风格。"

