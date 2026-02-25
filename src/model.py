
import torch
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import settings
from src.logger import setup_logger

logger = setup_logger(__name__)


class QwenLoRAModel:
    """
    Qwen LoRA 模型封装类
    
    提供模型加载、推理等功能
    """
    
    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = self._get_device()
        
    def _get_device(self) -&gt; str:
        """
        获取运行设备
        
        Returns:
            设备名称（cuda/cpu）
        """
        if settings.device != "auto":
            return settings.device
        
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("未检测到 CUDA，使用 CPU 推理")
        
        return device
    
    def load(self) -&gt; None:
        """
        加载模型和 tokenizer
        """
        logger.info("开始加载模型...")
        
        try:
            self._load_tokenizer()
            self._load_base_model()
            self._load_lora_weights()
            
            self.model.eval()
            logger.info("模型加载完成！")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise
    
    def _load_tokenizer(self) -&gt; None:
        """加载 tokenizer"""
        logger.debug(f"加载 tokenizer: {settings.base_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def _load_base_model(self) -&gt; None:
        """加载基础模型"""
        logger.debug(f"加载基础模型: {settings.base_model_path}")
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_path,
            device_map=self.device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    def _load_lora_weights(self) -&gt; None:
        """加载 LoRA 权重"""
        logger.debug(f"加载 LoRA 权重: {settings.lora_model_path}")
        
        self.model = PeftModel.from_pretrained(
            self.model,
            settings.lora_model_path
        )
        
        if self.device == "cuda":
            self.model = self.model.merge_and_unload()
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        构建对话提示词
        
        Args:
            messages: 对话消息列表
        
        Returns:
            格式化的提示词
        """
        prompt = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"&lt;|im_start|&gt;system\n{content}&lt;|im_end|&gt;\n"
            elif role == "user":
                prompt += f"&lt;|im_start|&gt;user\n{content}&lt;|im_end|&gt;\n"
            elif role == "assistant":
                prompt += f"&lt;|im_start|&gt;assistant\n{content}&lt;|im_end|&gt;\n"
        
        prompt += "&lt;|im_start|&gt;assistant\n"
        return prompt
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None
    ) -&gt; str:
        """
        生成回复
        
        Args:
            messages: 对话消息列表
            max_new_tokens: 最大生成 tokens
            temperature: 温度参数
            top_p: top_p 参数
            repetition_penalty: 重复惩罚
        
        Returns:
            生成的回复文本
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        max_new_tokens = max_new_tokens or settings.max_new_tokens
        temperature = temperature if temperature is not None else settings.temperature
        top_p = top_p if top_p is not None else settings.top_p
        repetition_penalty = repetition_penalty or settings.repetition_penalty
        
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature &gt; 0,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


_model_instance: Optional[QwenLoRAModel] = None


def get_model() -&gt; QwenLoRAModel:
    """
    获取单例模型实例
    
    Returns:
        QwenLoRAModel 实例
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = QwenLoRAModel()
    
    return _model_instance

