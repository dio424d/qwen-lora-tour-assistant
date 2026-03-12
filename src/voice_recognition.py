import io
import wave
import numpy as np
from typing import Optional, List
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel

# 文旅专属热词表
CULTURAL_TOURISM_HOTWORDS = [
    "大唐不夜城",
    "兵马俑",
    "肉夹馍",
    "文旅一卡通",
    "大雁塔",
    "华清池",
    "西安城墙",
    "回民街",
    "钟鼓楼",
    "秦始皇陵",
    "华清宫",
    "骊山",
    "临潼",
    "长安十二时辰",
    "大唐芙蓉园",
    "陕西历史博物馆",
    "碑林",
    "书院门",
    "德福巷",
    "永兴坊",
    "白鹿原",
    "袁家村",
    "马嵬驿",
    "法门寺",
    "太白山",
    "华山",
    "壶口瀑布",
    "黄帝陵",
    "延安革命纪念馆",
    "宝塔山",
    "枣园",
    "杨家岭",
    "王家坪",
    "清凉山",
    "南泥湾",
    "梁家河",
    "黄帝陵",
    "茂陵",
    "乾陵",
    "昭陵",
    "建陵",
    "大雁塔北广场",
    "大雁塔南广场",
    "小雁塔",
    "青龙寺",
    "大兴善寺",
    "西安博物院",
    "西安事变纪念馆",
    "八路军西安办事处纪念馆",
    "西安事变旧址",
    "张学良公馆",
    "杨虎城公馆",
    "高桂滋公馆",
    "止园",
    "七贤庄",
    "八路军办事处",
    "西京招待所",
    "西安事变",
    "双十二事变",
]


class AudioRecognitionRequest(BaseModel):
    """语音识别请求"""
    audio_data: Optional[str] = None  # Base64编码的音频数据
    use_hotwords: bool = True  # 是否使用热词表


class AudioRecognitionResponse(BaseModel):
    """语音识别响应"""
    text: str
    confidence: float
    engine: str  # 使用的识别引擎
    hotwords_matched: List[str] = []  # 匹配到的热词


class FunASRRecognizer:
    """FunASR语音识别器"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.hotwords = CULTURAL_TOURISM_HOTWORDS
        
    def load_model(self):
        """加载FunASR模型"""
        try:
            # 先尝试导入必要的库
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
            except ImportError as ie:
                print(f"FunASR依赖库未安装: {ie}")
                print("提示: 如需使用FunASR，请运行: pip install modelscope onnxruntime")
                return False
            
            # 使用FunASR的paraformer模型
            print("正在加载FunASR模型...")
            self.model = pipeline(
                task=Tasks.auto_speech_recognition,
                model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                model_revision='v1.0.0',
                device='cpu'  # 使用CPU，避免显存问题
            )
            self.model_loaded = True
            print("FunASR模型加载成功！")
            return True
        except Exception as e:
            print(f"FunASR模型加载失败: {e}")
            print("将使用Web Speech API作为语音识别方案")
            return False
    
    def recognize(self, audio_data: bytes) -> Optional[dict]:
        """识别音频"""
        if not self.model_loaded:
            return None
            
        try:
            # 将音频数据转换为模型需要的格式
            result = self.model(audio_data)
            return result
        except Exception as e:
            print(f"FunASR识别失败: {e}")
            return None
    
    def check_hotwords(self, text: str) -> List[str]:
        """检查文本中是否包含热词"""
        matched = []
        for hotword in self.hotwords:
            if hotword in text:
                matched.append(hotword)
        return matched


class WebSpeechFallback:
    """Web Speech API备选方案"""
    
    def __init__(self):
        self.hotwords = CULTURAL_TOURISM_HOTWORDS
    
    def recognize(self, audio_data: bytes) -> Optional[dict]:
        """
        Web Speech API备选方案
        注意：这需要在前端实现，后端只能提供接口
        """
        # 后端无法直接使用Web Speech API
        # 这里返回None，让前端处理
        return None
    
    def check_hotwords(self, text: str) -> List[str]:
        """检查文本中是否包含热词"""
        matched = []
        for hotword in self.hotwords:
            if hotword in text:
                matched.append(hotword)
        return matched


class VoiceRecognitionService:
    """语音识别服务"""
    
    def __init__(self):
        self.funasr = FunASRRecognizer()
        self.webspeech = WebSpeechFallback()
        self.use_funasr = True
        
        # 尝试加载FunASR模型
        try:
            self.funasr.load_model()
        except Exception as e:
            print(f"FunASR初始化失败，将使用Web Speech API作为备选: {e}")
            self.use_funasr = False
    
    async def recognize_audio(self, audio_data: bytes, use_hotwords: bool = True) -> AudioRecognitionResponse:
        """
        识别音频
        
        Args:
            audio_data: 音频数据（bytes）
            use_hotwords: 是否使用热词表
            
        Returns:
            语音识别响应
        """
        # 尝试使用FunASR
        if self.use_funasr:
            result = self.funasr.recognize(audio_data)
            if result and 'text' in result:
                text = result['text']
                hotwords_matched = []
                
                if use_hotwords:
                    hotwords_matched = self.funasr.check_hotwords(text)
                
                return AudioRecognitionResponse(
                    text=text,
                    confidence=result.get('confidence', 0.0),
                    engine='FunASR',
                    hotwords_matched=hotwords_matched
                )
        
        # FunASR失败，返回特定响应让前端使用Web Speech API
        return AudioRecognitionResponse(
            text="",
            confidence=0.0,
            engine='Web Speech API',
            hotwords_matched=[]
        )
    
    def get_hotwords(self) -> List[str]:
        """获取热词表"""
        return CULTURAL_TOURISM_HOTWORDS


# 全局语音识别服务实例
_voice_service: Optional[VoiceRecognitionService] = None


def get_voice_service() -> VoiceRecognitionService:
    """获取语音识别服务单例"""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceRecognitionService()
    return _voice_service