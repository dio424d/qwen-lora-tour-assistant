
import requests
import json

BASE_URL = "http://localhost:8000/v1"

def test_health():
    """测试服务健康状态"""
    response = requests.get("http://localhost:8000/health")
    print("健康检查:", response.json())

def test_list_models():
    """获取模型列表"""
    response = requests.get(f"{BASE_URL}/models")
    print("模型列表:", json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_chat_completion():
    """测试聊天接口"""
    payload = {
        "model": "qwen-lora",
        "messages": [
            {
                "role": "user",
                "content": "推荐一个适合夏天旅游的地方"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload
    )
    
    result = response.json()
    print("聊天响应:", json.dumps(result, indent=2, ensure_ascii=False))
    
    if "choices" in result:
        print("\n助手回复:", result["choices"][0]["message"]["content"])

if __name__ == "__main__":
    print("=" * 50)
    print("Qwen LoRA API 使用示例")
    print("=" * 50)
    print()
    
    try:
        test_health()
        print()
        
        test_list_models()
        print()
        
        test_chat_completion()
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")
        print("请确保先运行: python simple_api.py")
    except Exception as e:
        print(f"发生错误: {e}")

