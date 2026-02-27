
"""
对话记忆功能使用示例

演示如何使用带对话记忆的 API
"""

import requests
import json

BASE_URL = "http://localhost:8000/v1"


def test_chat_with_memory():
    """测试带对话记忆的聊天"""
    
    session_id = "test-session-001"
    
    print("=" * 60)
    print("对话记忆功能测试")
    print("=" * 60)
    
    messages = [
        {
            "role": "user",
            "content": "我想去云南旅游，有什么推荐吗？"
        }
    ]
    
    print(f"\n用户: {messages[0]['content']}")
    
    payload = {
        "model": "qwen-lora",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256,
        "session_id": session_id
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload
    )
    
    result = response.json()
    assistant_reply = result["choices"][0]["message"]["content"]
    print(f"助手: {assistant_reply}\n")
    
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    messages.append({
        "role": "user",
        "content": "那大概需要多少钱？"
    })
    
    print(f"用户: {messages[2]['content']}")
    
    payload["messages"] = messages
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload
    )
    
    result = response.json()
    assistant_reply = result["choices"][0]["message"]["content"]
    print(f"助手: {assistant_reply}\n")
    
    print("=" * 60)
    print("查看对话历史")
    print("=" * 60)
    
    history_response = requests.get(
        f"{BASE_URL}/sessions/{session_id}/history"
    )
    
    history = history_response.json()
    print(f"\n会话ID: {history['session_id']}")
    print(f"消息数量: {len(history['messages'])}\n")
    
    for i, msg in enumerate(history['messages'], 1):
        print(f"{i}. [{msg['role']}]: {msg['content'][:50]}...")
    
    print("\n" + "=" * 60)
    print("查看所有会话")
    print("=" * 60)
    
    sessions_response = requests.get(f"{BASE_URL}/sessions")
    sessions = sessions_response.json()
    
    print(f"\n总会话数: {len(sessions['sessions'])}\n")
    for session in sessions['sessions']:
        print(f"会话ID: {session['session_id']}")
        print(f"  总消息: {session['total_messages']}")
        print(f"  用户消息: {session['user_messages']}")
        print(f"  助手消息: {session['assistant_messages']}")
        print()


def test_new_session():
    """测试新会话（无记忆）"""
    
    print("=" * 60)
    print("新会话测试（无对话记忆）")
    print("=" * 60)
    
    payload = {
        "model": "qwen-lora",
        "messages": [
            {
                "role": "user",
                "content": "你好，介绍一下你自己"
            }
        ],
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload
    )
    
    result = response.json()
    assistant_reply = result["choices"][0]["message"]["content"]
    print(f"\n助手: {assistant_reply}\n")
    
    print("注意: 这次请求没有提供 session_id，所以不会保存对话历史")


if __name__ == "__main__":
    try:
        test_chat_with_memory()
        print("\n\n")
        test_new_session()
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")
        print("请确保先运行: python app.py")
    except Exception as e:
        print(f"发生错误: {e}")

