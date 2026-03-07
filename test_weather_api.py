#!/usr/bin/env python3
"""
测试天气查询 API
"""

import requests
import json

base_url = "http://localhost:8000"

print("=" * 60)
print("天气查询 API 测试")
print("=" * 60)

test_cases = [
    "北京天气",
    "上海的天气",
    "广州天气怎么样",
    "三亚的天气",
]

for test_input in test_cases:
    print(f"\n测试输入: '{test_input}'")
    print("-" * 60)
    
    payload = {
        "model": "qwen-lora",
        "messages": [
            {"role": "user", "content": test_input}
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ 响应成功")
            print(f"回复内容:\n{content}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
    
    except Exception as e:
        print(f"❌ 请求异常: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)