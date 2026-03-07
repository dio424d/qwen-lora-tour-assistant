#!/usr/bin/env python3
"""
详细测试高德地图 API 响应
"""

import os
import sys
import json

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.amap import AmapAPI
from src.config import settings

print("=" * 60)
print("高德地图 API 详细响应测试")
print("=" * 60)

# 检查 API 密钥
api_key = settings.amap_api_key
print(f"\nAPI 密钥: {api_key[:10]}..." if api_key else "未配置")

if not api_key:
    print("❌ 错误：未配置高德地图 API 密钥")
    sys.exit(1)

# 创建 API 客户端
amap = AmapAPI(api_key)

# 测试天气查询 - 打印完整响应
print("\n" + "=" * 60)
print("天气查询完整响应")
print("=" * 60)

result = amap.weather("北京")
print(json.dumps(result, ensure_ascii=False, indent=2))

# 测试不同参数
print("\n" + "=" * 60)
print("测试不同参数")
print("=" * 60)

# 测试 base 模式（不传入 extensions 参数）
import requests
url = "https://restapi.amap.com/v3/weather/weatherInfo"
params = {
    "key": api_key,
    "city": "北京",
    "output": "json"
}

print("\n基础天气查询（无 extensions 参数）:")
response = requests.get(url, params=params)
result_base = response.json()
print(json.dumps(result_base, ensure_ascii=False, indent=2))

# 测试 all 模式
params["extensions"] = "all"
print("\n完整天气查询（extensions=all）:")
response = requests.get(url, params=params)
result_all = response.json()
print(json.dumps(result_all, ensure_ascii=False, indent=2))