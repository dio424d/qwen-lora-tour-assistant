#!/usr/bin/env python3
"""
测试高德地图 API
"""

import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.amap import AmapAPI
from src.config import settings

print("=" * 60)
print("高德地图 API 测试")
print("=" * 60)

# 检查 API 密钥
api_key = settings.amap_api_key
print(f"\nAPI 密钥: {'已配置' if api_key else '未配置'}")

if not api_key:
    print("❌ 错误：未配置高德地图 API 密钥")
    print("请编辑 .env 文件，添加 QWEN_AMAP_API_KEY=您的API密钥")
    sys.exit(1)

# 创建 API 客户端
amap = AmapAPI(api_key)

# 测试 1: 天气查询
print("\n" + "=" * 60)
print("测试 1: 天气查询")
print("=" * 60)

test_cities = ["北京", "上海", "广州", "三亚"]
for city in test_cities:
    print(f"\n查询城市: {city}")
    result = amap.weather(city)
    print(f"API 响应状态: {result.get('status', 'N/A')}")
    print(f"API 响应信息: {result.get('info', 'N/A')}")
    
    if result.get("status") == "1":
        lives = result.get("lives", [])
        if lives:
            weather = lives[0]
            print(f"✅ 成功获取天气信息")
            print(f"   城市: {weather.get('city', 'N/A')}")
            print(f"   温度: {weather.get('temperature', 'N/A')}°C")
            print(f"   天气: {weather.get('weather', 'N/A')}")
        else:
            print(f"⚠️  API 返回成功但没有天气数据")
    else:
        print(f"❌ 获取天气信息失败")
        print(f"   错误信息: {result.get('info', '未知错误')}")

# 测试 2: 酒店搜索
print("\n" + "=" * 60)
print("测试 2: 酒店搜索")
print("=" * 60)

result = amap.search_text("酒店", "北京")
print(f"API 响应状态: {result.get('status', 'N/A')}")

if result.get("status") == "1":
    count = int(result.get("count", "0"))
    print(f"找到 {count} 个结果")
    
    pois = result.get("pois", [])
    if pois:
        print(f"\n前 3 个酒店:")
        for i, poi in enumerate(pois[:3], 1):
            print(f"{i}. {poi.get('name', 'N/A')}")
            print(f"   地址: {poi.get('address', 'N/A')}")
    else:
        print("⚠️ 没有找到酒店信息")
else:
    print(f"❌ 搜索失败: {result.get('info', '未知错误')}")

# 测试 3: 路线规划
print("\n" + "=" * 60)
print("测试 3: 路线规划")
print("=" * 60)

# 先获取地理编码
origin_geo = amap.geocode("北京市天安门")
dest_geo = amap.geocode("北京市故宫")

if origin_geo.get("status") == "1" and dest_geo.get("status") == "1":
    origin_loc = origin_geo["geocodes"][0].get("location", "")
    dest_loc = dest_geo["geocodes"][0].get("location", "")
    
    print(f"起点坐标: {origin_loc}")
    print(f"终点坐标: {dest_loc}")
    
    route_result = amap.direction_driving(origin_loc, dest_loc)
    
    if route_result.get("status") == "1":
        paths = route_result.get("route", {}).get("paths", [])
        if paths:
            path = paths[0]
            print(f"✅ 路线规划成功")
            print(f"   距离: {path.get('distance', 'N/A')} 米")
            print(f"   预计时间: {path.get('duration', 'N/A')} 秒")
        else:
            print("⚠️ 没有找到路线")
    else:
        print(f"❌ 路线规划失败: {route_result.get('info', '未知错误')}")
else:
    print("❌ 地理编码失败")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)