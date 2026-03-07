#!/usr/bin/env python3
"""
测试天气查询参数提取
"""

import re

test_cases = [
    "北京天气",
    "北京的天气",
    "北京天气怎么样",
    "上海的天气怎么样",
    "广州天气",
    "三亚天气",
    "深圳的天气",
]

print("=" * 60)
print("天气查询参数提取测试")
print("=" * 60)

for test_input in test_cases:
    print(f"\n输入: '{test_input}'")
    
    # 格式1: (.*?)的天气
    city_match = re.search(r'(.*?)的天气', test_input)
    if city_match:
        city = city_match.group(1).strip()
        print(f"  格式1匹配: '{city}'")
    else:
        print(f"  格式1未匹配")
    
    # 格式2: (.*?)天气
    city_match = re.search(r'(.*?)天气', test_input)
    if city_match:
        city = city_match.group(1).strip()
        print(f"  格式2匹配: '{city}'")
    else:
        print(f"  格式2未匹配")
    
    # 格式3: ([^\s天气]+)
    city_match = re.search(r'([^\s天气]+)', test_input)
    if city_match:
        city = city_match.group(1).strip()
        print(f"  格式3匹配: '{city}'")
    else:
        print(f"  格式3未匹配")