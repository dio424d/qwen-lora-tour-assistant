import requests
import time

# 测试服务是否正常运行
print("测试服务启动状态...")
time.sleep(2)

try:
    response = requests.get("http://localhost:8000/health")
    print(f"服务状态: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"服务未启动: {e}")

print("测试完成")