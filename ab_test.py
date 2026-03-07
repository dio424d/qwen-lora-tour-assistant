#!/usr/bin/env python3
"""
A/B 测试脚本 - 对比原始模型和微调模型

用于展示给面试官，证明微调效果

作者: AI Study Project
日期: 2026-03-03
"""

import time
import json
import threading
from src.model import QwenLoRAModel


class TimeoutException(Exception):
    """超时异常"""
    pass


def timeout_handler():
    """超时处理函数"""
    raise TimeoutException("生成响应超时")


def run_with_timeout(func, timeout=60):
    """
    带超时的函数执行
    
    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
    
    Returns:
        函数执行结果
    
    Raises:
        TimeoutException: 超时异常
    """
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutException("生成响应超时")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


def load_models():
    """
    加载原始模型和微调模型
    """
    print("=" * 60)
    print("正在加载模型...")
    print("=" * 60)
    
    # 原始模型（无LoRA）
    base_model = QwenLoRAModel()
    base_model._load_tokenizer()
    base_model._load_base_model()
    base_model.model.eval()
    print("✅ 原始模型加载完成")
    
    # 微调模型（带LoRA）
    lora_model = QwenLoRAModel()
    lora_model.load()
    print("✅ 微调模型加载完成")
    
    return base_model, lora_model


def prepare_test_cases():
    """
    准备测试用例
    """
    return [
        "推荐一个适合夏天旅游的地方",
        "云南旅游大概需要多少钱？"
    ]


def run_ab_test(base_model, lora_model, test_cases):
    """
    运行A/B测试
    """
    print("\n" + "=" * 60)
    print("开始A/B测试...")
    print("=" * 60)
    
    results = []
    total_time_base = 0
    total_time_lora = 0
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n📋 测试用例 {i}/{len(test_cases)}")
        print(f"问题: {question}")
        print("-" * 50)
        
        # 测试原始模型
        print("测试原始模型...")
        start_time = time.time()
        try:
            # 设置60秒超时
            base_response = run_with_timeout(
                lambda: base_model.generate(
                    [{"role": "user", "content": question}],
                    max_new_tokens=1024,  # 增加生成tokens数量，确保内容完整
                    temperature=0.7
                ),
                timeout=60
            )
            
            base_time = time.time() - start_time
            total_time_base += base_time
            
            print(f"原始模型: {base_response}")
            print(f"用时: {base_time:.2f}秒")
        except TimeoutException as e:
            print(f"❌ 原始模型生成超时: {e}")
            base_response = "生成超时"
            base_time = 60.0
        except Exception as e:
            print(f"❌ 原始模型生成错误: {e}")
            base_response = f"生成错误: {str(e)}"
            base_time = 0
        
        print("-" * 50)
        
        # 测试微调模型
        print("测试微调模型...")
        start_time = time.time()
        try:
            # 设置60秒超时
            lora_response = run_with_timeout(
                lambda: lora_model.generate(
                    [{"role": "user", "content": question}],
                    max_new_tokens=1024,  # 增加生成tokens数量，确保内容完整
                    temperature=0.7
                ),
                timeout=60
            )
            
            lora_time = time.time() - start_time
            total_time_lora += lora_time
            
            print(f"微调模型: {lora_response}")
            print(f"用时: {lora_time:.2f}秒")
        except TimeoutException as e:
            print(f"❌ 微调模型生成超时: {e}")
            lora_response = "生成超时"
            lora_time = 60.0
        except Exception as e:
            print(f"❌ 微调模型生成错误: {e}")
            lora_response = f"生成错误: {str(e)}"
            lora_time = 0
        
        results.append({
            "question": question,
            "base_response": base_response,
            "lora_response": lora_response,
            "base_time": base_time,
            "lora_time": lora_time
        })
    
    return results, total_time_base, total_time_lora


def generate_report(results, total_time_base, total_time_lora):
    """
    生成测试报告
    """
    print("\n" + "=" * 60)
    print("A/B测试报告")
    print("=" * 60)
    
    # 统计信息
    avg_time_base = total_time_base / len(results)
    avg_time_lora = total_time_lora / len(results)
    
    print(f"测试用例数量: {len(results)}")
    print(f"原始模型平均响应时间: {avg_time_base:.2f}秒")
    print(f"微调模型平均响应时间: {avg_time_lora:.2f}秒")
    
    # 保存结果到文件
    with open("ab_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_cases": results,
            "statistics": {
                "total_test_cases": len(results),
                "total_time_base": total_time_base,
                "total_time_lora": total_time_lora,
                "avg_time_base": avg_time_base,
                "avg_time_lora": avg_time_lora
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 测试结果已保存到 ab_test_results.json")
    
    # 生成HTML报告
    generate_html_report(results, total_time_base, total_time_lora)


def generate_html_report(results, total_time_base, total_time_lora):
    """
    生成HTML格式的测试报告
    """
    avg_time_base = total_time_base / len(results)
    avg_time_lora = total_time_lora / len(results)
    
    # 使用 format() 方法避免 CSS 大括号问题
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen LoRA 微调 A/B 测试报告</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: #4a5568;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .header p {
            color: #718096;
            font-size: 16px;
        }
        
        .stats {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
        }
        
        .stats h2 {
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .stat-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4a5568;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 14px;
            color: #718096;
        }
        
        .test-cases {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .test-cases h2 {
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 20px;
        }
        
        .test-case {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .test-case h3 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 16px;
        }
        
        .responses {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }
        
        .response {
            padding: 15px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .base-response {
            background: #f7fafc;
            border-left: 4px solid #3182ce;
        }
        
        .lora-response {
            background: #f0fff4;
            border-left: 4px solid #38a169;
        }
        
        .response h4 {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        
        .base-response h4 {
            color: #3182ce;
        }
        
        .lora-response h4 {
            color: #38a169;
        }
        
        .response-time {
            font-size: 12px;
            color: #718096;
            margin-top: 10px;
        }
        
        .footer {
            margin-top: 40px;
            text-align: center;
            color: white;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Qwen LoRA 微调 A/B 测试报告</h1>
            <p>对比原始模型与微调模型的性能和效果</p>
        </div>
        
        <div class="stats">
            <h2>📊 测试统计</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{test_cases_count}</div>
                    <div class="stat-label">测试用例</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_time_base:.2f}s</div>
                    <div class="stat-label">原始模型平均响应时间</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_time_lora:.2f}s</div>
                    <div class="stat-label">微调模型平均响应时间</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{current_time}</div>
                    <div class="stat-label">测试时间</div>
                </div>
            </div>
        </div>
        
        <div class="test-cases">
            <h2>🧪 测试用例对比</h2>
'''.format(
        test_cases_count=len(results),
        avg_time_base=avg_time_base,
        avg_time_lora=avg_time_lora,
        current_time=time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    for i, result in enumerate(results, 1):
        html_content += '''
            <div class="test-case">
                <h3>测试用例 {test_case_num}: {question}</h3>
                <div class="responses">
                    <div class="response base-response">
                        <h4>原始模型</h4>
                        <p>{base_response}</p>
                        <div class="response-time">响应时间: {base_time:.2f}秒</div>
                    </div>
                    <div class="response lora-response">
                        <h4>微调模型</h4>
                        <p>{lora_response}</p>
                        <div class="response-time">响应时间: {lora_time:.2f}秒</div>
                    </div>
                </div>
            </div>
'''.format(
            test_case_num=i,
            question=result['question'],
            base_response=result['base_response'],
            base_time=result['base_time'],
            lora_response=result['lora_response'],
            lora_time=result['lora_time']
        )
    
    html_content += '''
        </div>
        
        <div class="footer">
            <p>测试报告生成时间: {current_time}</p>
            <p>Qwen LoRA 旅游咨询助手项目</p>
        </div>
    </div>
</body>
</html>
'''.format(
        current_time=time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    with open("ab_test_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ HTML报告已生成: ab_test_report.html")


def main():
    """
    主函数
    """
    try:
        # 加载模型
        base_model, lora_model = load_models()
        
        # 准备测试用例
        test_cases = prepare_test_cases()
        
        # 运行A/B测试
        results, total_time_base, total_time_lora = run_ab_test(base_model, lora_model, test_cases)
        
        # 生成报告
        generate_report(results, total_time_base, total_time_lora)
        
        print("\n" + "=" * 60)
        print("🎉 A/B测试完成！")
        print("=" * 60)
        print("📁 生成的文件:")
        print("   - ab_test_results.json (测试结果数据)")
        print("   - ab_test_report.html (可视化报告)")
        print("\n🔍 查看测试报告:")
        print("   1. 打开 ab_test_report.html 在浏览器中查看")
        print("   2. 或查看 ab_test_results.json 原始数据")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()