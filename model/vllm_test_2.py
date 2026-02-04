import base64
import time
import os
import random
import psutil
import pynvml
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median, quantiles

# ===================== 全局配置（按需修改） =====================
# VLLM服务配置
VLLM_BASE_URL = "http://10.136.234.255:8001/v1"
API_KEY = "dummy_key"
MODEL_PATH = "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct"

# 测试素材配置（仅3张图片的文件夹路径）
IMAGE_FOLDER = "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图"  # 替换为你的3张图片所在文件夹
PROMPT_TEMPLATES = [  # 不同的问题模板（随机选）
    """
    任务：判断图片的核心主体，仅从以下一级分类的六个分类中选择1个（必须选，不新增）：
    一级分类列表：人像、动物（宠物）、植物、风景、食物、建筑
    如果图片不在这个六个主体中，请选择“其他”。
    输出要求：仅返回分类名称（如“人像”“食物”），不添加任何额外解释。
    """,
    """
    任务：描述图片的核心场景，仅返回1个关键词（如“室内人像”“户外风景”“宠物特写”），不添加额外解释。
    """,
    """
    任务：判断图片的主色调，仅从以下选项中选1个：红色、蓝色、绿色、黄色、黑色、白色、彩色，不添加额外解释。
    """,
    """
    任务：判断图片是否包含人物，仅返回“是”或“否”，不添加任何额外解释。
    """,
    """
    任务：判断图片的拍摄场景，仅从以下选项中选1个：室内、户外、水下、空中，不添加额外解释。
    """
]

# 压测配置（适配3张图片的梯度并发数，避免过度复用）
CONCURRENT_NUM_LIST = [50, 60, 100, 150, 200]  # 最高20并发（3张图各复用6-7次）
MAX_WORKERS = 200  # 线程池最大线程数（≥最大并发数）
REQUEST_TIMEOUT = 30  # 单请求超时时间（秒）

# ===================== 工具函数 =====================
# 初始化GPU监控（pynvml）
def init_gpu_monitor():
    try:
        pynvml.nvmlInit()
        return pynvml
    except Exception as e:
        print(f"GPU监控初始化失败：{e}，将跳过GPU状态打印")
        return None

# 获取GPU实时状态（显存/利用率）
def get_gpu_status(nvml):
    if not nvml:
        return "GPU监控未启用"
    status = []
    device_count = nvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        status.append(
            f"GPU{i}: 显存占用 {mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB, 利用率 {util.gpu}%"
        )
    return "\n".join(status)

# 图片转base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 获取文件夹内所有有效图片路径
def get_all_image_paths(folder):
    image_ext = [".jpg", ".jpeg", ".png", ".bmp"]
    paths = []

    # image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, file))
    # for file in os.listdir(folder):
    #     file_path = os.path.join(folder, file)
    #     if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_ext:
    #         paths.append(file_path)
    if not paths:
        raise ValueError(f"图片文件夹{folder}中未找到有效图片")
    print(f"检测到测试图片：{[os.path.basename(p) for p in paths]}（共{len(paths)}张），将循环复用")
    return paths

# 单请求函数（随机选问题+循环复用图片）
def send_request(request_id, client, image_path, prompt_templates):
    start_time = time.time()
    # 随机选一个问题模板
    prompt = random.choice(prompt_templates)
    try:
        # 加载图片（每个请求独立加载，复用图片路径）
        image_base64 = image_to_base64(image_path)
        # 发送请求
        completion = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            timeout=REQUEST_TIMEOUT
        )
        cost_time = time.time() - start_time
        return {
            "request_id": request_id,
            "image_path": os.path.basename(image_path),
            "prompt": prompt.strip()[:50] + "..." if len(prompt.strip())>50 else prompt.strip(),
            "cost_time": cost_time,
            "success": True,
            "result": completion.choices[0].message.content.strip(),
            "error": None
        }
    except Exception as e:
        cost_time = time.time() - start_time
        return {
            "request_id": request_id,
            "image_path": os.path.basename(image_path),
            "prompt": prompt.strip()[:50] + "..." if len(prompt.strip())>50 else prompt.strip(),
            "cost_time": cost_time,
            "success": False,
            "result": None,
            "error": str(e)[:100] + "..." if len(str(e))>100 else str(e)
        }

# 统计压测结果（含95分位响应时间）
def stat_results(results, total_time, concurrent_num):
    success_num = sum(1 for r in results if r["success"])
    fail_num = concurrent_num - success_num
    success_rate = success_num / concurrent_num * 100 if concurrent_num > 0 else 0
    qps = concurrent_num / total_time if total_time > 0 else 0
    
    # 响应时间统计
    cost_times = [r["cost_time"] for r in results if r["success"]]
    avg_cost = sum(cost_times) / len(cost_times) if cost_times else 0
    p95_cost = quantiles(cost_times, n=20)[18] if len(cost_times)>=20 else (max(cost_times) if cost_times else 0)
    median_cost = median(cost_times) if cost_times else 0

    return {
        "concurrent_num": concurrent_num,
        "total_time": total_time,
        "success_num": success_num,
        "fail_num": fail_num,
        "success_rate": success_rate,
        "qps": qps,
        "avg_cost": avg_cost,
        "median_cost": median_cost,
        "p95_cost": p95_cost
    }

if __name__ == "__main__":
    # 1. 初始化资源
    nvml = init_gpu_monitor()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)
    all_image_paths = get_all_image_paths(IMAGE_FOLDER)
    final_report = []  # 存储所有并发数的测试结果

    print("\n===== VLLM并发测试（随机图片+不同问题）=====")
    print(f"总图片数量：{len(all_image_paths)}张")
    print(f"问题模板数量：{len(PROMPT_TEMPLATES)}个")
    print(f"梯度并发数：{CONCURRENT_NUM_LIST}")
    print("="*50 + "\n")

    # 2. 梯度压测（逐个并发数测试）
    for concurrent_num in CONCURRENT_NUM_LIST:
        # 修改点1：随机抽取图片，增加多样性
        import random
        
        if concurrent_num <= len(all_image_paths):
            # 如果并发数小于等于图片总数，随机选择不重复的图片
            test_image_paths = random.sample(all_image_paths, concurrent_num)
        else:
            # 如果并发数大于图片总数，先随机选择所有图片，再补充随机图片
            test_image_paths = random.sample(all_image_paths, len(all_image_paths))
            # 补充剩余数量（允许重复，但尽量降低重复率）
            remaining = concurrent_num - len(all_image_paths)
            for i in range(remaining):
                # 随机选择图片，可以重复但打乱顺序
                test_image_paths.append(random.choice(all_image_paths))
            # 打乱顺序，避免相同图片连续出现
            random.shuffle(test_image_paths)
        
        # 修改点2：统计图片使用情况
        image_usage = {}
        for img_path in test_image_paths:
            img_name = os.path.basename(img_path)
            image_usage[img_name] = image_usage.get(img_name, 0) + 1
        
        # 找出最常用和最不常用的图片
        if image_usage:
            most_used = max(image_usage.items(), key=lambda x: x[1])
            least_used = min(image_usage.items(), key=lambda x: x[1])
            unique_images = len(image_usage)
        else:
            most_used = ("None", 0)
            least_used = ("None", 0)
            unique_images = 0
        
        print(f"开始测试并发数：{concurrent_num}")
        print(f"图片分配：使用{unique_images}张不同图片（共{len(all_image_paths)}张）")
        print(f"图片复用情况：最常用图片'{most_used[0]}'使用{most_used[1]}次，最少用图片'{least_used[0]}'使用{least_used[1]}次")
        print(f"当前GPU状态：\n{get_gpu_status(nvml)}")
        
        # 修改点3：显示前10个请求的图片分配（抽样查看）
        if concurrent_num <= 50:
            sample_size = min(10, concurrent_num)
            sample_images = [os.path.basename(test_image_paths[i]) for i in range(sample_size)]
            print(f"前{sample_size}个请求图片样本：{sample_images}")
        else:
            # 抽样显示
            sample_indices = random.sample(range(concurrent_num), min(10, concurrent_num))
            sample_images = [os.path.basename(test_image_paths[i]) for i in sample_indices]
            print(f"随机抽样10个请求图片：{sample_images}")
        
        # 执行并发请求
        start_total_time = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(send_request, i, client, test_image_paths[i], PROMPT_TEMPLATES)
                for i in range(concurrent_num)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        
        # 统计结果
        total_time = time.time() - start_total_time
        stat = stat_results(results, total_time, concurrent_num)
        final_report.append(stat)

        # 修改点4：添加图片多样性指标到统计结果
        stat["unique_images"] = unique_images
        stat["max_image_reuse"] = most_used[1]
        
        # 打印当前并发数的结果
        print(f"\n===== 并发数{concurrent_num}测试结果 =====")
        print(f"总耗时：{total_time:.2f} 秒")
        print(f"成功数：{stat['success_num']} | 失败数：{stat['fail_num']} | 成功率：{stat['success_rate']:.2f}%")
        print(f"QPS（每秒处理请求数）：{stat['qps']:.2f}")
        print(f"平均响应时间：{stat['avg_cost']:.2f} 秒")
        print(f"中位数响应时间：{stat['median_cost']:.2f} 秒")
        print(f"95分位响应时间：{stat['p95_cost']:.2f} 秒")
        print(f"图片多样性：使用{unique_images}张不同图片，最大复用{most_used[1]}次")
        print(f"测试后GPU状态：\n{get_gpu_status(nvml)}")

        # 打印失败请求详情（如有）
        if stat["fail_num"] > 0:
            print("\n⚠️ 失败请求详情：")
            fail_results = [r for r in results if not r["success"]]
            for r in fail_results[:5]:  # 仅打印前5条
                print(f"请求ID{r['request_id']} | 图片{r['image_path']} | 错误：{r['error']}")
        
        # 修改点5：显示一些成功请求的结果示例
        if stat["success_num"] > 0:
            print("\n✅ 成功请求示例（前3个）：")
            success_results = [r for r in results if r["success"]]
            for i, r in enumerate(success_results[:3]):
                print(f"  请求{r['request_id']}: 图片'{r['image_path']}' -> 回答: {r['result'][:50]}...")
        
        print("="*50 + "\n")
        time.sleep(2)  # 测试间隔，让GPU稍作休息

    # 3. 打印最终汇总报告（增加图片多样性列）
    print("===== 最终汇总报告（所有并发数）=====")
    print(f"{'并发数':<6} {'成功率(%)':<10} {'QPS':<8} {'平均响应时间(s)':<15} {'95分位响应时间(s)':<18} {'唯一图片数':<12} {'最大复用':<10}")
    print("-"*85)
    for stat in final_report:
        print(
            f"{stat['concurrent_num']:<6} "
            f"{stat['success_rate']:<10.2f} "
            f"{stat['qps']:<8.2f} "
            f"{stat['avg_cost']:<15.2f} "
            f"{stat['p95_cost']:<18.2f} "
            f"{stat.get('unique_images', 0):<12} "
            f"{stat.get('max_image_reuse', 0):<10}"
        )

    # 4. 输出极限并发数（成功率≥95%的最大并发数）
    valid_stats = [s for s in final_report if s["success_rate"] >= 95]
    if valid_stats:
        max_valid_concurrent = max(valid_stats, key=lambda x: x["concurrent_num"])
        print(f"\n✅ 极限并发数（成功率≥95%）：{max_valid_concurrent['concurrent_num']}")
        print(f"该并发数下QPS：{max_valid_concurrent['qps']:.2f}")
        print(f"95分位响应时间：{max_valid_concurrent['p95_cost']:.2f}秒")
        print(f"图片多样性：{max_valid_concurrent.get('unique_images', 0)}张不同图片")
    else:
        print("\n❌ 所有测试并发数的成功率均<95%，请检查VLLM配置或降低并发数")
    
    # 5. 分析性能趋势
    print("\n===== 性能趋势分析 =====")
    if len(final_report) >= 2:
        first_stat = final_report[0]
        last_stat = final_report[-1]
        
        qps_growth = (last_stat['qps'] / first_stat['qps'] - 1) * 100 if first_stat['qps'] > 0 else 0
        avg_latency_growth = (last_stat['avg_cost'] / first_stat['avg_cost'] - 1) * 100 if first_stat['avg_cost'] > 0 else 0
        
        print(f"从{first_stat['concurrent_num']}到{last_stat['concurrent_num']}并发：")
        print(f"  QPS变化：{first_stat['qps']:.2f} → {last_stat['qps']:.2f} ({qps_growth:+.1f}%)")
        print(f"  平均延迟变化：{first_stat['avg_cost']:.2f}s → {last_stat['avg_cost']:.2f}s ({avg_latency_growth:+.1f}%)")
        
        if qps_growth < 20 and last_stat['concurrent_num'] > first_stat['concurrent_num'] * 2:
            print("  ⚠️ 警告：并发数翻倍但QPS增长不足20%，可能存在性能瓶颈")
        
        # 找到QPS开始下降的转折点
        max_qps_stat = max(final_report, key=lambda x: x['qps'])
        if max_qps_stat['concurrent_num'] < last_stat['concurrent_num']:
            print(f"  📉 QPS在{max_qps_stat['concurrent_num']}并发时达到峰值{max_qps_stat['qps']:.2f}，之后开始下降")

    # 清理GPU监控资源
    if nvml:
        nvml.nvmlShutdown()

# # ===================== 主压测流程（核心：循环复用图片） =====================
# if __name__ == "__main__":
#     # 1. 初始化资源
#     nvml = init_gpu_monitor()
#     client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)
#     all_image_paths = get_all_image_paths(IMAGE_FOLDER)
#     final_report = []  # 存储所有并发数的测试结果

#     print("\n===== VLLM并发测试=====")
#     print(f"测试图片数量：{len(all_image_paths)}")
#     print(f"问题模板数量：{len(PROMPT_TEMPLATES)}")
#     print(f"梯度并发数：{CONCURRENT_NUM_LIST}")
#     print("="*50 + "\n")

#     # 2. 梯度压测（逐个并发数测试）
#     for concurrent_num in CONCURRENT_NUM_LIST:
#         # 核心修改：循环复用图片（取模实现）
#         test_image_paths = []
#         for i in range(concurrent_num):
#             test_image_paths.append(all_image_paths[i % len(all_image_paths)])
        
#         print(f"开始测试并发数：{concurrent_num}")
#         print(f"本次测试图片分配（循环复用）：{[os.path.basename(p) for p in test_image_paths]}")
#         print(f"当前GPU状态：\n{get_gpu_status(nvml)}")
        
#         # 执行并发请求
#         start_total_time = time.time()
#         results = []
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             futures = [
#                 executor.submit(send_request, i, client, test_image_paths[i], PROMPT_TEMPLATES)
#                 for i in range(concurrent_num)
#             ]
#             for future in as_completed(futures):
#                 results.append(future.result())
        
#         # 统计结果
#         total_time = time.time() - start_total_time
#         stat = stat_results(results, total_time, concurrent_num)
#         final_report.append(stat)

#         # 打印当前并发数的结果
#         print(f"\n===== 并发数{concurrent_num}测试结果 =====")
#         print(f"总耗时：{total_time:.2f} 秒")
#         print(f"成功数：{stat['success_num']} | 失败数：{stat['fail_num']} | 成功率：{stat['success_rate']:.2f}%")
#         print(f"QPS（每秒处理请求数）：{stat['qps']:.2f}")
#         print(f"平均响应时间：{stat['avg_cost']:.2f} 秒")
#         print(f"中位数响应时间：{stat['median_cost']:.2f} 秒")
#         print(f"95分位响应时间：{stat['p95_cost']:.2f} 秒")
#         print(f"测试后GPU状态：\n{get_gpu_status(nvml)}")

#         # 打印失败请求详情（如有）
#         if stat["fail_num"] > 0:
#             print("\n⚠️ 失败请求详情：")
#             fail_results = [r for r in results if not r["success"]]
#             for r in fail_results[:5]:  # 仅打印前5条
#                 print(f"请求ID{r['request_id']} | 图片{r['image_path']} | 错误：{r['error']}")
#         print("="*50 + "\n")
#         time.sleep(2)  # 测试间隔，让GPU稍作休息

#     # 3. 打印最终汇总报告
#     print("===== 最终汇总报告（所有并发数）=====")
#     print(f"{'并发数':<6} {'成功率(%)':<10} {'QPS':<8} {'平均响应时间(s)':<15} {'95分位响应时间(s)':<18}")
#     print("-"*60)
#     for stat in final_report:
#         print(
#             f"{stat['concurrent_num']:<6} "
#             f"{stat['success_rate']:<10.2f} "
#             f"{stat['qps']:<8.2f} "
#             f"{stat['avg_cost']:<15.2f} "
#             f"{stat['p95_cost']:<18.2f}"
#         )

#     # 4. 输出极限并发数（成功率≥95%的最大并发数）
#     valid_stats = [s for s in final_report if s["success_rate"] >= 95]
#     if valid_stats:
#         max_valid_concurrent = max(valid_stats, key=lambda x: x["concurrent_num"])
#         print(f"\n✅ 极限并发数（成功率≥95%）：{max_valid_concurrent['concurrent_num']}")
#         print(f"该并发数下QPS：{max_valid_concurrent['qps']:.2f}，95分位响应时间：{max_valid_concurrent['p95_cost']:.2f}秒")
#     else:
#         print("\n❌ 所有测试并发数的成功率均<95%，请检查VLLM配置或降低并发数")

#     # 清理GPU监控资源
#     if nvml:
#         nvml.nvmlShutdown()