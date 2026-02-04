import datetime
import time
import os
from vllm import LLM, SamplingParams

base_path = '/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen'
# --- 配置区 ---
model_path = f"{base_path}/Qwen2.5-VL-3B-Instruct"
GPU_COUNT = 4          # 你的机器总卡数
TARGET_UTIL = 0.85     # 显存占用目标 (85% 比较保险，确保达标)
DAY_START = 4          # 白天开始时间 (8:00)
NIGHT_START = 23       # 晚上开始时间 (22:00)

def is_daytime():
    now = datetime.datetime.now().hour
    return DAY_START <= now < NIGHT_START

def run_keep_alive():
    print(f"正在初始化 vLLM，目标显存占用: {TARGET_UTIL * 100}%...")
    
    # 初始化 vLLM
    # tensor_parallel_size=4 会把模型分布在4张卡上，同时占用4张卡的显存
    llm = LLM(
        model=model_path, 
        runner="generate",
        gpu_memory_utilization=TARGET_UTIL,
        tensor_parallel_size=GPU_COUNT, 
        enforce_eager=True,
        mm_processor_kwargs={
            "use_fast": True,
            "disable_video": True
        },
        max_model_len=10000, # 适当调小，防止初始分配过大失败
    )

    sampling_params = SamplingParams(max_tokens=10)
    # 构造一个简单的 prompt
    dummy_prompt = "Keep this GPU busy with a short response."

    print(f"脚本已启动。当前模式: {'白天(计算+显存)' if is_daytime() else '晚上(仅显存)'}")

    try:
        while True:
            current_hour = datetime.datetime.now().hour
            
            if is_daytime():
                # 白天模式：执行简单的推理来拉高计算利用率
                # print(f"[{datetime.datetime.now()}] 正在执行推理以保持计算利用率...")
                llm.generate(dummy_prompt, sampling_params, use_tqdm=False)
                # 稍微 sleep 一下，避免 100% CPU 占用，同时保持 GPU 活跃
                time.sleep(0.5) 
            else:
                # 晚上模式：只进入睡眠状态
                # vLLM 进程存在，显存就会被预分配（锁定），但没有推理产生计算
                if current_hour == NIGHT_START:
                    print(f"[{datetime.datetime.now()}] 进入晚上模式，暂停计算任务...")
                time.sleep(60) 

    except KeyboardInterrupt:
        print("停止保活脚本")

if __name__ == "__main__":
    run_keep_alive()