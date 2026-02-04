# import torch
# # 查看PyTorch版本
# print("PyTorch 版本：", torch.__version__)
# # 查看PyTorch绑定的CUDA运行时版本（虚拟环境内实际用的CUDA版本）
# print("虚拟环境内CUDA运行时版本：", torch.version.cuda)
# # 验证CUDA是否可用（虚拟环境内）
# print("CUDA是否可用：", torch.cuda.is_available())
# # 查看显卡型号（可选）
# if torch.cuda.is_available():
#     print("显卡型号：", torch.cuda.get_device_name(0))


import base64
from openai import OpenAI
import time

# 初始化客户端
client = OpenAI(
    base_url="http://10.136.234.255:8000/v1",
    api_key="dummy_key"
)

# 图片转base64编码（vLLM多模态接口要求）
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 测试图片路径（替换为你的图片路径，比如test.jpg）
image_path = "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图/4、食物细节/4.1 食物类型/3、甜品/2fa22518b0d836101cae279da625cc4.jpg"
image_base64 = image_to_base64(image_path)
start_time = time.time()
image_content = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
prompt = """
任务：判断图片的核心主体，仅从以下一级分类的六个分类中选择（可以多选，不新增）：
一级分类列表：人像、动物（宠物）、植物、风景、食物、建筑
如果图片不在这个六个主体中，请选择“其他”。
输出要求：仅返回分类名称（如“人像”“食物”），不添加任何额外解释。
"""
# 调用多模态对话（提问图片内容）
completion = client.chat.completions.create(
    model="/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_content}}
            ]
        }
    ],
    # temperature=0.7,
    # max_tokens=1024
)
end_time = time.time()
print(f"推理时间：{end_time - start_time:.2f} 秒")

# 打印结果
print("===== 多模态测试结果 =====")
print("完整返回内容：", completion)
print(completion.choices[0])
print(completion.choices[0].message)
print(completion.choices[0].message.content)


# export CUDA_VISIBLE_DEVICES=2,3
# ps -ef | grep 391954
# fuser -k /dev/nvidia3 强制释放GPU
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# vllm serve /workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8001 --limit-mm-per-prompt '{"image":2,"video":0}'  --gpu-memory-utilization 0.9  --max-model-len=65536  --data-parallel-size 2  --no-enable-prefix-caching
# nohup vllm serve /workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8000 --limit-mm-per-prompt '{"image":2,"video":0}'  --gpu-memory-utilization 0.7  --max-model-len=65536  --data-parallel-size 2  > vllm_service_8000.log 2>&1 &
# nohup vllm serve /workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8001 --limit-mm-per-prompt '{"image":2,"video":0}'  --gpu-memory-utilization 0.8  --max-model-len=65536  --data-parallel-size 2  > vllm_service_8001.log 2>&1 &

# vllm bench serve \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --model /workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct \
#   --dataset-name random \
#   --random-input-len 5000 \
#   --random-output-len 200 \
#   --num-prompts 300 



# ============ Serving Benchmark Result ============
# Successful requests:                     300       
# Failed requests:                         0         
# Benchmark duration (s):                  100.42    
# Total input tokens:                      1500000   
# Total generated tokens:                  60000     
# Request throughput (req/s):              2.99      
# Output token throughput (tok/s):         597.50    
# Peak output token throughput (tok/s):    3621.00   
# Peak concurrent requests:                300.00    
# Total Token throughput (tok/s):          15534.88  
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          39744.96  
# Median TTFT (ms):                        36530.55  
# P99 TTFT (ms):                           93628.96  
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          211.22    
# Median TPOT (ms):                        231.29    
# P99 TPOT (ms):                           278.15    
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           211.22    
# Median ITL (ms):                         274.10    
# P99 ITL (ms):                            284.10    
# ==================================================

