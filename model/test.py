from vllm import LLM
model_path="./Qwen/Qwen2.5-VL-3B-Instruct"

# For generative models (runner=generate) only
llm = LLM(model=model_path, 
          runner="generate",
          gpu_memory_utilization=0.55,
          enforce_eager=True,
          mm_processor_kwargs={
            "use_fast": True,            # 启用快速多模态处理器
            "disable_video": True}, 
          max_model_len=100000,       # 关闭视频编码器（核心：节省~1G显存，你仅用图像）
          )  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)