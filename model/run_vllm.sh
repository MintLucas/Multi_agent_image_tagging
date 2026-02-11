#!/bin/bash

# ================= 配置区域 =================
# 请确保这是您真实的 vLLM 路径 (使用 which vllm 确认)
VLLM_CMD="/workspace/work/moniforge3/envs/zp_vllm/bin/vllm"
# 【核心优化】：强制 Python 不缓存输出，日志实时可见
export PYTHONUNBUFFERED=1
# 模型路径
MODEL_PATH="/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen3-VL-4B-Instruct"
echo "正在启动 vLLM 服务..."
echo "vLLM路径: $VLLM_CMD"

# 检查 vLLM 文件是否存在
if [ ! -f "$VLLM_CMD" ]; then
    echo "错误: 找不到 vLLM 执行文件: $VLLM_CMD"
    exit 1
fi

# -------------------------------------------------------
# 启动服务 1：占用 GPU 0,1 | 端口 8000
# -------------------------------------------------------
# echo ">>> 启动服务 1 (GPU 0,1 -> Port 8000)..."
# CUDA_VISIBLE_DEVICES=0,1 nohup "$VLLM_CMD" serve "$MODEL_PATH" \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --limit-mm-per-prompt '{"image":2,"video":0}' \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 65536 \
#     --trust-remote-code \
#     --tensor-parallel-size 2 \
#     > vllm_8000.log 2>&1 &

# sleep 2
# VLLM_CMD="/workspace/work/moniforge3/envs/lyf50_vllm/bin/vllm"
# -------------------------------------------------------
# 启动服务 2：占用 GPU 2,3 | 端口 8001
# -------------------------------------------------------
# MODEL_PATH="/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-7B-Instruct"
echo "正在启动 vLLM 服务..."
echo "vLLM路径: $VLLM_CMD"
echo ">>> 启动服务 2 (GPU 2,3 -> Port 8001)..."
CUDA_VISIBLE_DEVICES=2,3 nohup "$VLLM_CMD" serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8001 \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --gpu-memory-utilization 0.8 \
    --max-model-len 65536 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    > vllm_8001.log 2>&1 &

echo "服务启动命令已发送。"
echo "请查看日志: tail -f vllm_8000.log"