#!/bin/bash

# ================= 配置区域 =================
ENV_NAME="zp_vllm_new"  # 目标环境名称
PYTHON_VERSION="3.11" # 推荐 3.10 或 3.11
# ===========================================

set -e # 遇到错误立即停止

echo "========================================================"
echo "   🔨 开始构建支持 Qwen2.5-VL 的纯净环境: $ENV_NAME"
echo "========================================================"

# 1. 初始化 Conda
eval "$(conda shell.bash hook)"

# 2. 清理旧环境 (如果有)
if conda info --envs | grep -q "$ENV_NAME"; then
    echo ">>> 🗑️  检测到旧环境 '$ENV_NAME'，正在删除以确保纯净..."
    conda deactivate 2>/dev/null || true
    conda env remove -n $ENV_NAME -y
    echo ">>> 旧环境已清理。"
fi

# 3. 创建全新环境
echo ">>> 🆕 正在创建新环境 (Python $PYTHON_VERSION)..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 4. 激活环境
echo ">>> 🔌 激活环境..."
conda activate $ENV_NAME

# 5. 安装核心依赖 (强制最新版)
echo ">>> 📦 正在安装核心依赖 (这可能需要几分钟，取决于网速)..."

# 升级 pip 以避免安装时的解析错误
pip install --upgrade pip

# 【关键步骤】
# 1. 安装 vLLM >= 0.7.2 (支持 Qwen2.5-VL 的最低要求)
# 2. 安装 Transformers >= 4.49.0 (包含新模型架构定义)
# 3. 安装 Outlines (支持 guided_json)
# 4. 安装 Flash Attention (vLLM 强依赖)
echo ">>> 正在下载并安装 vLLM, Transformers, Outlines..."

# 使用清华源加速（如果需要），如果不需要请去掉 -i 参数
# pip install vllm==0.7.2 transformers>=4.49.0 outlines accelerate pillow pydantic -i https://pypi.tuna.tsinghua.edu.cn/simple

# 标准安装命令 (自动寻找最新版)
pip install "vllm>=0.7.2" "transformers>=4.49.0" "outlines>=0.1.0" accelerate pillow pydantic requests

# 6. 验证安装
echo "========================================================"
echo "   ✅ 环境构建完成！版本检查："
echo "========================================================"

python -c "import vllm; print(f'vLLM Version: {vllm.__version__} (Expect >= 0.7.2)')"
python -c "import transformers; print(f'Transformers: {transformers.__version__} (Expect >= 4.49.0)')"
python -c "import outlines; print(f'Outlines: {outlines.__version__}')"

echo "========================================================"
echo "🎉 请执行以下步骤启动服务："
echo "1. conda activate $ENV_NAME"
echo "2. 检查您的 start_vllm_final.sh，将 VLLM_CMD 修改为："
echo "   $(which vllm)"
echo "3. 运行 bash start_vllm_final.sh"
echo "========================================================"