目前的流程是 `L1分类 (VLM)` -> `解析` -> `L2细节 (VLM)`。

* **痛点1（主要）：** **Double Vision Encoding**。Qwen2-VL处理高分辨率图片时，Vision Encoder产生的Token数非常多（一张1080P图片可能产生数千个Token）。串行调用两次意味着要对同一张图做两次视觉编码，这是最大的耗时来源。
* **痛点2：** **网络与调度开销**。Agent框架在节点切换时的中间处理、JSON解析、Prompt构建都会引入非推理耗时。

以下是具体的优化方案，按推荐优先级排序：

### 方案一：One-Pass 混合输出策略（架构级重构）

**目标：** 消除第二次Vision Encoding，一次推理拿到所有结果。

Qwen2.5-VL-3B虽然参数小，但其指令遵循能力足够强。不要将任务人为拆分成“先看主体，再看细节”，而是利用Prompt的结构化约束，让模型**一次性**输出层级结果。

**优化后工作流：**
`Image` -> `VLM (One-Shot)` -> `Hierarchical JSON`

**Prompt 改造示例：**
不需要分别Prompt，而是构建一个包含层级逻辑的System Prompt。

```python
# 伪代码：One-Pass Prompt
prompt = """
分析图片，输出严格的JSON格式。
首先判断【一级主体】，根据主体类型输出对应的【二级细节】。
如果主体是"人像"，必须包含"人像细节"字段；如果主体是"动物"，必须包含"动物细节"。

输出格式定义(Schema):
{
    "primary_category": "人像" | "动物" | "风景" | "食物" | "建筑",
    "details": {
         // 当且仅当 primary_category 为 "人像" 时填充
        "person_attributes": {
            "gender": "...",
            "composition": "..."
        },
        // 当且仅当 primary_category 为 "动物" 时填充
        "pet_attributes": { ... }
    }
}
"""

```

* **收益：** 耗时直接砍半，理论上可以直接达到 **3s 左右**（取决于Token数）。
* **风险：** 3B模型在复杂指令下可能会输出幻觉。
* **对策：** 配合 **Guided Decoding (Grammar Sampling)** 使用。

### 方案二：vLLM 引导解码 (Guided Decoding)

您使用的是 vLLM，这是目前针对结构化输出最强的推理后端之一。不要让模型自己去“理解”JSON格式，而是强制约束Logits。

**技术实现：**
利用 vLLM 的 `guided_json` 参数。这将强制模型生成的 Token 必须符合您定义的 JSON Schema。这不仅保证了格式100%正确（免去Retry耗时），还能显著减少模型输出“废话”的Token数量，从而降低 `Generated Token` 的耗时。

```python
# 定义 Pydantic 模型对应您的标签文档
from pydantic import BaseModel, Field
from enum import Enum

class MainCategory(str, Enum):
    PORTRAIT = "人像"
    ANIMAL = "动物"
    # ...

class PortraitDetails(BaseModel):
    gender: str = Field(..., enum=["男性", "女性"])
    composition: list[str] = Field(..., description="构图，如自拍、全身")

class ImageAnalysis(BaseModel):
    category: MainCategory
    # 使用 Optional 配合描述，引导模型根据 category 填充
    portrait_info: PortraitDetails | None = None
    
# 调用 vLLM
sampling_params = SamplingParams(
    temperature=0.1,
    # 核心优化点：强制约束输出结构
    guided_json=ImageAnalysis.model_json_schema()
)

```

### 方案三：视觉编码层面的“分辨率截断”

**这是针对 Qwen-VL 系列最立竿见影的提速手段。**

Qwen2-VL 使用的是 Naive Dynamic Resolution，它会将图片切片。

* 如果用户上传一张 4032x3024 的照片，模型会产生数千个 Image Tokens，推理极慢。
* 对于“分类”和“粗粒度标签”任务，**根本不需要原图分辨率**。

**操作：**
在送入 vLLM 之前，在代码层强制 Resize 图片。

* **策略：** 将长边限制在 **448px - 768px** 之间。
* **数据支撑：** 对于识别“是猫还是狗”或者“是自拍还是全身”，512px 的分辨率对于 Vision Transformer 已经绰绰有余。
* **收益：** Input Token 数量呈指数级下降，Prefill 时间从 1s+ 降至 200ms 级别。

```python
# 在 base64 编码前添加 resize 逻辑
def process_image(image_path):
    with Image.open(image_path) as img:
        # 限制最大边长为 768，大幅减少 token 数
        img.thumbnail((768, 768)) 
        # 转 base64 ...

```

### 方案四：Hybrid 路由策略 (SigLIP + VLM)

您提到 YOLO v11 效果不好，这是因为 YOLO 是 Object Detection 模型，且类别映射僵硬。您应该使用 **CLIP** 或 **SigLIP** (Google出品，目前SOTA的Zero-shot分类模型) 来做 Level 1 分类。

**新架构：**

1. **L1 分类 (SigLIP-So400M):** 这是一个纯向量模型，耗时 **<50ms**。准确率极高，特别是对于 `人像` vs `动物` vs `风景` 这种大类。
2. **路由:** * 如果是 `建筑` / `植物` -> 直接结束（无需调用 VLM，节省大量时间）。
* 如果是 `人像` -> 调用 VLM 仅提取人像细节。
* 如果是 `动物` -> 调用 VLM 仅提取动物细节。



**收益：**

* 对于不需要细节的类别（如文档中提到的植物/建筑），延时降至毫秒级。
* 对于需要细节的类别，省去了一次 VLM 调用。
* SigLIP 可以在同一个 L20 GPU 上作为旁路运行，显存占用极小。

### 方案五：vLLM 工程级参数调优 (Prefix Caching)

您的 Prompt（包含大量标签定义）非常长。如果每次请求都重新计算这些 Prompt 的 KV Cache，非常浪费。

**配置：**
在 vLLM 启动参数中开启：
`--enable-prefix-caching`

**操作：**
确保您的 System Prompt（包含所有标签定义的部分）在所有请求中是**完全一致**的前缀。

* **效果：** vLLM 会缓存这段长文本的显存状态。第二个请求进来时，直接跳过 System Prompt 的计算，只计算图片部分的 Attention。
* **收益：** TTFT (Time To First Token) 显著降低。

---
