from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import torch

# use method: 
# netstat -tulpn | grep :8000
# kill -9 pid

# 初始化FastAPI应用
app = FastAPI(title="Qwen2.5-VL-3B-Instruct API", version="1.0")

# 全局加载模型和处理器（仅加载一次，提升性能）
# 模型路径替换为你的本地路径
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

# 加载模型（强制GPU运行）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",  # 自动选择半精度，减少显存占用
    device_map="auto",  # 自动分配到GPU（4张L20会优先用GPU）
    trust_remote_code=True
).eval()  # 推理模式，禁用dropout

# 加载处理器
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# 定义推理核心函数
def infer(image: Image.Image, question: str) -> str:
    """
    核心推理函数：接收图片和问题，返回模型回答
    """
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # 传入PIL图片对象
                {"type": "text", "text": question}
            ],
        }
    ]

    # 预处理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")  # 强制输入到GPU

    # 推理生成
    with torch.no_grad():  # 禁用梯度，节省显存
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # 可根据需求调整
            # do_sample=False,  # 贪心解码，速度快
            # temperature=0.0
        )

    # 解析输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 定义POST接口：接收图片和问题
@app.post("/chat", summary="Qwen2.5-VL多模态对话")
async def chat(
    file: UploadFile = File(..., description="上传的图片文件（支持jpg/png/jpeg）"),
    question: str = Form(..., description="针对图片的问题文本")
):
    try:
        # 1. 验证文件格式
        allowed_extensions = {"jpg", "jpeg", "png"}
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="仅支持jpg/jpeg/png格式的图片")

        # 2. 读取图片文件并转为PIL对象
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 3. 调用推理函数
        result = infer(image, question)

        # 4. 返回结果
        return JSONResponse(
            status_code=200,
            content={
                "code": 0,
                "msg": "success",
                "question": question,
                "answer": result
            }
        )

    except Exception as e:
        # 异常处理
        raise HTTPException(status_code=500, detail=f"推理失败：{str(e)}")

# 健康检查接口（可选）
@app.get("/health", summary="服务健康检查")
async def health():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

# 启动服务（直接运行该脚本时触发）
if __name__ == "__main__":
    import uvicorn
    # 启动服务：host=0.0.0.0允许外部访问，port=8000（可自定义）
    uvicorn.run(
        "qwen_vl_api:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # 单进程（模型加载在全局，多进程会重复加载显存不足）
        reload=False  # 生产环境关闭热重载
    )