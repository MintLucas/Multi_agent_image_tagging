import requests
from PIL import Image
import io
import base64

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    


def process_url_image(image_url: str, max_edge: int = 768) -> str:
    """
    下载URL图片 -> 内存中Resize -> 转Base64
    这样可以确保 vLLM 接收到的永远是小图，无论源图多大
    """
    try:
        # 1. 下载图片 (设置超时防止卡死)
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # 检查是否下载成功
        
        # 2. 从内存字节读取图片
        image_bytes = io.BytesIO(response.content)
        
        with Image.open(image_bytes) as img:
            # 转换为RGB (防止PNG透明通道报错)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 3. Resize (核心优化步骤)
            # 打印一下看看原始多大
            # print(f"URL原图大小: {img.size}") 
            img.thumbnail((max_edge, max_edge))
            
            # 4. 转 Base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return f"data:image/jpeg;base64,{img_str}"
            
    except Exception as e:
        # 下载或处理失败，返回 None 或抛出异常
        print(f"URL图片处理失败: {e}")
        raise ValueError(f"无法下载或处理该URL: {e}")

def encode_image_resized(image_path, max_edge=768):
    """
    读取图片 -> Resize(长边限制在max_edge) -> 转Base64
    Qwen2.5-VL 推荐 768px 或 1024px，对于分类任务 768px 绰绰有余且速度极快。
    """
    try:
        with Image.open(image_path) as img:
            # 转换为RGB，防止PNG透明通道在保存为JPEG时报错
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保持比例缩放，限制长边
            # 原始图片可能 4000x3000 -> Resize 后 768x576 -> Token数减少 ~90%
            img.thumbnail((max_edge, max_edge))
            
            # 转为二进制流
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 返回带前缀的格式 (适配 vLLM/OpenAI 接口)
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"❌ 图片处理失败: {e}")
        return None
    

if __name__ == '__main__':
    # --- 使用场景举例 ---
    res = process_url_image("https://wuta-tmp.oss-cn-shanghai.aliyuncs.com/dingzhoufengtest/sina_tag/9%E3%80%81%E5%9B%BE%E7%89%87%E8%B4%A8%E9%87%8F/2%E3%80%81%E8%80%81%E7%85%A7%E7%89%87/50eddeef2c2290b8338cd44bd510a91.jpg")