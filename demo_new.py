import base64
import json
import time
import os
import io
from typing import List, Literal
from PIL import Image
from pydantic import BaseModel, Field

# 假设您的 model.py 已经按照建议修改，支持传入 schema
# 如果 model.py 还没改，请参考代码底部的 "注意" 部分
from model import CallVLMModel 

# ==========================================
# 优化方案 2 & 1: 定义 Pydantic Schema (Guided Decoding)
# ==========================================
# 利用代码定义结构，替代 Prompt 中的长文本约束

class PortraitDetails(BaseModel):
    # 1. 核心属性（强制必选，无 default）
    gender: List[Literal["男性", "女性"]] = Field(
        description="性别判定"
    )
    age: List[Literal["儿童", "少年", "青年", "中年", "老年"]] = Field(
        description="严格年龄段：儿童(0-10岁)、少年(11-18岁)、青年(19-35岁)、中年(36-59岁)、老年(60岁+)"
    )
    count: List[Literal["单人", "多人"]] = Field(
        description="人数判定：单人、多人（画面≥2人，背景中出现的非主体路人也算多人）"
    )
    
    # 2. 构图与风格（强制必选）
    composition: List[Literal["自拍", "合影", "正面", "侧面", "全身", "半身", "面部特写"]] = Field(
        description="构图标准：自拍(含手臂/自拍杆痕迹)；全身(头顶至脚底完整)；半身(头顶至大腿/腰部)；面部特写(头部占比≤30%或仅头部)"
    )
    usage: List[Literal["生活照", "证件照", "情侣照"]] = Field(
        description="用途：生活照(日常)、证件照(纯色背景/正式)、情侣照(亲密互动/甜蜜氛围)"
    )
    
    # 3. 发型特征（强制必选，除非完全被遮挡，否则必须判断）
    hair_length: List[Literal["长发", "短发"]] = Field(
        description="发长：长发(过肩/≥30cm)、短发(≤下巴/寸头/波波头)"
    )
    hair_type: List[Literal["卷发", "直发"]] = Field(
        description="发质：卷发(明显波浪/烫卷)、直发(顺直无明显卷曲)"
    )
    hair_style: List[Literal["扎发", "披发"]] = Field(
        description="形式：扎发(马尾/丸子头/辫子)、披发(自然散开)"
    )
    
    # 4. 面部与姿态（强制必选）
    expression: List[Literal["微笑", "大笑", "严肃", "闭眼"]] = Field(
        description="表情：微笑(嘴角上扬)、大笑(露齿/开朗)、严肃(无笑容/专注)、闭眼(休息/睡眠)"
    )
    posture: List[Literal["坐姿", "站立"]] = Field(
        description="姿态：坐姿(椅子/地面/沙发)、站立(自然站立/摆拍)"
    )

    # 5. 饰品（保留可选，default=[]，因为可能真的没有）
    accessories: List[Literal["眼镜", "帽子", "口罩", "耳环", "项链"]] = Field(
        default=[], 
        description="饰品(可多选)：眼镜(含墨镜/透明镜)、帽子、口罩、耳环、项链。无饰品则留空。"
    )

# 获取 JSON Schema 对象
JSON_SCHEMA = PortraitDetails.model_json_schema()

# ==========================================
# 优化方案 3: 图片分辨率压缩 (Resolution Truncation)
# ==========================================
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

# ==========================================
# 配置与主逻辑
# ==========================================

IMAGE_FOLDER = '/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图/2、人像细节/2.1 性别与年龄/'

# 优化后的极简 Prompt (Scheme 1)
# 所有的约束都在 JSON Schema 里了，Prompt 只需要告诉模型“做什么”
PROMPT = """
分析图片中的人像细节。
请严格根据提供的 JSON Schema 输出标签。
对于不确定的属性（如无法看清发型或没有饰品），请返回空列表。
"""
PROMPT =  """
        任务：基于图片，提取“人像”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 性别：男性、女性
        - 年龄：儿童（0-10岁）、少年（11-18岁）、青年（19-35岁）、中年（36-59岁）、老年（60岁及以上）【重要：年龄标签仅能从这5个选项中选择，严禁自创任何其他表述（如“青壮年”“青少年”“成年”“中老年”等）】
        - 人数：单人、多人（≥2人，注意：画面背景中不是主体的背景人物也属于多人，也要选择多人这个标签）
        - 构图：自拍（含手臂/自拍杆痕迹或高角度近距离）、合影（多人同框且分布均匀）、正面（人脸对称）、侧面（单侧脸颊/眼睛占比大）、全身（画面需完整容纳人物的头顶至脚底（或脚尖处），注意能同时看到一个人的头和脚就是全身）、半身（画面只能容纳在人物的头顶至大腿中部或腰部之间）、面部特写（画面仅保留人物的头部或完整面部区域，人物主体在整张画面中的占比≤30%，其余部分为背景或留白；若包含少量颈部（不超过颈部 1/3 长度），且整体占比仍符合≤30% 的要求，也可归类为面部特写）
        - 用途：生活照（日常随拍）、证件照（背景为纯色无杂物的红/蓝/白标准证件背景板，人物为正面头部/肩部特写且居中，着装整洁正式、多为免冠无夸张饰品，光线均匀无明显阴影，整体为身份证/护照/毕业证等官方证件专用照片风格）、情侣照（画面中有明显情侣互动姿势，如牵手、拥抱、亲吻等，人物表情甜蜜幸福，构图多为近距离或半身合影，整体氛围浪漫温馨，符合情侣专属照片风格）
        - 饰品：眼镜（注意：黑色墨镜，近视镜，透明眼镜等等也属于眼镜，不要漏掉）、帽子、口罩、耳环、项链
        - 发型长度：长发（头发长度过肩，或垂落至背部、胸前，整体发长≥30cm）、短发（头发长度≤下巴，常见寸头、波波头、齐耳短发等，整体发长＜15cm）
        - 发型直卷：卷发（头发呈自然卷/烫卷形态，有明显波浪、螺旋或羊毛卷纹理，非拉直状态）、直发（头发整体顺直无明显卷曲，垂落形态顺滑，无卷度或仅有轻微弧度）
        - 发型形式：扎发（头发被束起固定，含马尾、丸子头、麻花辫、高颅顶束发、半扎发等形态，非完全散开）、披发（头发完全自然散开，无束起、绑扎的痕迹，整体呈垂落/蓬松散开状态）
        - 表情：微笑（嘴角上扬，露出牙齿或不露齿均可，整体面部表情愉悦）、大笑（哈哈大笑，笑的豁然开朗，漏出牙齿的笑）严肃（面部表情平静，无明显笑容，嘴唇紧闭或微张，眼神专注有神）、闭眼（双眼稍微眯眼，注意稍微眯眼，不是正常的看镜头）
        - 姿态：坐姿（人物以坐着的姿势出现，含椅子、地面、沙发等多种坐姿场景）、站立（人物以站立的姿势出现，含自然站立、摆拍等多种站姿场景）
        """
output_require = """
        输出要求：严格用JSON格式返回，key为二级分类类型（如“性别”“年龄”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。并且标签必须使用完整标签名称，不能使用简称或缩写，并且不要带括号内解释内容。
        错误格式示例（禁止格式）：{"性别":["女性"], "年龄":["成年"], "人数":["单人", "自拍"], "备注":"图片为室内自拍"}
        正确格式示例（必须遵循的格式）：{"性别":["女性"], "年龄":["成年"], "人数":["单人"], "构图":["正面", "半身"], "用途":["生活照"], "饰品":["帽子"], "发型长度":["短发"], "发型直卷":["直发"], "发型形式":["披发"], "表情":["微笑"], "姿态":["站立"]}
        """
def main():
    # 1. 获取图片列表
    image_paths = []
    for root, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    print("=" * 50)
    print(f"📁 扫描到文件夹：{IMAGE_FOLDER}")
    print(f"✅ 共发现 {len(image_paths)} 张图片")
    print(f"⚡ 优化策略已启用：Resize(768px) + Guided JSON Schema")
    print("=" * 50 + "\n")

    # 2. 初始化模型
    model = CallVLMModel()

    # 3. 批量处理
    total_images = len(image_paths)
    success_count = 0
    total_elapsed_time = 0.0

    print("🚀 开始批量处理图片...\n")
    for idx, img_path in enumerate(image_paths, 1):
        print(f"{'='*30} 处理第 {idx}/{total_images} 张图片 {'='*30}")
        print(f"图片名称：{os.path.basename(img_path)}")
        
        try:
            # --- 优化点：使用 Resize 后的编码函数 ---
            print("正在编码图片 (Max 768px)...")
            base64_image = encode_image_resized(img_path, max_edge=768)
            
            if not base64_image:
                continue

            start_time = time.time()
            
            print("正在调用模型处理 (Guided Decoding)...")
            
            # --- 关键修改：调用模型时传入 schema ---
            # 假设您的 CallVLMModel.call_qwen_local_vl0 方法已更新以接收 schema
            # 如果是直接调用 OpenAI SDK，这里对应 extra_body={"guided_json": JSON_SCHEMA}
            # 这里我模拟传入一个 extra_args 字典，请确保您的 model.py 能处理它
            qwen_response = model.call_qwen_new(
                base64_image, 
                PROMPT,
                # 注意：您需要修改 model.py 的 call_qwen_local_vl0 函数签名来接收这个参数
                # 或者在 model.py 内部写死也可以，但在 demo 里传入是最灵活的
                schema= JSON_SCHEMA,
                service_index = 0
            )
            
            end_time = time.time()
            single_elapsed_time = end_time - start_time
            total_elapsed_time += single_elapsed_time
            success_count += 1
            
            # 结果解析
            content = qwen_response.get('content', '')
            
            # 如果用了 Guided Decoding，返回的一定是合法 JSON，无需清洗 markdown 符号
            # 但为了保险（防止 vLLM 版本不支持），还是保留简单的清洗逻辑
            if isinstance(content, str):
                try:
                    # 尝试直接解析
                    result_json = json.loads(content)
                except:
                    # 只有在 Guided Decoding 未生效时才会走到这里
                    clean_content = content.strip().replace("```json", "").replace("```", "")
                    result_json = json.loads(clean_content)
            else:
                result_json = content

            print(f"✅ 处理完成！")
            print(f"⏱️ 耗时：{single_elapsed_time:.2f} 秒 (目标 < 3.0s)")
            print(f"📉 Token消耗：Prompt={qwen_response.get('prompt_tokens',0)} | Completion={qwen_response.get('completion_tokens',0)}")
            print(f"🏷️ 标签结果：\n{json.dumps(result_json, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            print(f"❌ 处理失败！错误信息：{str(e)}")
        
        print(f"{'='*70}\n")

    # 4. 汇总
    print("=" * 50)
    print("📊 批量处理汇总统计")
    print("=" * 50)
    print(f"总图片数：{total_images}")
    print(f"平均每张耗时：{total_elapsed_time / success_count:.2f} 秒" if success_count else "N/A")
    print("=" * 50)

if __name__ == "__main__":
    main()