import base64
import json
from PIL import Image
from io import BytesIO
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from model import CallVLMModel
from utils import encode_image, encode_image_resized, process_url_image
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage, AIMessage
from logger import get_logger
import os
import time
import pandas as pd

# ========== FastAPI相关导入 ==========
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ========== 引入新定义的 Schemas ==========
from schemas import (
    FirstLevelSchema, 
    PortraitDetailsSchema, 
    ClothingDetailsSchema,
    PetDetailsSchema,
    FoodDetailsSchema,
    SceneryDetailsSchema,
    SceneTypeSchema
)

fast_app = FastAPI(title="图片标签生成API", description="单张图片标签提取接口，基于LangGraph实现", version="1.0.0")

# 白名单保持不变，用于最后的合法性校验
TAG_WHITELIST = {
    "主体": ["人像", "动物（宠物）", "植物", "风景", "食物", "建筑"],
    "人像": {
        "性别": ["男性", "女性"],
        "年龄": ["儿童", "少年", "青年", "中年", "老年"],
        "人数": ["单人", "多人"],
        "拍摄方式": ["自拍", "他拍", "合影"],
        "构图": ["全身", "半身", "面部特写"],
        "角度": ["正面", "侧面", "背影"],
        "用途": ["生活照", "证件照", "情侣照"],
        "饰品": ["帽子", "口罩", "耳环", "项链"],
        "发型长度": ["长发", "短发"],
        "发型直卷": ["卷发", "直发"],
        "发型形式": ["扎发", "披发"],
        "表情": ["微笑", "大笑", "严肃", "闭眼"],
        "姿态": ["坐姿", "站立"],
        "服饰": {
            "眼镜": ["眼镜", "无眼镜"],
            "基本款式": ["西装", "职业装", "T恤", "衬衫", "毛衣", "羽绒服", "裙子", "运动装", "睡衣", "校服", "婚纱", "泳装"],
            "题材": ["cosplay", "lolita", "jk", "旗袍", "新中式", "民族服装", "夏装", "冬装", "春秋装"],
            "风格": ["休闲风", "街头风", "正式风", "学院风"]
        }
    },
    "动物（宠物）": {
        "种类": ["狗", "猫", "鸟", "鱼", "兔子", "其他"],
        "数量": ["单只", "多只"],
        "视角与状态": ["宠物正面", "宠物全身", "室内宠物图", "户外宠物图"]
    },
    "食物": {
        "食物类型": ["中餐", "西餐", "甜品", "奶茶", "火锅", "水果", "烧烤", "主菜", "小吃", "饮品"],
        "拍摄场景": ["桌面摆盘", "俯拍", "特写", "居家烹饪", "餐厅环境"]
    },
    "风景": {
        "地貌场景": ["海边", "山脉", "森林", "草原", "沙漠", "瀑布", "湖泊", "花海", "峡谷"],
        "城市天空": ["天空", "城市夜景", "日落", "星空"],
        "季节相关": ["春季", "夏季", "秋季", "冬季"]
    },
    "场景": {
        "空间": ["室内", "室外"],
        "场所类型": ["自然", "家居", "餐厅", "健身房", "游乐园", "音乐节", "KTV", "演唱会"],
        "时间": ["白天", "夜晚"],
        "天气": ["晴天", "阴天", "多云", "雨天", "雪天", "雾天", "彩虹"],
        "光线": ["自然光", "逆光"],
        "特殊元素": ["烟花", "圣诞树", "气球", "彩带", "蛋糕", "粽子", "元宵", "月饼", "礼物盒"],
        "水印": ["水印"],
        "图片质量": ["无路人", "有路人", "老照片"],
        "节日": ["生日", "婚礼", "圣诞", "春节", "中秋", "端午", "万圣节", "国庆"]
    }
}

class ImagePathRequest(BaseModel):
    image_info: str

logger = get_logger(service="lg_builder")
model = CallVLMModel()

# 状态定义保持不变
class ImageTaggingState(TypedDict):
    image_info: str
    first_level: dict
    second_level_person: dict
    second_level_person_cloth: dict
    second_level_pet: dict
    second_level_food: dict
    second_level_scenery: dict
    all_scene_type: dict
    final_labels: list[str]
    messages: list[AnyMessage]
    first_level_token_price: float
    second_level_person_token_price: float
    second_level_person_cloth_token_price: float
    second_level_pet_token_price: float
    second_level_food_token_price: float
    second_level_scenery_token_price: float
    all_scene_type_token_price: float
    total_tokens_price: float
    start_time: float
    end_time: float
    token_price_input: float
    token_price_output: float

# ==========================================
# 节点函数优化 (Prompt精简 + Schema调用)
# ==========================================

def first_level_classification(state: ImageTaggingState) -> ImageTaggingState:
    start_time = time.time()
    image_info = state["image_info"]
    
    # Prompt 只需要定义业务逻辑，不需要教模型JSON格式
    prompt = """
    任务：判断图片的核心主体，仅从以下一级分类的六个分类中选择（可以多选，不新增，如果都不含有就选其他）：
    一级分类列表：人像、动物（宠物）、植物、风景（注意：图片中有花朵，或者花海，以及植物都属于风景标签）、食物（注意：冰饮、饮品、奶茶、酒之类喝的也属于食物标签）、建筑、其他
    一级分类列表：人像、动物（宠物）、植物、风景（注意：图片中有花朵，或者花海，以及植物都属于风景标签）、食物（注意：冰饮、饮品、奶茶、酒之类喝的也属于食物标签）、建筑、其他
    注意：对于风景来说，如果图片中有积雪、飘雪、雪花、冰雕的话，可以加上标签“风景”; 如果图片中有蓝天白云、日落、星空、城市夜景等，也可以加上标签“风景”。
    输出要求：严格用JSON格式返回，key为一级分类类型（如“主体”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    如果一张图片同时含有了 人像，动物，风景。 请全部列出。
    """
    extend_require = """
        错误示例（禁止）：{"主体":"人像","备注":"图片为人"}
    正确示例（必须遵循）：{"主体":["人像","动物（宠物）","风景"]}
    """
    logger.info("-----First_level_classification (Guided)-----")
    # 传入 Schema
    schema = FirstLevelSchema.model_json_schema()
    all_response = model.call_qwen_new(image_info, prompt, schema=schema)
    
    first_level_token_price = (all_response["prompt_tokens"]/1000)*state["token_price_input"] + (all_response["completion_tokens"]/1000)*state["token_price_output"]
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=all_response["content"]))

    try:
        # Guided Decoding返回的一定是JSON，直接loads即可
        # 兼容性处理：防止偶尔带markdown包裹
        clean_content = all_response["content"].strip()
        if clean_content.startswith("```"):
             clean_content = clean_content.replace("```json", "").replace("```", "")
        first_level_label = json.loads(clean_content)
    except Exception as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        first_level_label = {}

    logger.info(f"一级分类标签：{first_level_label}")
    return {"first_level": first_level_label,
            "first_level_token_price": first_level_token_price,
            "start_time": start_time}

def second_level_person(state: ImageTaggingState) -> ImageTaggingState:
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])
    
    if "人像" not in main_labels:
        return 
    else:
        image_info = state["image_info"]
        # 精简后的 Prompt，保留判断标准
        prompt = """
        任务：基于图片，提取“人像”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 性别：男性、女性
        - 年龄：儿童（0-10岁）、少年（11-18岁）、青年（19-35岁）、中年（36-59岁）、老年（60岁及以上）【重要：年龄标签仅能从这5个选项中选择，严禁自创任何其他表述（如“青壮年”“青少年”“成年”“中老年”等）】
        - 人数：单人、多人（画面里出现≥2个人物）
        - 拍摄方式：自拍（含手持手机、露出手臂、自拍杆、高角度近距离俯拍任一特征）、他拍（非自拍非合影的单人拍摄）、合影（两人及以上同框）；
        - 构图：全身（完整呈现人物头顶至脚底）、半身（头顶至大腿中部 / 腰部）、面部特写（仅头部或完整面部，主体占比≤30%）；
        - 角度：正面（人脸对称正对镜头）、侧面（单侧脸颊 / 眼睛为主）、背影（仅看到背部无面部）
        - 用途：生活照（日常随拍）、证件照（背景纯色无杂物红 / 蓝 / 白，人物正面头部 / 肩部特写居中，着装正式整洁免冠无夸张饰品，光线均匀无明显阴影，为身份证 / 护照 / 毕业证等官方证件专用）、情侣照（画面中有两名异性人物距离较近、身体靠近或紧挨，或有牵手、拥抱、依偎、对视、搭肩等情侣互动，氛围浪漫甜蜜）
        - 饰品：帽子、口罩、耳环、项链
        - 发型长度：长发（头发长度过肩，或垂落至背部、胸前，整体发长≥30cm）、短发（头发长度≤下巴，常见寸头、波波头、齐耳短发等，整体发长＜15cm）
        - 发型直卷：卷发（头发呈自然卷/烫卷形态，有明显波浪、螺旋或羊毛卷纹理，非拉直状态）、直发（头发整体顺直无明显卷曲，垂落形态顺滑，无卷度或仅有轻微弧度）
        - 发型形式：扎发（头发被束起固定，含马尾、丸子头、麻花辫、高颅顶束发、半扎发等形态，非完全散开）、披发（头发完全自然散开，无束起、绑扎的痕迹，整体呈垂落/蓬松散开状态）
        - 表情：微笑（嘴角上扬，露出牙齿或不露齿均可，整体面部表情愉悦）、大笑（哈哈大笑，笑的豁然开朗，漏出牙齿的笑）严肃（面部表情平静，无明显笑容，嘴唇紧闭或微张，眼神专注有神）、闭眼（双眼稍微眯眼，注意稍微眯眼，不是正常的看镜头）
        - 姿态：坐姿（人物以坐着的姿势出现，含椅子、地面、沙发等多种坐姿场景）、站立（人物以站立的姿势出现，含自然站立、摆拍等多种站姿场景）
        输出要求：严格用JSON格式返回，key为二级分类类型（如“性别”“年龄”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。并且标签必须使用完整标签名称，不能使用简称或缩写，并且不要带括号内解释内容。
        错误格式示例（禁止格式）：{"性别":["女性"], "年龄":["成年"], "人数":["单人", "自拍"], "备注":"图片为室内自拍"}
        正确格式示例（必须遵循的格式）：{"性别":["女性"], "年龄":["成年"], "人数":["单人"], "构图":["正面", "半身"], "用途":["生活照"], "饰品":["帽子"], "发型长度":["短发"], "发型直卷":["直发"], "发型形式":["披发"], "表情":["微笑"], "姿态":["站立"]}
        """
        logger.info("-----Second_level_person (Guided)-----")
        schema = PortraitDetailsSchema.model_json_schema()
        response = model.call_qwen_new(image_info, prompt, schema=schema)
        
        price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_content = response["content"].strip()
            if clean_content.startswith("```"):
                 clean_content = clean_content.replace("```json", "").replace("```", "")
            data = json.loads(clean_content)
        except Exception as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            data = {}
            
        logger.info(f"二级人像细节标签：{data}")
        return {"second_level_person": data, "second_level_person_token_price": price}

def third_level_person_cloth(state: ImageTaggingState) -> ImageTaggingState:
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])
    
    if "人像" not in main_labels:
        return 
    else:
        image_info = state["image_info"]
        prompt = """
        任务：基于图片，提取“人像”的服饰标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 眼镜：眼镜、无眼镜
        - 服饰款式：西装、职业装、T恤、衬衫、毛衣、羽绒服、裙子、运动装、睡衣、校服、婚纱、泳装
        - 服饰题材：cosplay、lolita、jk、旗袍、新中式、民族服装、夏装、冬装、春秋装
        - 服饰风格：休闲风、街头风、正式风、学院风
        输出要求：严格用JSON格式返回，key为分类类型（如“基本款式”“题材”“风格”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误示例（禁止）：{"基本款式":["西装"], "题材":["新中式"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"基本款式":["西装","职业装"], "题材":["新中式"], "风格":["正式风"]}
        """
        schema = ClothingDetailsSchema.model_json_schema()
        response = model.call_qwen_new(image_info, prompt, schema=schema)
        
        price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_content = response["content"].strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_content)
        except Exception as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            data = {}
        
        logger.info(f"三级人像服饰标签：{data}")
        return {"second_level_person_cloth": data, "second_level_person_cloth_token_price": price}

def second_level_pet(state: ImageTaggingState) -> ImageTaggingState:
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])
    
    if "动物（宠物）" not in main_labels:
        return 
    else:
        image_info = state["image_info"]
        prompt = """
        任务：基于图片，提取“动物”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 种类：狗、猫、鸟、鱼、兔子、其他（注意只涉及这5种动物，不确定的话就选 其他）
        - 数量：单只、多只
        - 视角与状态：宠物正面、宠物全身、室内宠物图、户外宠物图
        输出要求：严格用JSON格式返回，key为二级分类类型（如“种类”“数量”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        再次声明：仅限于上述5种动物（狗、猫、鸟、鱼、兔子）分类，不是这5中动物的其他种类。就选 其他。
        错误示例（禁止）：{"种类":["犀牛"], "数量":["单人"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"种类":["狗"], "数量":["单只"], "视角与状态":["宠物正面","室内宠物图"]}
        """
        schema = PetDetailsSchema.model_json_schema()
        response = model.call_qwen_new(image_info, prompt, schema=schema)
        
        price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_content = response["content"].strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_content)
        except Exception as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            data = {}
        logger.info(f"二级动物细节标签：{data}")
        return {"second_level_pet": data, "second_level_pet_token_price": price}

def second_level_scenery(state: ImageTaggingState) -> ImageTaggingState:
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])
    
    if "风景" not in main_labels:
        return 
    else:
        image_info = state["image_info"]
        prompt = """
        【核心规则（优先级最高）】：
        1. 仅标注图片中**明确可见、100%确定**的元素，无则完全不标注该类别，坚决杜绝猜测、虚构标签；
        2. 即使只有1个标签也可，无需凑数；不确定的标签直接忽略，宁少勿错；
        3. 所有标签必须从预设选项中选择，禁止新增任何未列出的标签。
        任务：基于图片，提取“风景”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 地貌场景：海边、山脉、森林、草原、沙漠、瀑布、湖泊、花海、峡谷
        - 城市天空：天空（注意：如果含蓝天白云，一定要加上天空这个标签）、城市夜景、日落、星空
        - 季节相关：春季（含有樱花、桃花、梨花、嫩芽、柳树、蒲公英、油菜花、洋甘菊等）、夏季（含有荷花、荷叶、浓绿树荫、繁茂草丛、烈日、西瓜等，如果图片中有人穿短袖，泳装，或者佩戴太阳镜，黑色墨镜也可以判断为夏季）、秋季（含有枫叶、银杏、落叶、枯草、麦浪等）、冬季（首先可以根据如果图片中含有积雪、飘雪、冰雕、冰凌、雾凇、枯枝、梅花来判断，其次如果人像穿了羽绒服、冬季棉袄之类的都可以判定为冬季）
        输出要求：严格用JSON格式返回，key为二级分类类型（如“种类”“数量”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误示例（禁止）：{"地貌场景":["海边"], "城市天空":["天空"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"地貌场景":["海边"], "城市天空":["水面"], "季节相关":["春季"]}
        """
        schema = SceneryDetailsSchema.model_json_schema()
        response = model.call_qwen_new(image_info, prompt, schema=schema)
        
        price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_content = response["content"].strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_content)
        except Exception as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            data = {}
        logger.info(f"二级风景细节标签：{data}")
        return {"second_level_scenery": data, "second_level_scenery_token_price": price}

def second_level_food(state: ImageTaggingState) -> ImageTaggingState:
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])
    
    if "食物" not in main_labels:
        return 
    else:
        image_info = state["image_info"]
        prompt = """
        任务：基于图片，提取“食物细节”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 食物类型：中餐、西餐、甜品、奶茶、火锅、水果、烧烤、主菜、小吃、饮品
        - 拍摄场景：桌面摆盘、俯拍、特写、居家烹饪、餐厅环境
        输出要求：严格用JSON格式返回，key为二级分类类型（如“食物类型”“拍摄场景”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误示例（禁止）：{"食物类型":["饮品"], "拍摄场景":["单人"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"食物类型":["火锅"], "拍摄场景":["俯拍"]}
        """
        schema = FoodDetailsSchema.model_json_schema()
        response = model.call_qwen_new(image_info, prompt, schema=schema)
        
        price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_content = response["content"].strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_content)
        except Exception as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            data = {}
        
        return {"second_level_food": data, "second_level_food_token_price": price}

def all_scene_type(state: ImageTaggingState) -> ImageTaggingState:
    image_info = state["image_info"]
    prompt = """
    【核心规则（优先级最高）】：
    1. 仅标注图片中**明确可见、100%确定**的元素，无则完全不标注该类别，坚决杜绝猜测、虚构标签；
    2. 即使只有1个标签也可，无需凑数；不确定的标签直接忽略，宁少勿错；
    3. 所有标签必须从预设选项中选择，禁止新增任何未列出的标签。

    任务：基于图片提取场景类型，仅从以下预设选项中选择（无则不选）：
    - 空间：室内（封闭空间，如家里/商场）、室外（开放空间，如街道/公园）
    - 场所类型：自然（山川/湖泊/森林等自然景观）、家居（家庭居住场景）、餐厅（餐饮消费场景）、健身房、游乐园、音乐节、KTV、演唱会
    - 时间：白天（有自然光、可见太阳/天空明亮）、夜晚（无自然光、需人工照明、有月亮）
    - 天气：晴天（无云/少量云，阳光充足）、阴天（云层厚，无阳光）、多云（部分云层）、雨天（有雨/积水）、雪天（有雪/积雪）、雾天（能见度低的雾气）、彩虹（可见完整/部分彩虹）
    - 光线：逆光（光源在主体后方，主体有光晕）、自然光（光源均匀，无明显阴影，主体清晰可见）
    - 特殊元素：
      - 烟花：画面中有燃放的烟火/烟花特效（排除印刷/图片里的烟花图案）；
      - 圣诞树：圣诞装饰用的松树/仿真圣诞树（排除普通松树）；
      - 气球：彩色装饰用实体气球（排除印刷/广告里的气球图案）；
      - 彩带：彩色装饰彩带/拉花（实体）；
      - 蛋糕：生日/节日用实体蛋糕（排除图片/广告里的蛋糕）；
      - 粽子：端午三角粽/粽叶包裹的粽子；
      - 元宵：水煮糯米圆子（元宵/汤圆）；
      - 月饼：中秋圆形糕点（月饼）；
      - 礼物盒：包装好的实体礼品盒；
     - 水印：[必选] 
      * 水印：只要看到角落有文字/ID、平台Logo、时间戳或画面有防盗纹，必须选此项。
      * 无水印：画面完全干净，无任何附加标记。
    - 图片质量：
      - 无路人：画面主体只有人像，无其他无关人物；
      - 有路人：画面中有除主体外的其他人物；
      - 老照片：画面泛黄/有划痕/色调老旧、富有年代感（排除仅模糊但无年代感的图片）；
    - 节日：
      - 生日：有蛋糕/蜡烛/生日帽等生日主题元素；
      - 婚礼：有婚纱/戒指/喜字等婚礼元素；
      - 圣诞：有圣诞树/圣诞帽/雪花装饰等圣诞元素；
      - 春节：有春联/福字/鞭炮/红包等春节元素；
      - 中秋：有月饼/圆月/灯笼等中秋元素；
      - 端午：有粽子/龙舟/艾草等端午元素；
      - 万圣节：有南瓜灯/鬼怪装饰/巫师帽等万圣节元素；
      - 国庆：有国旗/国庆装饰/天安门等国庆元素。
    """
    extend_require = """
        输出要求：
    1. 严格用JSON格式返回，key为二级分类名称（如“场所类型”），value为标签列表；
    2. 无对应标签的类别**完全不显示**，不出现空列表；
    3. 不添加任何额外文字、解释、标点或备注。

    正确示例1（有多个标签）：
    {"场所类型":["室内","家居"], "时间":["白天"], "天气":["晴天"], "特殊元素":["蛋糕","气球"], "图片质量":["无路人"], "节日":["生日"]}

    正确示例2（仅少量标签，无则不选）：
    {"场所类型":["室外","自然"], "时间":["白天"], "天气":["阴天"], "图片质量":["老照片"]}

    正确示例3（极少标签）：
    {"场所类型":["餐厅"], "图片质量":["有路人"]}
    """
    schema = SceneTypeSchema.model_json_schema()
    response = model.call_qwen_new(image_info, prompt, schema=schema)
    
    price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response["content"]))
    
    try:
        clean_content = response["content"].strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_content)
    except Exception as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        data = {}
    logger.info(f"场景类型标签：{data}")
    return {"all_scene_type": data, "all_scene_type_token_price": price}

# ==========================================
# 辅助函数保持不变
# ==========================================
def is_tag_legal(tag_str: str) -> bool:
    tag_parts = tag_str.split("-")
    if len(tag_parts) < 2: return False
    
    if tag_parts[0] == "主体" and len(tag_parts) == 2:
        return tag_parts[1] in TAG_WHITELIST.get("主体", [])
    
    if tag_parts[0] == "人像" and tag_parts[1] == "服饰" and len(tag_parts) == 4:
        _, _, cloth_type, cloth_value = tag_parts
        return TAG_WHITELIST["人像"]["服饰"].get(cloth_type, []).count(cloth_value) > 0
    
    if len(tag_parts) == 3:
        main_type, sub_type, value = tag_parts
        if main_type not in TAG_WHITELIST: return False
        if main_type == "人像" and sub_type != "服饰":
            return TAG_WHITELIST["人像"].get(sub_type, []).count(value) > 0
        return TAG_WHITELIST[main_type].get(sub_type, []).count(value) > 0
    return False

def format_output(state: ImageTaggingState) -> ImageTaggingState:
    final_labels = []
    end_time = time.time()
    
    # 1. 主体
    first_level = state.get("first_level", {})
    second_level_person = state.get("second_level_person", {})
    all_scene_type = state.get("all_scene_type", {})
    
    # ================= 核心修正逻辑 =================
    # 逻辑：如果场景检测到“有路人”，则人像数量强制修正为“多人”
    # 原因：场景节点对背景路人更敏感，以此为准解决冲突
    scene_quality = all_scene_type.get("图片质量", [])
    if "有路人" in scene_quality:
        # 检查当前是否标记了单人，如果是，则修正
        if "单人" in second_level_person.get("人数", []):
            logger.info("逻辑修正：检测到'有路人'，将人像'单人'修正为'多人'")
            second_level_person["人数"] = ["多人"]
    
    for subject in first_level.get("主体", []):
        tag = f"主体-{subject}"
        if is_tag_legal(tag): final_labels.append(tag)

    # 2. 人像二级
    second_level_person = state.get("second_level_person", {})
    for label_type, values in second_level_person.items():
        if isinstance(values, list):
            for value in values:
                tag = f"人像-{label_type}-{value}"
                if is_tag_legal(tag): final_labels.append(tag)

    # 3. 人像服饰
    second_level_person_cloth = state.get("second_level_person_cloth", {})
    for label_type, values in second_level_person_cloth.items():
        if isinstance(values, list):
            for value in values:
                tag = f"人像-服饰-{label_type}-{value}"
                if is_tag_legal(tag): final_labels.append(tag)

    # 4. 宠物
    second_level_pet = state.get("second_level_pet", {})
    for label_type, values in second_level_pet.items():
        if isinstance(values, list):
            for value in values:
                tag = f"动物（宠物）-{label_type}-{value}"
                if is_tag_legal(tag): final_labels.append(tag)

    # 5. 食物
    second_level_food = state.get("second_level_food", {})
    for label_type, values in second_level_food.items():
        if isinstance(values, list):
            for value in values:
                tag = f"食物-{label_type}-{value}"
                if is_tag_legal(tag): final_labels.append(tag)

    # 6. 风景
    second_level_scenery = state.get("second_level_scenery", {})
    for label_type, values in second_level_scenery.items():
        if isinstance(values, list):
            for value in values:
                tag = f"风景-{label_type}-{value}"
                if is_tag_legal(tag): final_labels.append(tag)

    # 7. 场景
    all_scene_type = state.get("all_scene_type", {})
    for label_type, values in all_scene_type.items():
        if isinstance(values, list):
            for value in values:
                tag = f"场景-{label_type}-{value}"
                if is_tag_legal(tag): final_labels.append(tag)

    final_labels = sorted(list(set(final_labels)))
    return {"final_labels": final_labels, "end_time": end_time}

# ==========================================
# Workflow 定义
# ==========================================
workflow = StateGraph(ImageTaggingState)
workflow.add_node("first_level_classification", first_level_classification)
workflow.add_node("second_level_person", second_level_person)
workflow.add_node("third_level_person_cloth", third_level_person_cloth)
workflow.add_node("second_level_pet", second_level_pet)
workflow.add_node("second_level_scenery", second_level_scenery)
workflow.add_node("second_level_food", second_level_food)
workflow.add_node("all_scene_type", all_scene_type)
workflow.add_node("format_output", format_output)

workflow.add_edge(START, "first_level_classification")
workflow.add_edge(START, "all_scene_type")

# 并行边
workflow.add_edge("first_level_classification", "second_level_person")
workflow.add_edge("first_level_classification", "third_level_person_cloth")
workflow.add_edge("first_level_classification", "second_level_pet")
workflow.add_edge("first_level_classification", "second_level_scenery")
workflow.add_edge("first_level_classification", "second_level_food")

# 汇聚到格式化
workflow.add_edge("second_level_person", "format_output")
workflow.add_edge("third_level_person_cloth", "format_output")
workflow.add_edge("second_level_pet", "format_output")
workflow.add_edge("second_level_scenery", "format_output")
workflow.add_edge("second_level_food", "format_output")
workflow.add_edge("all_scene_type", "format_output")
workflow.add_edge("format_output", END)

app = workflow.compile()

# URL/File 校验辅助函数
def is_http_https_url(s: str) -> bool:
    return s.strip().lower().startswith(("http://", "https://"))

def is_valid_image_file(s: str) -> bool:
    if not os.path.exists(s): return False
    return s.lower().endswith(('.png', '.jpg', '.jpeg'))

# 单图处理入口
def process_single_image(img_path: str) -> dict:
    try:
        content_stripped = img_path.strip()
        if is_http_https_url(content_stripped):
            image_content = process_url_image(content_stripped)
        elif is_valid_image_file(content_stripped):
            image_content = encode_image_resized(content_stripped)
        else:
            raise ValueError(f"无效的图片路径或URL：{img_path}")

        initial_state: ImageTaggingState = {
            "image_info": image_content,
            "first_level": {}, 
            "second_level_person": {}, 
            "second_level_person_cloth": {},
            "second_level_pet": {}, 
            "second_level_food": {}, 
            "second_level_scenery": {},
            "all_scene_type": {}, 
            "final_labels": [], 
            "messages": [],
            "first_level_token_price": 0.0,
            "second_level_person_token_price": 0.0,
            "second_level_person_cloth_token_price": 0.0,
            "second_level_pet_token_price": 0.0,
            "second_level_food_token_price": 0.0,
            "second_level_scenery_token_price": 0.0,
            "all_scene_type_token_price": 0.0,
            "total_tokens_price": 0.0,
            "start_time": time.time(),
            "end_time": 0.0,
            "token_price_input": 0.0012,
            "token_price_output": 0.0036
        }

        result = app.invoke(initial_state)

        elapsed_time = result["end_time"] - result["start_time"]
        token_fields = [
            "first_level_token_price",
            "second_level_person_token_price",
            "second_level_person_cloth_token_price",
            "second_level_pet_token_price",
            "second_level_food_token_price",
            "second_level_scenery_token_price",
            "all_scene_type_token_price"
        ]
        total_tokens_price = sum([result.get(field, 0.0) for field in token_fields])

        return {
            "image_info": img_path,
            "final_labels": result["final_labels"],
            "total_labels_count": len(result["final_labels"]),
            "elapsed_time": round(elapsed_time, 2),
            "token_cost": round(total_tokens_price, 4),
            "status": "success",
            "error": ""
        }

    except Exception as e:
        error_msg = str(e)[:200]
        logger.error(f"处理失败 {img_path}: {error_msg}")
        return {
            "image_info": img_path,
            "final_labels": [],
            "total_labels_count": 0,
            "elapsed_time": 0.0,
            "token_cost": 0.0,
            "status": "failed",
            "error": error_msg
        }
        
import asyncio
@fast_app.post("/process_image", response_description="单张图片标签处理结果")
async def api_process_image(request: ImagePathRequest):
    img_path = request.image_info.strip()
    if not img_path:
        raise HTTPException(status_code=400, detail="图片路径不能为空")
    result = await asyncio.to_thread(process_single_image, img_path)
    return {"res":result, "code": 200, "task_id": img_path}

if __name__ == "__main__":
    # 启动FastAPI服务，默认端口8000
    uvicorn.run(
        "image_uds_local_new:fast_app",  # 注意：如果文件名不是main.py，需替换为实际文件名
        host="0.0.0.0",      # 允许外部访问
        port=8081
        # reload=True          # 开发模式自动重载
    )