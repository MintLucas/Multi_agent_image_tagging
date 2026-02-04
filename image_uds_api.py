import base64
import json
from PIL import Image
from io import BytesIO
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from model import CallVLMModel
from utils import encode_image
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

fast_app = FastAPI(title="图片标签生成API", description="单张图片标签提取接口，基于LangGraph实现", version="1.0.0")

# ========== 定义请求模型 ==========
class ImagePathRequest(BaseModel):
    image_path: str  # 输入：单张图片的绝对路径

# 构建日志记录器
logger = get_logger(service="lg_builder")
model = CallVLMModel()


class ImageTaggingState(TypedDict):
    """图片标签工具的状态定义"""
    image_base64: str  # 图片Base64编码（必需）

    first_level: dict  # 一级标签（如"人像"、"食物"）
    second_level_person: dict  # 二级标签（JSON解析后的字典，如{"性别":["女"], "构图":["自拍"]}）
    second_level_person_cloth: dict  # 三级服饰标签（JSON解析后的字典，如{"基本款式":["西装"], "题材":["新中式"]}）
    second_level_pet: dict  # 二级标签（JSON解析后的字典，如{"种类":["狗"], "数量":["单只"]}）
    second_level_food: dict
    second_level_scenery: dict
    all_scene_type: dict  # 场景类型标签（JSON解析后的字典，如{"场所类型":["室内"], "时间":["白天"]}）  
    final_labels: list[str]  # 最终格式化标签（如["主体-人像", "性别-女"]，可选，由 format_output 生成）

    # messages: Annotated[list[AnyMessage], operator.add]  # 自动累加消息
    messages: list[AnyMessage]

    first_level_token_price: float  # 一级标签Token花费
    second_level_person_token_price: float  # 二级人像标签Token花费
    second_level_person_cloth_token_price: float  # 三级服饰标签Token花费
    second_level_pet_token_price: float  # 二级动物标签Token花费
    second_level_food_token_price: float  # 二级食物标签Token花费
    second_level_scenery_token_price: float  # 二级风景标签Token花费
    all_scene_type_token_price: float  # 场景类型标签Token花费
    total_tokens_price: float  # 总Token花费

    start_time: float          # 开始时间
    end_time: float            # 结束时间
    token_price_input: float    # 元/千Token（qwen2.5-vl-3b-instruct输入价格）   
    token_price_output: float   # 元/千Token（qwen2.5-vl-3b-instruct输出价格）

def first_level_classification(state: ImageTaggingState) -> ImageTaggingState:
    """一级分类：调用Qwen-VL确定核心主体（仅从一级分类中选择）"""
    start_time = time.time()
    print("开始时间：", state["start_time"])
    image_b64 = state["image_base64"]
    prompt = """
    任务：判断图片的核心主体，仅从以下一级分类的六个分类中选择（可以多选，不新增，如果都不含有就选其他）：
    一级分类列表：人像、动物（宠物）、植物、风景、食物（注意：冰饮、饮品、奶茶、酒之类喝的也属于食物标签）、建筑、其他
    一级分类列表：人像、动物（宠物）、植物、风景、食物（注意：冰饮、饮品、奶茶、酒之类喝的也属于食物标签）、建筑、其他
    注意：对于风景来说，如果图片中有积雪、飘雪、雪花、冰雕的话，可以加上标签“风景”; 如果图片中有蓝天白云、日落、星空、城市夜景等，也可以加上标签“风景”。
    输出要求：严格用JSON格式返回，key为一级分类类型（如“主体”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    如果一张图片同时含有了 人像，动物，风景。 请全部列出。
    错误示例（禁止）：{"主体":"人像","备注":"图片为人"}
    正确示例（必须遵循）：{"主体":["人像","动物（宠物）","风景"]}
    """
    logger.info("-----First_level_classification-----")
    all_response = model.call_qwen_vl(image_b64, prompt)
    first_level_token_price = (all_response["prompt_tokens"]/1000)*state["token_price_input"] + (all_response["completion_tokens"]/1000)*state["token_price_output"]
    # state["total_tokens"] = state.get("total_tokens", 0) + all_response.get("total_tokens", 0)
    state["messages"].append(HumanMessage(content=prompt))  # 记录prompt
    state["messages"].append(AIMessage(content=all_response["content"]))  # 记录模型输出
    try:
        clean_response = all_response["content"].strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        # logger.info(f"最终纯JSON字符串：{pure_json_str}")
        first_level_label = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        first_level_label = {}
    except Exception as e:
        logger.info(f"⚠️ 处理失败：{str(e)}")
        first_level_label = {}

    # 优化：确保 labels 字段符合 LabelState 结构
    # state["labels"]["first_level"] = first_level_label
    logger.info(f"一级分类标签：{first_level_label}")
    # 可选：记录消息到状态（如果需要交互历史）
    # new_messages = [HumanMessage(content=prompt), AIMessage(content=first_level_label)]
    
    return {"first_level": first_level_label,
            "first_level_token_price": first_level_token_price,
            "start_time": start_time
            }

def second_level_person(state: ImageTaggingState) -> ImageTaggingState:
    """二级分类：针对一级“人像”，提取细分标签（性别、年龄、构图等）"""
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])  # 获取一级主体标签列表
    
    if "人像" not in main_labels:
        logger.info("-----Second_level_person----- 一级标签人像，跳过人像细节")
        return 
    else:
        image_b64 = state["image_base64"]
        prompt = """
        任务：基于图片，提取“人像”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 性别：男性、女性
        - 年龄：儿童（0-10岁）、少年（11-18岁）、青年（19-35岁）、中年（36-59岁）、老年（60岁及以上）
        - 人数：单人、多人（≥2人，注意：画面背景中不是主体的背景人物也属于多人，也要选择多人这个标签）
        - 构图：自拍（含手臂/自拍杆痕迹或高角度近距离）、合影（多人同框且分布均匀）、正面（人脸对称）、侧面（单侧脸颊/眼睛占比大）、全身（画面需完整容纳人物的头顶至脚底（或脚尖处））、半身（人物在画面中占比处于 40%-90% 区间，画面裁切通常在人物的大腿中部至腰部之间（也可根据创作需求在臀部或膝盖上方））、面部特写（画面仅保留人物的头部或完整面部区域，人物主体在整张画面中的占比≤30%，其余部分为背景或留白；若包含少量颈部（不超过颈部 1/3 长度），且整体占比仍符合≤30% 的要求，也可归类为面部特写）
        - 用途：生活照（日常随拍）、证件照（背景为纯色无杂物的红/蓝/白标准证件背景板，人物为正面头部/肩部特写且居中，着装整洁正式、多为免冠无夸张饰品，光线均匀无明显阴影，整体为身份证/护照/毕业证等官方证件专用照片风格）、情侣照（画面中有明显情侣互动姿势，如牵手、拥抱、亲吻等，人物表情甜蜜幸福，构图多为近距离或半身合影，整体氛围浪漫温馨，符合情侣专属照片风格）
        - 饰品：眼镜（注意：黑色墨镜，近视镜，透明眼镜等等也属于眼镜，不要漏掉）、帽子、口罩、耳环、项链
        - 发型长度：长发（头发长度过肩，或垂落至背部、胸前，整体发长≥30cm）、短发（头发长度≤下巴，常见寸头、波波头、齐耳短发等，整体发长＜15cm）
        - 发型直卷：卷发（头发呈自然卷/烫卷形态，有明显波浪、螺旋或羊毛卷纹理，非拉直状态）、直发（头发整体顺直无明显卷曲，垂落形态顺滑，无卷度或仅有轻微弧度）
        - 发型形式：扎发（头发被束起固定，含马尾、丸子头、麻花辫、高颅顶束发、半扎发等形态，非完全散开）、披发（头发完全自然散开，无束起、绑扎的痕迹，整体呈垂落/蓬松散开状态）
        - 表情：微笑（嘴角上扬，露出牙齿或不露齿均可，整体面部表情愉悦）、大笑（哈哈大笑，笑的豁然开朗，漏出牙齿的笑）严肃（面部表情平静，无明显笑容，嘴唇紧闭或微张，眼神专注有神）、闭眼（双眼完全闭合，眼睑覆盖眼球，呈休息或睡眠状态）
        - 姿态：坐姿（人物以坐着的姿势出现，含椅子、地面、沙发等多种坐姿场景）、站立（人物以站立的姿势出现，含自然站立、摆拍等多种站姿场景）
        输出要求：严格用JSON格式返回，key为二级分类类型（如“性别”“年龄”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误格式示例（禁止格式）：{"性别":["女性"], "年龄":["成年（18-60岁）"], "人数":["单人", "自拍"], "备注":"图片为室内自拍"}
        正确格式示例（必须遵循的格式）：{"性别":["女性"], "年龄":["成年"], "人数":["单人"], "构图":["正面", "半身"], "用途":["生活照"], "饰品":["帽子"], "发型长度":["短发"], "发型直卷":["直发"], "发型形式":["披发"], "表情":["微笑"], "姿态":["站立"]}
        """
        response = model.call_qwen_vl_32b(image_b64, prompt)
        second_level_person_token_price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        logger.info("-----Second_level_person-----")
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        # new_messages = [HumanMessage(content=prompt), AIMessage(content=response)]              
        
        try:
            clean_response = response["content"].strip().replace("\n", "").replace(" ", "")
            if clean_response.startswith("```json"):
                pure_json_str = clean_response[7:]
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-3]
            else:
                pure_json_str = clean_response.lstrip("```").rstrip("```")
            # logger.info(f"最终纯JSON字符串：{pure_json_str}")
            second_level_person_dict = json.loads(pure_json_str)
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            second_level_person_dict = {}
        except Exception as e:
            logger.info(f"⚠️ 处理失败：{str(e)}")
            second_level_person_dict = {}
        logger.info(f"二级人像细节标签：{second_level_person_dict}")
        # state["second_level"] = second_level_dict
        return {"second_level_person": second_level_person_dict,
                "second_level_person_token_price": second_level_person_token_price
                }

def third_level_person_cloth(state: ImageTaggingState) -> ImageTaggingState:
    """三级服饰分类：针对一级“人像”，二级人像细节，进一步提取服饰标签（基本款式、风格/题材）"""
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])  # 获取一级主体标签列表
    
    if "人像" not in main_labels:
        logger.info("-----Second_level_person----- 一级标签无人像，跳过人像细节")
        return 
    else:
        image_b64 = state["image_base64"]
        prompt = """
        任务：基于图片，提取“人像”的服饰标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 服饰款式：西装、职业装、T恤、衬衫、毛衣、羽绒服、裙子、运动装、睡衣、校服、婚纱、泳装
        - 服饰题材：cosplay、lolita、jk、旗袍、新中式、民族服装、夏装、冬装、春秋装
        - 服饰风格：休闲风、街头风、正式风、学院风
        输出要求：严格用JSON格式返回，key为分类类型（如“基本款式”“题材”“风格”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误示例（禁止）：{"基本款式":["西装"], "题材":["新中式"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"基本款式":["西装","职业装"], "题材":["新中式"], "风格":["正式风"]}
        """
        response = model.call_qwen_vl(image_b64, prompt)
        second_level_person_cloth_token_price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        logger.info("-----third_level_person_cloth-----")
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        # new_messages = [HumanMessage(content=prompt), AIMessage(content=response)]              
        
        try:
            clean_response = response["content"].strip().replace("\n", "").replace(" ", "")
            if clean_response.startswith("```json"):
                pure_json_str = clean_response[7:]
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-3]
            else:
                pure_json_str = clean_response.lstrip("```").rstrip("```")
            # logger.info(f"最终纯JSON字符串：{pure_json_str}")
            second_level_person_cloth_dict = json.loads(pure_json_str)
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            second_level_person_cloth_dict = {}
        except Exception as e:
            logger.info(f"⚠️ 处理失败：{str(e)}")
            second_level_person_cloth_dict = {}
        
        logger.info(f"三级人像服饰标签：{second_level_person_cloth_dict}")
        return {"second_level_person_cloth": second_level_person_cloth_dict,
                "second_level_person_cloth_token_price": second_level_person_cloth_token_price
                }



def second_level_pet(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])  # 获取一级主体标签列表
    
    if "动物（宠物）" not in main_labels:
        logger.info("-----Second_level_pet----- 一级标签无动物（宠物），跳过动物分类")
        return 
    else:
        image_b64 = state["image_base64"]
        prompt = """
        任务：基于图片，提取“动物”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 种类：狗、猫、鸟、鱼、兔子、其他（注意只涉及这5种动物，不确定的话就选 其他）
        - 数量：单只、多只（≥2只宠物）
        - 视角与状态：宠物正面、宠物全身、室内宠物图、户外宠物图
        输出要求：严格用JSON格式返回，key为二级分类类型（如“种类”“数量”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        再次声明：仅限于上述5种动物（狗、猫、鸟、鱼、兔子）分类，不是这5中动物的其他种类。就选 其他。
        错误示例（禁止）：{"种类":["犀牛"], "数量":["单人"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"种类":["狗"], "数量":["单只"], "视角与状态":["宠物正面","室内宠物图"]}
        """
        response = model.call_qwen_vl(image_b64, prompt)
        second_level_pet_token_price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        logger.info("-----Second_level_pet-----")
        from langchain_core.messages import HumanMessage, AIMessage
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_response = response["content"].strip().replace("\n", "").replace(" ", "")
            if clean_response.startswith("```json"):
                pure_json_str = clean_response[7:]
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-3]
            else:
                pure_json_str = clean_response.lstrip("```").rstrip("```")
            # logger.info(f"最终纯JSON字符串：{pure_json_str}")
            second_level_pet_dict = json.loads(pure_json_str)
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            second_level_pet_dict = {}
        except Exception as e:
            logger.info(f"⚠️ 处理失败：{str(e)}")
            second_level_pet_dict = {}
        logger.info(f"二级动物细节标签：{second_level_pet_dict}")
        # state["labels"]["second_level"] = second_level_pet_dict
        return {"second_level_pet": second_level_pet_dict,
                "second_level_pet_token_price": second_level_pet_token_price
                }

def second_level_scenery(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])  # 获取一级主体标签列表
    
    if "风景" not in main_labels:
        logger.info("-----Second_level_pet----- 一级标签无风景，跳过风景细分分类")
        return 
    else:
        image_b64 = state["image_base64"]
        prompt = """
        任务：基于图片，提取“风景”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 地貌场景：海边、山脉、森林、草原、沙漠、瀑布、湖泊、花海、峡谷
        - 城市天空：天空（注意：如果含蓝天白云，一定要加上天空这个标签）、城市夜景、日落、星空
        - 季节相关：春季（含有樱花、桃花、梨花、嫩芽、柳树、蒲公英、油菜花、洋甘菊等）、夏季（含有荷花、荷叶、浓绿树荫、繁茂草丛、烈日、西瓜等，如果图片中有人穿短袖，泳装，或者戴太阳镜也可以判断为夏季）、秋季（含有枫叶、银杏、落叶、枯草、麦浪等）、冬季（首先可以根据如果图片中含有积雪、飘雪、冰雕、冰凌、雾凇、枯枝、梅花来判断，其次如果人像穿了羽绒服、冬季棉袄之类的都可以判定为冬季）
        输出要求：严格用JSON格式返回，key为二级分类类型（如“种类”“数量”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误示例（禁止）：{"地貌场景":["海边"], "城市天空":["天空"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"地貌场景":["海边"], "城市天空":["水面"], "季节相关":["春季"]}
        """
        response = model.call_qwen_vl(image_b64, prompt)
        second_level_scenery_token_price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        logger.info("-----Second_level_scenery-----")
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_response = response["content"].strip().replace("\n", "").replace(" ", "")
            if clean_response.startswith("```json"):
                pure_json_str = clean_response[7:]
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-3]
            else:
                pure_json_str = clean_response.lstrip("```").rstrip("```")
            # print("最终纯JSON字符串：", pure_json_str)
            second_level_scenery_dict = json.loads(pure_json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败：{str(e)}")
            second_level_scenery_dict = {}
        except Exception as e:
            print(f"⚠️ 处理失败：{str(e)}")
            second_level_scenery_dict = {}
        logger.info(f"二级风景细节标签：{second_level_scenery_dict}")
        # state["labels"]["second_level"] = second_level_dict
        return {"second_level_scenery": second_level_scenery_dict,
                "second_level_scenery_token_price": second_level_scenery_token_price
                }

def second_level_food(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    first_level = state["first_level"]
    main_labels = first_level.get("主体", [])  # 获取一级主体标签列表
    
    if "食物" not in main_labels:
        logger.info("-----Second_level_pet----- 一级标签无食物，跳过食物细分分类")
        return 
    else:
        image_b64 = state["image_base64"]
        prompt = """
        任务：基于图片，提取“食物细节”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
        - 食物类型：中餐、西餐、甜品、奶茶、火锅、水果、烧烤、主菜、小吃、饮品（）
        - 拍摄场景：桌面摆盘、俯拍、特写、居家烹饪、餐厅环境
        输出要求：严格用JSON格式返回，key为二级分类类型（如“食物类型”“拍摄场景”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
        错误示例（禁止）：{"食物类型":["饮品"], "拍摄场景":["单人"], "备注":"图片为室内自拍"}
        正确示例（必须遵循）：{"食物类型":["火锅"], "拍摄场景":["俯拍"]}
        """
        response = model.call_qwen_vl(image_b64, prompt)
        second_level_food_token_price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
        state["messages"].append(HumanMessage(content=prompt))
        state["messages"].append(AIMessage(content=response["content"]))
        
        try:
            clean_response = response["content"].strip().replace("\n", "").replace(" ", "")
            if clean_response.startswith("```json"):
                pure_json_str = clean_response[7:]
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-3]
            else:
                pure_json_str = clean_response.lstrip("```").rstrip("```")
            # logger.info(f"最终纯JSON字符串：{pure_json_str}")
            second_level_food_dict = json.loads(pure_json_str)
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ JSON解析失败：{str(e)}")
            second_level_food_dict = {}
        except Exception as e:
            logger.info(f"⚠️ 处理失败：{str(e)}")
            second_level_food_dict = {}
        
        # state["labels"]["second_level"] = second_level_dict
        return {"second_level_food": second_level_food_dict,
                "second_level_food_token_price": second_level_food_token_price
                }


def all_scene_type(state: ImageTaggingState) -> ImageTaggingState:
    image_b64 = state["image_base64"]
    prompt = """
    任务：基于图片，提取其场景类型，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
    - 场所类型：室内、室外、自然、家居、餐厅、健身房、游乐园、音乐节、KTV、演唱会
    - 时间：白天、夜晚
    - 天气：晴天、阴天、多云、雨天、雪天、雾天
    - 光线：彩虹、逆光
    输出要求：严格用JSON格式返回，key为二级分类类型（如“食物类型”“拍摄场景”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    错误示例（禁止）：{"场所类型":["室内"], "时间":["夜晚"], "天气":"彩虹", "备注":"图片为室内自拍"}
    正确示例（必须遵循）：{"场所类型":["室内","家居"], "时间":["白天"], "天气":["晴天"], "光线":["彩虹"]}
    """
    response = model.call_qwen_vl(image_b64, prompt)
    all_scene_type_token_price = (response["prompt_tokens"]/1000)*state["token_price_input"] + (response["completion_tokens"]/1000)*state["token_price_output"]
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response["content"]))
    
    try:
        clean_response = response["content"].strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        # logger.info(f"最终纯JSON字符串：{pure_json_str}")
        all_scene_dict = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        all_scene_dict = {}
    except Exception as e:
        logger.info(f"⚠️ 处理失败：{str(e)}")
        all_scene_dict = {}
    logger.info(f"场景类型标签：{all_scene_dict}")
    # end_time = time.time()
    # print("结束时间：", end_time)
    # state["labels"]["second_level"] = second_level_dict
    return {"all_scene_type": all_scene_dict,
            "all_scene_type_token_price": all_scene_type_token_price
            }


# def second_level_building(state: ImageTaggingState) -> ImageTaggingState:
#     # 逻辑不变，仅修改参数类型为 ImageTaggingState
#     image_b64 = state["image_base64"]
#     prompt = """任务：基于图片，提取“建筑”的二级标签..."""
#     response = model.call_qwen_vl(image_b64, prompt)
#     # 可选：记录消息
#     from langchain_core.messages import HumanMessage, AIMessage
#     state["messages"].append(HumanMessage(content=prompt))
#     state["messages"].append(AIMessage(content=response))
    
#     try:
#         clean_response = response.strip().replace("\n", "").replace(" ", "")
#         second_level_labels = json.loads(clean_response)
#     except:
#         second_level_labels = {}
#     state["labels"]["second_level"] = second_level_labels
#     return state

def route_by_first_level(state: ImageTaggingState) -> str:
    """路由逻辑：根据一级分类，返回对应的二级节点名称"""
    first_level = state["labels"]["first_level"]
    route_map = {
        "人像": "second_level_person",
        "动物（宠物）": "second_level_pet",
        "风景": "second_level_scenery",
        "食物": "second_level_food",
        "建筑": "second_level_building"
    }
    return route_map.get(first_level, "second_level_scenery")

# def format_output(state: ImageTaggingState) -> ImageTaggingState:
#     """将一级+二级标签整合为最终格式（如["主体-人像", "性别-女", "构图-自拍"]）"""
#     first_level = state["labels"]["first_level"]
#     second_level = state["labels"].get("second_level", {})
    
#     final_labels = [f"主体-{first_level}"]
#     for label_type, label_values in second_level.items():
#         for value in label_values:
#             final_labels.append(f"{label_type}-{value}")
    
#     state["final_labels"] = final_labels
#     return state
def format_output(state: ImageTaggingState) -> ImageTaggingState:
    """
    汇总所有标签到 final_labels：
    格式示例：
    ["主体-人像", "人像-性别-女", "人像-服饰-基本款式-西装", "动物（宠物）-种类-狗", "场景-场所类型-室内"]
    """
    final_labels = []
    end_time = time.time()
    # ========== 处理一级标签（主体） ==========
    first_level = state.get("first_level", {})  # 读取一级标签
    main_subjects = first_level.get("主体", [])  # 提取主体列表（如["人像", "动物（宠物）"]）
    for subject in main_subjects:
        final_labels.append(f"主体-{subject}")

    # ========== 2. 处理人像二级标签 ==========
    second_level_person = state.get("second_level_person", {})  # 读取人像二级标签
    for label_type, values in second_level_person.items():
        if not isinstance(values, list):  # 容错：非列表则跳过
            continue
        for value in values:
            final_labels.append(f"人像-{label_type}-{value}")

    # ========== 3. 处理人像服饰三级标签 ==========
    second_level_person_cloth = state.get("second_level_person_cloth", {})  # 读取人像服饰标签
    for label_type, values in second_level_person_cloth.items():
        if not isinstance(values, list):
            continue
        for value in values:
            final_labels.append(f"人像-服饰-{label_type}-{value}")

    # ========== 处理宠物二级标签 ==========
    second_level_pet = state.get("second_level_pet", {})  # 读取宠物二级标签
    for label_type, values in second_level_pet.items():
        if not isinstance(values, list):
            continue
        for value in values:
            final_labels.append(f"动物（宠物）-{label_type}-{value}")

    # ========== 处理食物二级标签 ==========
    second_level_food = state.get("second_level_food", {})  # 读取食物二级标签
    for label_type, values in second_level_food.items():
        if not isinstance(values, list):
            continue
        for value in values:
            final_labels.append(f"食物-{label_type}-{value}")

    # ========== 处理风景二级标签 ==========
    second_level_scenery = state.get("second_level_scenery", {})  # 读取风景二级标签
    for label_type, values in second_level_scenery.items():
        if not isinstance(values, list):
            continue
        for value in values:
            final_labels.append(f"风景-{label_type}-{value}")

    # ========== 处理场景类型标签 ==========
    all_scene_type = state.get("all_scene_type", {})  # 读取场景类型标签
    for label_type, values in all_scene_type.items():
        if not isinstance(values, list):
            continue
        for value in values:
            final_labels.append(f"场景-{label_type}-{value}")

    # ========== 6. 去重 + 排序（可选，提升可读性） ==========
    final_labels = sorted(list(set(final_labels)))

    # ========== 7. 写入最终状态 ==========
    # state["final_labels"] = final_labels
    return {"final_labels": final_labels,
            "end_time": end_time
            }

# 优化5：初始化 StateGraph 时，指定状态类型为 ImageTaggingState（而非 dict）
workflow = StateGraph(ImageTaggingState)

# 所有节点和边的定义不变，仅状态类型被规范
workflow.add_node("first_level_classification", first_level_classification)
workflow.add_node("second_level_person", second_level_person)
workflow.add_node("third_level_person_cloth", third_level_person_cloth)
workflow.add_node("second_level_pet", second_level_pet)
workflow.add_node("second_level_scenery", second_level_scenery)
workflow.add_node("second_level_food", second_level_food)
workflow.add_node("all_scene_type", all_scene_type)
# workflow.add_node("second_level_building", second_level_building)
workflow.add_node("format_output", format_output)

workflow.add_edge(START, "first_level_classification")
workflow.add_edge(START, "all_scene_type")
# workflow.add_conditional_edges(
#     "first_level_classification",
#     route_by_first_level,
#     {
#         "second_level_person": "second_level_person",
#         "second_level_food": "second_level_food",
#         "second_level_pet": "second_level_pet",
#         "second_level_scenery": "second_level_scenery",
#         "second_level_building": "second_level_building"
#     }
# )
# 添加并行边
workflow.add_edge("first_level_classification", "second_level_person")
workflow.add_edge("first_level_classification", "third_level_person_cloth")
workflow.add_edge("first_level_classification", "second_level_pet")
workflow.add_edge("first_level_classification", "second_level_scenery")
workflow.add_edge("first_level_classification", "second_level_food")
# workflow.add_edge("first_level_classification", "second_level_building")


workflow.add_edge("second_level_person", "format_output")
workflow.add_edge("third_level_person_cloth", "format_output")
workflow.add_edge("second_level_pet", "format_output")
workflow.add_edge("second_level_scenery", "format_output")
workflow.add_edge("second_level_food", "format_output")
workflow.add_edge("all_scene_type", "format_output")
workflow.add_edge("format_output", END)

app = workflow.compile()

# png_data = app.get_graph().draw_mermaid_png()
# with open("hitl_flowchart.png", "wb") as f:
#     f.write(png_data)

# ========== 单张图片处理函数（提取原有批量逻辑的单张处理部分） ==========
def process_single_image_api(img_path: str) -> dict:
    """处理单张图片，返回指定格式的结果字典"""
    try:
        # 1. 图片预处理：编码为base64
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片路径不存在：{img_path}")
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError(f"文件不是有效图片格式：{img_path}")
        img_b64 = encode_image(img_path)

        # 2. 初始化状态
        initial_state: ImageTaggingState = {
            "image_base64": img_b64,
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
            "start_time": time.time(),  # 初始化开始时间
            "end_time": 0.0,
            "token_price_input": 0.0012,
            "token_price_output": 0.0036
        }

        # 3. 调用langgraph状态图处理
        result = app.invoke(initial_state)

        # 4. 计算耗时和成本
        result["end_time"] = time.time()
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

        # 5. 构造返回结果（修正原代码重复的status字段）
        return {
            "image_path": img_path,
            "final_labels": result["final_labels"],
            "total_labels_count": len(result["final_labels"]),
            "elapsed_time": round(elapsed_time, 2),
            "token_cost": round(total_tokens_price, 4),
            "status": "success",
            "error": ""
        }

    except Exception as e:
        # 异常处理：返回失败状态
        error_msg = str(e)[:200]
        logger.error(f"处理失败 {img_path}: {error_msg}")
        return {
            "image_path": img_path,
            "final_labels": [],
            "total_labels_count": 0,
            "elapsed_time": 0.0,
            "token_cost": 0.0,
            "status": "failed",
            "error": error_msg
        }
import asyncio
# ========== POST接口定义 ==========
@fast_app.post("/process_image", response_description="单张图片标签处理结果")
async def api_process_image(request: ImagePathRequest):
    """
    单张图片标签提取接口
    - 输入：图片绝对路径
    - 输出：指定格式的标签结果字典
    """
    # 1. 校验请求参数
    img_path = request.image_path.strip()
    if not img_path:
        raise HTTPException(status_code=400, detail="图片路径不能为空")
    
    # 2. 调用单张处理函数
    result = await asyncio.to_thread(process_single_image_api, img_path)
    # result = process_single_image(img_path)
    
    # 3. 返回结果
    return result

# ========== 启动服务 ==========
if __name__ == "__main__":
    # 启动FastAPI服务，默认端口8000
    uvicorn.run(
        "image_understanding_api:fast_app",  # 注意：如果文件名不是main.py，需替换为实际文件名
        host="0.0.0.0",      # 允许外部访问
        port=8081
        # reload=True          # 开发模式自动重载
    )
# netstat -tulpn | grep :8081
# kill -9 <PID>
