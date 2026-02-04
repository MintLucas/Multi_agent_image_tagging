import base64
import json
from PIL import Image
from io import BytesIO
from model import CallVLMModel
from utils import encode_image
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage, AIMessage
from logger import get_logger


# 构建日志记录器
logger = get_logger(service="lg_builder")
model = CallVLMModel()

class LabelState(TypedDict):
    """嵌套状态：存储一/二级标签"""
    first_level: str  # 一级标签（如"人像"、"食物"）
    second_level: dict  # 二级标签（JSON解析后的字典，如{"性别":["女"], "构图":["自拍"]}）

class ImageTaggingState(TypedDict):
    """图片标签工具的状态定义"""
    image_base64: str  # 图片Base64编码（必需）
    labels: LabelState  # 一/二级标签（嵌套状态，必需）
    final_labels: list[str]  # 最终格式化标签（如["主体-人像", "性别-女"]，可选，由 format_output 生成）
    # messages: Annotated[list[AnyMessage], operator.add]  # 自动累加消息
    messages: list[AnyMessage]

def first_level_classification(state: ImageTaggingState) -> ImageTaggingState:
    """一级分类：调用Qwen-VL确定核心主体（仅从一级分类中选择）"""
    image_b64 = state["image_base64"]
    prompt = """
    任务：判断图片的核心主体，仅从以下一级分类的六个分类中选择1个（必须选，不新增）：
    一级分类列表：人像、动物（宠物）、植物、风景、食物、建筑、其他
    输出要求：仅返回分类名称（如“人像”“食物”），不添加任何额外解释。如果图片中没有明确的主体，就选 其他。
    """
    logger.info("-----First_level_classification-----")
    first_level_label = model.call_qwen_vl(image_b64, prompt)
    
    # 优化：确保 labels 字段符合 LabelState 结构
    state["labels"]["first_level"] = first_level_label
    # 可选：记录消息到状态（如果需要交互历史）
    state["messages"].append(HumanMessage(content=prompt))  # 记录prompt
    state["messages"].append(AIMessage(content=first_level_label))  # 记录模型输出
    # new_messages = [HumanMessage(content=prompt), AIMessage(content=first_level_label)]
    
    return state

def second_level_person(state: ImageTaggingState) -> ImageTaggingState:
    """二级分类：针对一级“人像”，提取细分标签（性别、年龄、构图等）"""
    image_b64 = state["image_base64"]
    prompt = """
    任务：基于图片，提取“人像”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
    - 性别：男性、女性
    - 年龄：儿童（0-12岁）、青少年（13-18岁）、成年（18-60岁）、老人（≥60岁）
    - 人数：单人、多人（≥2人）
    - 构图：自拍（含手臂/自拍杆痕迹或高角度近距离）、合影（多人同框且分布均匀）、正面（人脸对称）、侧面（单侧脸颊/眼睛占比大）、全身（≥80%）、半身（人物占画面40%-80%）、面部特写（仅头部/面部，≤30%）、证件照（带证件照背景）、情侣
    - 饰品：眼镜、口罩、耳环、项链
    - 发型：长发、短发、卷发、直发、染发、扎发、披发
    输出要求：严格用JSON格式返回，key为二级分类类型（如“性别”“构图”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    错误示例（禁止）：{"性别":["女性"], "人数":["单人", "自拍"], "备注":"图片为室内自拍"}
    正确示例（必须遵循）：{"性别":["女性"], "年龄":["成年"], "人数":["单人"], "构图":["正面", "半身"], "饰品":["帽子"], "发型":["短发"]}
    """
    response = model.call_qwen_vl(image_b64, prompt)
    logger.info("-----Second_level_person-----")
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response))
    # new_messages = [HumanMessage(content=prompt), AIMessage(content=response)]              
    
    try:
        clean_response = response.strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        logger.info(f"最终纯JSON字符串：{pure_json_str}")
        second_level_dict = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        second_level_dict = {}
    except Exception as e:
        logger.info(f"⚠️ 处理失败：{str(e)}")
        second_level_dict = {}
    
    state["labels"]["second_level"] = second_level_dict
    return state


def second_level_person_cloth(state: ImageTaggingState) -> ImageTaggingState:
    """二级分类：针对一级“人像”，提取细分标签（性别、年龄、构图等）"""
    image_b64 = state["image_base64"]
    prompt = """
    任务：基于图片，提取“人像”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
    - 基本款式：西装、职业装、T恤、衬衫、毛衣、羽绒服、裙子、运动装、睡衣、校服、婚纱、泳装
    - 题材：cosplay、lolita、jk、旗袍、新中式、民族服装、夏装、冬装、春秋装
    - 风格：休闲风、街头风、正式风、学院风
    输出要求：严格用JSON格式返回，key为二级分类类型（如“基本款式”“题材”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    错误示例（禁止）：{"基本款式":["西装"], "题材":["cosplay"], "备注":"图片为室内自拍"}
    正确示例（必须遵循）：{"基本款式":["西装"], "题材":["新中式"], "风格":["休闲风"]}
    """
    response = model.call_qwen_vl(image_b64, prompt)
    logger.info("-----Second_level_person-----")
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response))
    # new_messages = [HumanMessage(content=prompt), AIMessage(content=response)]              
    
    try:
        clean_response = response.strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        logger.info(f"最终纯JSON字符串：{pure_json_str}")
        second_level_dict = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        second_level_dict = {}
    except Exception as e:
        logger.info(f"⚠️ 处理失败：{str(e)}")
        second_level_dict = {}
    
    state["labels"]["second_level"] = second_level_dict
    return state

def second_level_pet(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    image_b64 = state["image_base64"]
    prompt = """
    任务：基于图片，提取“动物”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
    - 种类：狗、猫、鸟、鱼、兔子、其他（注意只涉及这5种动物，不确定的话就选 其他）
    - 数量：单只、多只（≥2只宠物）
    - 视角：宠物正面、宠物全身
    输出要求：严格用JSON格式返回，key为二级分类类型（如“种类”“数量”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    再次声明：仅限于上述5种动物（狗、猫、鸟、鱼、兔子）分类，不是这5中动物的其他种类。就选 其他。
    再次声明：仅限于上述5种动物（狗、猫、鸟、鱼、兔子）分类，不是这5中动物的其他种类。就选 其他。
    错误示例（禁止）：{"种类":["犀牛"], "数量":["单人"], "备注":"图片为室内自拍"}
    正确示例（必须遵循）：{"种类":["狗"], "数量":["单只"], "视角":["宠物正面"]}
    """
    response = model.call_qwen_vl(image_b64, prompt)
    logger.info("-----Second_level_pet-----")
    from langchain_core.messages import HumanMessage, AIMessage
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response))
    
    try:
        clean_response = response.strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        logger.info(f"最终纯JSON字符串：{pure_json_str}")
        second_level_dict = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        second_level_dict = {}
    except Exception as e:
        logger.info(f"⚠️ 处理失败：{str(e)}")
        second_level_dict = {}
    
    state["labels"]["second_level"] = second_level_dict
    return state

def second_level_scenery(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    image_b64 = state["image_base64"]
    prompt = """
    任务：基于图片，提取“风景”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
    - 地貌场景：海边、山脉、森林、草原、沙漠、瀑布、湖泊、花海、峡谷
    - 城市天空：天空、水面、城市夜景、建筑群、桥梁、日落、星空
    - 季节相关：春季（含有樱花、桃花、梨花、嫩芽、柳絮、蒲公英、油菜花等）、夏季（含有荷花、荷叶、浓绿树荫、繁茂草丛、烈日等）、秋季（含有枫叶、银杏、落叶、枯草、麦浪等）、冬季（含有积雪、飘雪、冰凌、雾凇、枯枝、梅花等）
    输出要求：严格用JSON格式返回，key为二级分类类型（如“种类”“数量”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    错误示例（禁止）：{"地貌场景":["海边"], "城市天空":["天空"], "备注":"图片为室内自拍"}
    正确示例（必须遵循）：{"地貌场景":["海边"], "城市天空":["水面"], "季节相关":["春季"]}
    """
    response = model.call_qwen_vl(image_b64, prompt)
    logger.info("-----Second_level_scenery-----")
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response))
    
    try:
        clean_response = response.strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        print("最终纯JSON字符串：", pure_json_str)
        second_level_dict = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析失败：{str(e)}")
        second_level_dict = {}
    except Exception as e:
        print(f"⚠️ 处理失败：{str(e)}")
        second_level_dict = {}
    state["labels"]["second_level"] = second_level_dict
    return state

def second_level_food(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    image_b64 = state["image_base64"]
    prompt = """
    任务：基于图片，提取“食物细节”的二级标签，仅从以下预设选项中选择（可多选，不确定的标签坚决不选）：
    - 食物类型：中餐、西餐、甜品、奶茶、火锅、水果、烧烤、主菜、小吃、饮品
    - 拍摄场景：桌面摆盘、俯拍、特写、居家烹饪、餐厅环境
    输出要求：严格用JSON格式返回，key为二级分类类型（如“食物类型”“拍摄场景”），value为标签列表（空列表不显示），不添加任何额外文字、解释或标点。
    错误示例（禁止）：{"食物类型":["饮品"], "拍摄场景":["单人"], "备注":"图片为室内自拍"}
    正确示例（必须遵循）：{"食物类型":["火锅"], "拍摄场景":["俯拍"]}
    """
    response = model.call_qwen_vl(image_b64, prompt)
    # 可选：记录消息
    from langchain_core.messages import HumanMessage, AIMessage
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response))
    
    try:
        clean_response = response.strip().replace("\n", "").replace(" ", "")
        if clean_response.startswith("```json"):
            pure_json_str = clean_response[7:]
            if pure_json_str.endswith("```"):
                pure_json_str = pure_json_str[:-3]
        else:
            pure_json_str = clean_response.lstrip("```").rstrip("```")
        logger.info(f"最终纯JSON字符串：{pure_json_str}")
        second_level_dict = json.loads(pure_json_str)
    except json.JSONDecodeError as e:
        logger.info(f"⚠️ JSON解析失败：{str(e)}")
        second_level_dict = {}
    except Exception as e:
        logger.info(f"⚠️ 处理失败：{str(e)}")
        second_level_dict = {}
    
    state["labels"]["second_level"] = second_level_dict
    return state


def second_level_building(state: ImageTaggingState) -> ImageTaggingState:
    # 逻辑不变，仅修改参数类型为 ImageTaggingState
    image_b64 = state["image_base64"]
    prompt = """任务：基于图片，提取“建筑”的二级标签..."""
    response = model.call_qwen_vl(image_b64, prompt)
    # 可选：记录消息
    from langchain_core.messages import HumanMessage, AIMessage
    state["messages"].append(HumanMessage(content=prompt))
    state["messages"].append(AIMessage(content=response))
    
    try:
        clean_response = response.strip().replace("\n", "").replace(" ", "")
        second_level_labels = json.loads(clean_response)
    except:
        second_level_labels = {}
    state["labels"]["second_level"] = second_level_labels
    return state

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

def format_output(state: ImageTaggingState) -> ImageTaggingState:
    """将一级+二级标签整合为最终格式（如["主体-人像", "性别-女", "构图-自拍"]）"""
    first_level = state["labels"]["first_level"]
    second_level = state["labels"].get("second_level", {})
    
    final_labels = [f"主体-{first_level}"]
    for label_type, label_values in second_level.items():
        for value in label_values:
            final_labels.append(f"{label_type}-{value}")
    
    state["final_labels"] = final_labels
    return state

# 优化5：初始化 StateGraph 时，指定状态类型为 ImageTaggingState（而非 dict）
workflow = StateGraph(ImageTaggingState)

# 所有节点和边的定义不变，仅状态类型被规范
workflow.add_node("first_level_classification", first_level_classification)
workflow.add_node("second_level_person", second_level_person)
workflow.add_node("second_level_pet", second_level_pet)
workflow.add_node("second_level_scenery", second_level_scenery)
workflow.add_node("second_level_food", second_level_food)
workflow.add_node("second_level_building", second_level_building)
workflow.add_node("format_output", format_output)

workflow.add_edge(START, "first_level_classification")
workflow.add_conditional_edges(
    "first_level_classification",
    route_by_first_level,
    {
        "second_level_person": "second_level_person",
        "second_level_food": "second_level_food",
        "second_level_pet": "second_level_pet",
        "second_level_scenery": "second_level_scenery",
        "second_level_building": "second_level_building"
    }
)
workflow.add_edge("second_level_person", "format_output")
workflow.add_edge("second_level_pet", "format_output")
workflow.add_edge("second_level_scenery", "format_output")
workflow.add_edge("second_level_food", "format_output")
workflow.add_edge("second_level_building", "format_output")
workflow.add_edge("format_output", END)

app = workflow.compile()

png_data = app.get_graph().draw_mermaid_png()
with open("hitl_flowchart.png", "wb") as f:
    f.write(png_data)

if __name__ == "__main__":
    print("🎉 图片标签生成工具启动！")
    image_path = '/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图/5、风景细节/5.3 季节相关/2、夏季/68b8ed553cb11d4fb0434d76808cd44.jpg'
    choice = encode_image(image_path)
    
    # 优化6：初始状态必须符合 ImageTaggingState 结构（所有必填字段初始化）
    initial_state: ImageTaggingState = {
        "image_base64": choice,
        "labels": {"first_level": "", "second_level": {}},  # 符合 LabelState 嵌套结构
        "final_labels": [],  # 初始为空，由 format_output 填充
        "messages": []  # 初始为空消息列表（如果添加了 messages 字段）
    }
    
    result = app.invoke(initial_state)
    
    print("\n🎯 最终生成的图片标签：")
    for i, label in enumerate(result["final_labels"]):
        print(f"  {i}. {label}")
    
    # # 可选：打印交互消息历史（如果需要调试）
    # print("\n📝 交互消息历史：")
    # for msg in result["messages"]:
    #     msg.pretty_print()