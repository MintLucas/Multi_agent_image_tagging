#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/02/05 11:32
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : schemas.py
# @Usage   : Describe the file's purpose
from pydantic import BaseModel, Field
from typing import List, Literal

# ==========================================
# 1. 一级分类 Schema
# ==========================================
class FirstLevelSchema(BaseModel):
    主体: List[Literal["人像", "动物（宠物）", "植物", "风景", "食物", "建筑", "其他"]] = Field(
        description="图片核心主体判断。注意：花朵/花海/植物属于'风景'；饮品/酒属于'食物'；含积雪/蓝天/夜景均可算'风景'。"
    )

# ==========================================
# 2. 人像二级细节 Schema
# ==========================================
class PortraitDetailsSchema(BaseModel):
    # 核心字段：强制必填 (移除 default=[])，配合详细定义
    性别: List[Literal["男性", "女性"]] = Field(description="性别判定")
    
    年龄: List[Literal["儿童", "少年", "青年", "中年", "老年"]] = Field(
        description="严格年龄段：儿童(0-10岁)、少年(11-18岁)、青年(19-35岁)、中年(36-59岁)、老年(60岁+)"
    )
    
    人数: List[Literal["单人", "多人"]] = Field(
        description="单人（画面里仅一个人物），多人（画面里出现≥2 个人物）"
    )
    
    拍摄方式: List[Literal["自拍", "他拍", "合影"]] = Field(
            description="判断是谁拍的。自拍(看到手机/手臂延伸)；合影(刻意多人配合)；他拍(常规拍摄)。"
        )
    
    构图: List[Literal["全身", "半身", "面部特写"]] = Field(
        description="判断画面涵盖范围。全身(头到脚)；半身(腰/腿以上)；面部特写(仅头颈)；。"
    )
    
    角度: List[Literal["正面", "侧面", "背影"]] = Field(
        description="人物相对于镜头的朝向。"
    )
    用途: List[Literal["生活照", "证件照", "情侣照"]] = Field(
        description="用途：生活照(日常)；证件照(纯色背景/无夸张饰品/正面)；情侣照(亲密互动/甜蜜氛围)。"
    )
    
    发型长度: List[Literal["长发", "短发"]] = Field(
        description="发长：长发(过肩/≥30cm)；短发(≤下巴/寸头/波波头)。"
    )
    
    发型直卷: List[Literal["卷发", "直发"]] = Field(
        description="发质：卷发(自然卷/烫卷/波浪)；直发(顺直无明显卷曲)。"
    )
    
    发型形式: List[Literal["扎发", "披发"]] = Field(
        description="形式：扎发(马尾/丸子头/辫子)；披发(自然散开)。"
    )
    
    表情: List[Literal["微笑", "大笑", "严肃", "闭眼"]] = Field(
        description="表情：微笑(嘴角上扬)；大笑(露齿/开朗)；严肃(无笑容/专注)；闭眼(休息/睡眠)。"
    )
    
    姿态: List[Literal["坐姿", "站立"]] = Field(
        description="姿态：坐姿(椅子/地面/沙发)；站立(自然站立/摆拍)。"
    )


# ==========================================
# 3. 人像服饰三级 Schema
# ==========================================
# class ClothingDetailsSchema(BaseModel):
#     # 1. 字段名必须与 Prompt 和 白名单 保持高度一致
#     # 2. 增强 description，告诉模型不要偷懒
#     基本款式: List[Literal["西装", "职业装", "T恤", "衬衫", "毛衣", "羽绒服", "裙子", "运动装", "睡衣", "校服", "婚纱", "泳装"]] = Field(
#         default=[], 
#         description="【强制扫描全身】：仔细观察人物上装和下装。必须识别出至少一种主要服装类型（如穿了裙子必选'裙子'，穿了西服必选'西装'）。"
#     )
    
#     题材: List[Literal["cosplay", "lolita", "jk", "旗袍", "新中式", "民族服装", "夏装", "冬装", "春秋装"]] = Field(
#         default=[], 
#         description="【风格识别】：判断服装的特定题材。特别是Lolita、JK、Cosplay、汉服等特色服饰，特征明显时必须检出。"
#     )
    
#     风格: List[Literal["休闲风", "街头风", "正式风", "学院风"]] = Field(
#         default=[], 
#         description="整体穿搭氛围判断。"
#     )

#     # 3. 将眼镜放在最后，作为补充细节
#     眼镜: List[Literal["眼镜", "否"]] = Field(
#         description="【面部细节】：是否佩戴眼镜/墨镜。",
#         max_items=1
#     )

class ClothingDetailsSchema(BaseModel):
    基本款式: List[Literal["西装", "职业装", "T恤", "衬衫", "毛衣", "羽绒服", "裙子", "运动装", "睡衣", "校服", "婚纱", "泳装"]] = Field(
        default=[], 
        description="【强制扫描全身】：仔细观察人物上装和下装。必须识别出至少一种主要服装类型（如穿了裙子必选'裙子'，穿了西服必选'西装'）。"
    )
    
    题材: List[Literal["cosplay", "lolita", "jk", "旗袍", "新中式", "民族服装", "夏装", "冬装", "春秋装"]] = Field(
        default=[], 
        description="【风格识别】：判断服装的特定题材。特别是Lolita、JK、Cosplay、汉服等特色服饰，特征明显时必须检出。"
    )
    
    风格: List[Literal["休闲风", "街头风", "正式风", "学院风"]] = Field(
        default=[], 
        description="整体穿搭氛围判断。"
    )
    
    饰品: List[Literal["帽子", "口罩", "耳环", "项链", "发饰", "围巾"]] = Field(
        default=[], 
        description="【配饰扫描】：检查头部(帽子/发饰)、面部(口罩)、耳部(耳环)、颈部(项链/围巾)。"
    )
    
    眼镜: List[Literal["眼镜", "否"]] = Field(
        description="【面部细节】：是否佩戴眼镜/墨镜。必须二选一。",
        max_items=1
    )

# ==========================================
# 4. 动物（宠物）Schema
# ==========================================
class PetDetailsSchema(BaseModel):
    种类: List[Literal["狗", "猫", "鸟", "鱼", "兔子", "其他"]] = Field(
        description="动物种类：仅限狗、猫、鸟、鱼、兔子，其他动物选'其他'。"
    )
    数量: List[Literal["单只", "多只"]] = Field(description="数量判定")
    视角与状态: List[Literal["宠物正面", "宠物全身", "室内宠物图", "户外宠物图"]] = Field(
        default=[], description="拍摄视角与场景状态"
    )

# ==========================================
# 5. 食物 Schema
# ==========================================
class FoodDetailsSchema(BaseModel):
    食物类型: List[Literal["中餐", "西餐", "甜品", "奶茶", "火锅", "水果", "烧烤", "主菜", "小吃", "饮品"]] = Field(
        description="食物具体类别"
    )
    拍摄场景: List[Literal["桌面摆盘", "俯拍", "特写", "居家烹饪", "餐厅环境"]] = Field(
        default=[], description="食物的拍摄角度或环境"
    )

# ==========================================
# 6. 风景 Schema
# ==========================================
class SceneryDetailsSchema(BaseModel):
    地貌场景: List[Literal["海边", "山脉", "森林", "草原", "沙漠", "瀑布", "湖泊", "花海", "峡谷"]] = Field(
        default=[], description="自然地貌类型"
    )
    城市天空: List[Literal["天空", "城市夜景", "日落", "星空"]] = Field(
        default=[], description="天空或城市景观。注意：含蓝天白云必选'天空'。"
    )
    季节相关: List[Literal["春季", "夏季", "秋季", "冬季"]] = Field(
        default=[], description="季节特征：春季(花/嫩芽)；夏季(荷花/浓荫/烈日/短袖)；秋季(枫叶/落叶/金黄)；冬季(雪/冰/枯枝/羽绒服)。"
    )

# ==========================================
# 7. 全局场景类型 Schema
# ==========================================
class SceneTypeSchema(BaseModel):
    空间: List[Literal["室内", "室外"]] = Field(
        description="物理空间属性。室内(有屋顶/墙壁)；室外(露天环境)。"
    )
    场所类型: List[Literal[ "自然", "家居", "餐厅", "健身房", "游乐园", "音乐节", "KTV", "演唱会"]] = Field(
        default=[], description="所处空间类型"
    )
    时间: List[Literal["白天", "夜晚"]] = Field(
        default=[], description="白天(自然光) vs 夜晚(人工照明)"
    )
    天气: List[Literal["晴天", "阴天", "多云", "雨天", "雪天", "雾天", "彩虹"]] = Field(
        default=[], description="天气状况"
    )
    光线: List[Literal["自然光", "逆光"]] = Field(
        default=[], description="特殊光影"
    )
    特殊元素: List[Literal["烟花", "圣诞树", "气球", "彩带", "蛋糕", "粽子", "元宵", "月饼", "礼物盒"]] = Field(
        default=[], description="画面中明确存在的实体物品（排除印刷图案）"
    )
    # 2. 新增：独立水印字段（强制二选一）
    水印: List[Literal["水印", "无水印"]] = Field(
        description="【强制检查】：扫描画面四角及中央。只要发现文字水印、Logo、时间戳或防盗纹，必须选'水印'；完全干净选'无水印'。不可为空。"
    )
    图片质量: List[Literal["无路人", "有路人", "老照片"]] = Field(
        default=[], description="画面质量与干扰因素：无路人(仅主体)；有路人(画面中有除主体外的其他人物)；老照片(泛黄/年代感)。"
    )
    节日: List[Literal["生日", "婚礼", "圣诞", "春节", "中秋", "端午", "万圣节", "国庆"]] = Field(
        default=[], description="明显的节日氛围元素"
    )