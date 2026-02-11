import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import random
parent_dir = "/workspace/work/zhipeng16/git/yolo8-plus-iopaint"
sys.path.append(parent_dir)
from util.token_util_new import token_fresh

# 路径配置和导入语句之后添加
try:
    # 实例化 token_fresh 验证导入成功
    tf = token_fresh()
    print("导入 token_fresh 成功！")
except Exception as e:
    print(f"导入失败：{e}")


# 加载API Key
load_dotenv()


class CallVLMModel:
    """视觉语言模型调用，仅保留必要参数"""
    def __init__(self):
        # 初始化各模型的客户端
        self.qwen_client = OpenAI(
            api_key='sk-2421657025644998a802f1a0c29e4ec5',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.qwen_local_client0 = OpenAI(
            api_key="dummy_key",
            base_url="http://10.136.234.255:8000/v1",
        )
        self.qwen_local_client1 = OpenAI(
            api_key="dummy_key",
            base_url="http://10.136.234.255:8001/v1",
        )
        self.doubao_token_helper = token_fresh()

    # 自动判断图片类型，动态构建image_url的url值
    def is_http_https_url(self, s: str) -> bool:
        """辅助函数：判断字符串是否为HTTP/HTTPS开头的URL"""
        # 忽略大小写，判断是否以http://或https://开头
        return s.strip().lower().startswith(("http://", "https://"))

    # vllm本地部署的qwen2.5-vl-3b-instruct调用(8000端口/device=0,1/DP2)
    def call_qwen_local_vl0(self,image_content: str, prompt: str) -> str:
        """封装qwen2.5-vl-3b-instruct调用，保留必要参数"""
        if self.is_http_https_url(image_content):
            # 情况1：是HTTP/HTTPS URL，直接使用该URL
            image_url_value = image_content.strip()
        else:
            # 情况2：不是URL，默认视为Base64图片，拼接data URI格式
            image_url_value = f"data:image/jpeg;base64,{image_content.strip()}"
        completion = self.qwen_local_client0.chat.completions.create(
            model="/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url_value
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        # return completion.choices[0].message.content.strip()
        response_content = completion.choices[0].message.content.strip()
        # total_tokens = completion.usage.total_tokens if completion.usage else 0 
        prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0 
        completion_tokens = completion.usage.completion_tokens if completion.usage else 0 
        
        return {
            "content": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    # vllm本地部署的qwen2.5-vl-3b-instruct调用(8001端口-device=2,3-DP2)
    def call_qwen_local_vl1(self,image_content: str, prompt: str) -> str:
        """封装qwen2.5-vl-3b-instruct调用，保留必要参数"""

        if self.is_http_https_url(image_content):
            # 情况1：是HTTP/HTTPS URL，直接使用该URL
            image_url_value = image_content.strip()
        else:
            # 情况2：不是URL，默认视为Base64图片，拼接data URI格式
            image_url_value = f"data:image/jpeg;base64,{image_content.strip()}"

        completion = self.qwen_local_client1.chat.completions.create(
            model="/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url_value
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        # return completion.choices[0].message.content.strip()
        response_content = completion.choices[0].message.content.strip()
        # total_tokens = completion.usage.total_tokens if completion.usage else 0 
        prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0 
        completion_tokens = completion.usage.completion_tokens if completion.usage else 0 
        
        return {
            "content": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }




    def call_qwen_vl(self,base64_image: str, prompt: str) -> str:
        """封装qwen2.5-vl-3b-instruct调用，保留必要参数"""
        completion = self.qwen_client.chat.completions.create(
            model="qwen2.5-vl-3b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        # return completion.choices[0].message.content.strip()
        response_content = completion.choices[0].message.content.strip()
        # total_tokens = completion.usage.total_tokens if completion.usage else 0 
        prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0 
        completion_tokens = completion.usage.completion_tokens if completion.usage else 0 
        
        return {
            "content": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


    def call_qwen_new(self, image_content: str, prompt: str, schema: dict = None, service_index: int = 0) -> dict:
        """
        封装 Qwen3-VL-4B-Instruct 调用
        Args:
            image_content: 图片路径或Base64或URL
            prompt: 提示词
            schema: (可选) Pydantic生成的JSON Schema，用于强制结构化输出
            service_index: (可选) 指定服务节点索引，None 则随机
        """
        
        # 1. 选择客户端 (修复 bug: if service_index 会误判 0 为 False)
        service_index_list = [self.qwen_local_client0, self.qwen_local_client1]
        if service_index is not None and 0 <= service_index < len(service_index_list):
            client = service_index_list[service_index]
        else:
            client = random.choice(service_index_list)

        # 2. 处理图片格式 (关键修正)
        # vLLM/Qwen 对 base64 的要求：必须包含 "data:image/jpeg;base64," 前缀
        if self.is_http_https_url(image_content):
            image_url_value = image_content.strip()
        else:
            content_stripped = image_content.strip()
            # 自动补全 base64 前缀
            if not content_stripped.startswith("data:"):
                # 默认假设是 jpeg，如果是 png 可以改，但通常模型能自适应
                image_url_value = f"data:image/jpeg;base64,{content_stripped}"
            else:
                image_url_value = content_stripped

        # 3. 构造请求参数
        # 注意：不要硬编码模型路径，建议从 client 或 self.model_name 获取
        # 这里假设你 self 里存了模型名，或者你确认两个端口模型一致
        model_name = "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen3-VL-4B-Instruct"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url_value
                        }
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.1,  # 打标任务建议低温
            "max_tokens": 512,  # Qwen3 上下文更长，可以给多点防止截断
            "top_p": 0.95        # 增加一点点确定性
        }

        # 4. 结构化输出 (JSON Schema)
        # 你的写法是 OpenAI 格式，vLLM >= 0.6.0 完美支持
        # 但 Qwen3 有时对 `strict: True` 敏感，如果报错可以尝试去掉 strict
        if schema is not None:
            # request_kwargs["extra_body"] = {
            #      "guided_json": schema
            # }
            # 💡 备选方案：如果上面的 extra_body 不工作，再切回下面的 response_format
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tagging_result",
                    "schema": schema,
                    "strict": True # 如果报错，改为 False
                }
            }

        # 5. 发起调用
        try:
            completion = client.chat.completions.create(**request_kwargs)
            
            # 增加空值检查
            if not completion.choices:
                raise ValueError("模型返回了空的 choices 列表")

            response_content = completion.choices[0].message.content.strip()
            
            # 兼容 usage 为 None 的情况
            usage = getattr(completion, 'usage', None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            return {
                "content": response_content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            # 打印详细错误栈，方便排查是参数错还是网络错
            import traceback
            print(f"❌ 模型调用出错 (Service {service_index if service_index is not None else 'Random'}):")
            print(f"   Error: {e}")
            # traceback.print_exc() # 调试时打开
            return {
                "content": "{}", 
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e) # 把错误信息带出去
            }

    # 合并后的调用函数，支持指定服务节点(0或1)以及Schema约束
    def call_qwen_new2(self, image_content: str, prompt: str, schema: dict = None, service_index: int = 0) -> dict:
        """
        封装qwen2.5-vl-3b-instruct调用
        Args:
            image_content: 图片路径或Base64或URL
            prompt: 提示词
            schema: (新增) Pydantic生成的JSON Schema，用于强制结构化输出
            service_index: (新增) 服务节点索引，0对应8000端口，1对应8001端口
        """
        
        # 1. 选择客户端
        # 传index用第几个客户端没传的话随机选；负载均衡
        service_index_list = [self.qwen_local_client0, self.qwen_local_client1]
        if service_index != None:
            client = service_index_list[service_index]
        else:
            client = random.choice(service_index_list)

        # 2. 处理图片格式
        if self.is_http_https_url(image_content):
            # 情况1：是HTTP/HTTPS URL
            image_url_value = image_content.strip()
        else:
            # 情况2：不是URL，处理Base64
            # 如果传入的已经是 "data:image..." 格式，就不重复加前缀（兼容性处理）
            content_stripped = image_content.strip()
            if content_stripped.startswith("data:image"):
                image_url_value = content_stripped
            else:
                image_url_value = f"data:image/jpeg;base64,{content_stripped}"

        # 3. 构造请求参数
        request_kwargs = {
            "model": "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen3-VL-4B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url_value}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": 0.1,  # 建议加上温度控制，打标越低越好,
            "max_tokens": 512  # 限制最大生成长度（JSON标签通常不会超过512个token），卡死时会强制截断,
        }

        # 4. 【核心优化】如果传入了 schema，启用 Guided Decoding
        # if schema is not None:
        #     request_kwargs["extra_body"] = {"guided_json": schema}

        if schema is not None:
            request_kwargs["response_format"] = {
                "type": "json_schema", 
                "json_schema": {
                    "name": "result",        # 名字随便起
                    "schema": schema,        # 这里放入你的 Pydantic schema
                    "strict": True           # 强制严格模式
                }
            }
            # 注意：移除原来的 extra_body 代码

        # 5. 发起调用
        try:
            completion = client.chat.completions.create(**request_kwargs)
            
            response_content = completion.choices[0].message.content.strip()
            prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0 
            completion_tokens = completion.usage.completion_tokens if completion.usage else 0 
            
            return {
                "content": response_content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except Exception as e:
            # 简单的错误捕获，方便调试
            print(f"❌ 模型调用出错 (Service {service_index}): {e}")
            return {
                "content": "{}",  # 返回空JSON字符串防崩
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    def call_qwen_vl_32b(self,image_content: str, prompt: str) -> str:
        """封装qwen2.5-vl-7b-instruct调用，保留必要参数"""
        
        if self.is_http_https_url(image_content):
            # 情况1：是HTTP/HTTPS URL，直接使用该URL
            image_url_value = image_content.strip()
        else:
            # 情况2：不是URL，默认视为Base64图片，拼接data URI格式
            image_url_value = f"data:image/jpeg;base64,{image_content.strip()}"

        completion = self.qwen_client.chat.completions.create(
            model="qwen2.5-vl-32b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url_value
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        # return completion.choices[0].message.content.strip()
        response_content = completion.choices[0].message.content.strip()
        # total_tokens = completion.usage.total_tokens if completion.usage else 0 
        prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0 
        completion_tokens = completion.usage.completion_tokens if completion.usage else 0 
        
        return {
            "content": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


    def call_qwen(self, prompt: str) -> str:
        """封装qwen2.5调用，保留必要参数"""
        completion = self.qwen_client.chat.completions.create(
            model="qwen2.5-7b-instruct-1m",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return completion.choices[0].message.content.strip()
        # for chunk in completion:
        #     print(chunk.choices[0].delta.content, end="", flush=True)

    # def call_qwen_vl_max(self, base64_image: str, prompt: str) -> str:
    #     """
    #     封装 dashscope:qwen-vl-max 模型调用（接收Base64图片+文本prompt）
    #     :param base64_image: 图片的纯净Base64编码字符串（不含 "data:image/jpeg;base64," 前缀）
    #     :param prompt: 模型指令/提问文本（明确任务要求，不能为空）
    #     :return: 模型响应内容（字符串格式，异常时返回空字符串）
    #     """
    #     # 1. 参数合法性校验，避免无效调用
    #     if not isinstance(base64_image, str) or not base64_image.strip():
    #         self.log.error("传入的base64_image无效，不能为空或非字符串类型")
    #         return ""
    #     if not isinstance(prompt, str) or not prompt.strip():
    #         self.log.error("传入的prompt无效，不能为空或非字符串类型")
    #         return ""

    #     # 2. 构造Data URI格式（qwen-vl-max支持的图片传入格式）
    #     # 若图片为png格式，可将前缀改为 "data:image/png;base64,"
    #     img_data_uri = f"data:image/jpeg;base64,{base64_image.strip()}"

    #     # 3. 格式化图片为模型所需入参结构（复用现有工具方法）
    #     formatted_imgs = self.doubao_token_helper.img_to_model_ext(
    #         imgs=[img_data_uri],
    #         role=[],
    #         type="url"  # Data URI按url类型解析，模型可正常识别
    #     )

    #     # 4. 调用qwen-vl-max模型（复用现有call_model_zp方法，已适配qwen格式）
    #     model_response = self.doubao_token_helper.call_model_zp(
    #         user_prompt=prompt,
    #         system_prompt="",  # 无需系统提示词时保持为空
    #         model_type="dashscope:qwen-vl-max",  # 映射为dashscope:qwen-vl-max，更规范
    #         # 也可直接写 model_type="dashscope:qwen-vl-max"
    #         source="356732087",
    #         user_prompt_media=formatted_imgs  # 传入格式化后的图片数据
    #     )

    #     # 5. 处理响应结果（去首尾空格，确保返回格式整洁）
    #     final_response = model_response.strip() if model_response else ""
    #     return {
    #         "content": final_response.strip() if final_response else "",
    #         "prompt_tokens": 0,
    #         "completion_tokens": 0,
    #     }


    def call_doubao_vl(self, base64_image: str, prompt: str) -> dict:
        """
        封装Doubao-Seed-1.6-flash-250615
        :param local_img_path: 本地图片绝对路径
        :param prompt: 对图片的提问文本
        :return: 与其他方法格式一致的返回字典
        """
        # 格式化本地图片
        # formatted_imgs = self.doubao_token_helper.img_to_model_ext(
        #     imgs=[local_img_path],
        #     role=[],
        #     type='file'
        # )
        # 构造base64格式的图片URL（拼接Data URI前缀）
        img_data_uri = f"data:image/jpeg;base64,{base64_image}"
        
        # 格式化图片（type='url'，传入拼接好的Data URI）
        formatted_imgs = self.doubao_token_helper.img_to_model_ext(
            imgs=[img_data_uri],  # 传入Data URI格式的图片地址
            role=[],
            type='url'  # 改为url类型，识别Data URI格式
        )
        # print(formatted_imgs)
        # 调用Doubao视觉模型
        model_response = self.doubao_token_helper.call_model_zp(
            user_prompt=prompt,
            system_prompt="",
            model_type="volcengine:Doubao-Seed-1.6-flash-250615",
            source="356732087",
            user_prompt_media=formatted_imgs
        )
        # print("Doubao-1.5-vision-pro-250328模型响应：", model_response)
        print("Doubao-Seed-1.6-flash-250615模型响应：", model_response)
        # 保持与其他方法一致的返回格式（无token统计时设为0）
        return {
            "content": model_response.strip() if model_response else "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    
    def call_doubao_vision_lite_250315(self, base64_image: str, prompt: str) -> dict:
        """
        封装Doubao-1.5-vision-lite-250315（视觉轻量模型）
        :param base64_image: 图片Base64编码字符串（不含data URI前缀）
        :param prompt: 对图片的提问文本
        :return: 与其他方法格式一致的返回字典
        """
        # 构造base64格式的图片URL（拼接Data URI前缀）
        img_data_uri = f"data:image/jpeg;base64,{base64_image}"
        
        # 格式化图片（type='url'，传入拼接好的Data URI）
        formatted_imgs = self.doubao_token_helper.img_to_model_ext(
            imgs=[img_data_uri],  # 传入Data URI格式的图片地址
            role=[],
            type='url'  # 改为url类型，识别Data URI格式
        )
        
        # 调用Doubao视觉轻量模型
        model_response = self.doubao_token_helper.call_model_zp(
            user_prompt=prompt,
            system_prompt="",
            model_type="volcengine:Doubao-1.5-vision-lite-250315",  # 对应视觉轻量模型标识
            source="356732087",
            user_prompt_media=formatted_imgs
        )
        print("Doubao-1.5-vision-lite-250315模型响应：", model_response)
        
        # 保持与其他方法一致的返回格式（无token统计时设为0）
        return {
            "content": model_response.strip() if model_response else "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }