import requests
import time

# 配置接口地址和测试图片路径
# API_URL = "http://localhost:14547/process_image_local1"
API_URL = "http://10.136.234.255:14547/process_image_local"
TEST_IMG_PATH = "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图/3、宠物细节/3.3 视角与状态/3、室内宠物图/17c6204b4b60209b603c63a5978498e.jpg"  # 替换为实际路径

# 核心请求逻辑
if __name__ == "__main__":
    start_time = time.time()
    # 构造请求数据
    data = {"image_path": TEST_IMG_PATH, "task_id": "djakfjl"}
    
    try:
        # 发送POST请求
        res = requests.post(API_URL, json=data, timeout=300)
        res.raise_for_status()  # 捕获HTTP错误
        end_time = time.time()
        print("请求耗时：{:.2f}秒".format(end_time - start_time))
        # 打印结果
        print("接口返回结果：")
        print(res.json())
        
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到API服务，请检查服务是否启动")
    except Exception as e:
        print(f"请求失败：{str(e)}")


# curl -X POST 'http://10.136.234.255:14547/process_image_local0' 
# -H 'Content-Type: application/json' \
# -d '{
#         "image_path": "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图/2、人像细节/2.1 性别与年龄/1、男性/d431919d98ad87ac155fe798d45b138.jpg"
#         }'