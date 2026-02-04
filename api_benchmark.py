"""
运行命令：
locust -f api_benchmark.py  --host http://127.0.0.1:14547  --headless -u 300 -r 10 -t 60s
"""

import os
import random
import uuid
from locust import HttpUser, task, between


image_folder = '/workspace/work/zhipeng16/git/Multi_agent_image_tagging/无他图片标签测试图'
image_paths = []
for root, _, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))

class User(HttpUser):
    wait_time = between(1, 1.5)

    @task
    def task_post_archive(self):
        # trace_id = f'cevi{uuid.uuid4().hex}'
        testServer = 'http://localhost:14547'
        path = '/process_image_local'
        url = f'{testServer}{path}'
        headers = {'Content-Type': 'application/json'}
        data = {"image_path": random.choice(image_paths)}
        self.client.post(url, json=data, headers=headers)