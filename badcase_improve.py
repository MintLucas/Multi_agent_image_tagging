#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2026/2/10 11:40
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: badcase_improve.py
# @Software: PyCharm
# @Usage:
import pandas as pd
import requests
import json
import os
import datetime
import uuid
import time


def retest_low_accuracy_tags(threshold, excel_path):
    """
    针对识别率低于指定阈值的标签，重新调用接口进行测试，并保存结果。

    Args:
        threshold (float): 识别率阈值 (0.0 - 1.0)，例如 0.8 代表 80%。
                           低于此值的标签对应的图片将被重测。
        excel_path (str): 之前生成的 Excel 文件路径。

    Returns:
        str: 输出的 JSON 文件路径或错误信息。
    """
    # 1. 读取 Excel 数据
    if not os.path.exists(excel_path):
        return f"错误：文件 {excel_path} 不存在"

    try:
        # 读取统计表，筛选低识别率标签
        df_stats = pd.read_excel(excel_path, sheet_name='识别率统计', engine='openpyxl')
        low_acc_tags = df_stats[df_stats['识别率'] <= threshold]['路径标签'].tolist()

        if not low_acc_tags:
            return "提示：没有识别率低于该阈值的标签，无需重测。"

        print(f"发现 {len(low_acc_tags)} 个标签识别率低于 {threshold:.0%}，准备重测...")

        # 读取原始数据表
        # 如果没有'原始数据'sheet，尝试读取第一个
        xls = pd.ExcelFile(excel_path, engine='openpyxl')
        sheet_name = '原始数据' if '原始数据' in xls.sheet_names else xls.sheet_names[0]
        df_data = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')

        # 筛选出需要重测的行
        target_rows = df_data[df_data['路径标签'].isin(low_acc_tags)]
        print(f"共筛选出 {len(target_rows)} 张图片需要重测。")

    except Exception as e:
        return f"读取 Excel 失败: {e}"

    # 2. 准备接口参数 [cite: 2, 3, 6]
    api_url = "http://10.136.234.255:8081/process_image"
    headers = {"Content-Type": "application/json"}

    results_list = []

    # 3. 循环调用接口
    for index, row in target_rows.iterrows():
        image_url = row.get('路径URL')
        image_path_str = row.get('路径名')

        # 构造请求数据
        # 优先使用 URL，如果没有 URL 则使用路径 (根据接口文档 image_info 说明 [cite: 6])
        image_info = image_url if pd.notna(image_url) and str(image_url).strip() else image_path_str

        # 生成唯一 task_id [cite: 6]
        task_id = str(uuid.uuid4())

        payload = {
            "image_info": image_info,
            "task_id": task_id
        }

        # 初始化结果条目 (保持与原始输入 JSON 结构一致，以便迭代分析)
        # 解析路径获取 except_tags (假设路径以 '/' 分隔，最后一部分是文件名)
        # 例如: "一级/二级/三级/image.jpg" -> ["一级", "二级", "三级"]
        path_parts = str(image_path_str).split('/')
        if len(path_parts) > 1:
            except_tags = path_parts[:-1]  # 去掉文件名
        else:
            except_tags = []

        result_entry = {
            "image_name": os.path.basename(str(image_path_str)),
            "image_path": image_path_str,
            "image_url": image_url,
            "except_tags": except_tags,
            "is_matched": False,  # 默认为 False，等待后续分析脚本计算
            "process_result": {}
        }

        try:
            # 发送 POST 请求 [cite: 3, 23]
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                api_response = response.json()
                # 接口返回的 format 包含 code, result 等 [cite: 36, 38]
                # 我们主要需要 result 部分放入 process_result
                if 'result' in api_response:
                    result_entry['process_result'] = api_response['result']
                else:
                    # 如果直接返回了结果字典，视情况兼容
                    result_entry['process_result'] = api_response
            else:
                print(f"请求失败 [{response.status_code}]: {image_path_str}")
                result_entry['error'] = f"HTTP {response.status_code}"

        except Exception as e:
            print(f"请求异常: {image_path_str} - {e}")
            result_entry['error'] = str(e)

        results_list.append(result_entry)

        # 简单的进度打印
        if len(results_list) % 10 == 0:
            print(f"已处理 {len(results_list)}/{len(target_rows)}...")

        # 避免请求过快 (可选)
        time.sleep(0.05)

    # 4. 保存结果到 JSON
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    output_filename = f"images_result_with_labels_{today_str}_match_result.json"

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=4)
        return f"成功！重测结果已保存为: {output_filename} (共 {len(results_list)} 条数据)"
    except Exception as e:
        return f"保存 JSON 失败: {e}"



# --- 使用示例 ---
if __name__ == "__main__":
    # 参数1: 阈值 (例如 0.5 代表 50% 以下的标签需要重测)
    # 参数2: Excel 文件路径
    input_excel = 'images_result_with_labels_20260129_match_result_processed.xlsx'

    # 请在您的本地环境中取消注释并运行以下行：
    res = retest_low_accuracy_tags(0.0, input_excel)
    print(res)