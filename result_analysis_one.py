import pandas as pd
import requests
import json
import os
import time
import uuid
import datetime
import numpy as np
import matplotlib.pyplot as plt
import re  # <--- [新增] 引入正则模块

class ImageTagPipeline:
    def __init__(self, api_url="http://49.7.36.149:80/process_image_local"):
        self.api_url = api_url
        # 设置绘图风格和字体（优先兼容 Mac）
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    # --- [辅助方法] 清洗标签 ---
    def _clean_tags(self, path_parts):
        """
        输入: ['10、节日与活动', '10.1 节日', '1、生日']
        输出: ['节日与活动', '节日', '生日']
        """
        cleaned = []
        for part in path_parts:
            # 正则替换：去掉开头的数字、点、空格、顿号
            new_part = re.sub(r'^[\d\.\s、]+', '', part)
            cleaned.append(new_part)
        return cleaned
    # =========================================================================
    # 核心功能 1: JSON -> Excel (包含 3 个 Sheet: 原始数据, 统计, 概览)
    # =========================================================================
    def json_to_excel(self, json_path):
        """
        读取算法返回的 JSON，生成包含完整统计信息的 Excel。
        """
        print(f"[-] 正在处理文件: {json_path}")
        if not os.path.exists(json_path):
            print(f"Error: 文件 {json_path} 不存在")
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error: 读取 JSON 失败 - {e}")
            return None

        processed_rows = []
        for entry in data:
            image_path = entry.get('image_path', '')
            image_url = entry.get('image_url', '')

            # 解析 Level 3 标签
            except_tags = self._clean_tags(entry.get('except_tags', []))
            target_tag = except_tags[-1] if except_tags else ''

            # 解析预测结果
            process_result = entry.get('process_result') or {}
            final_labels = process_result.get('final_labels') or process_result.get("res", {}).get("final_labels", [])
            # 判断是否命中
            is_included = '否'
            if target_tag:
                for label in final_labels:
                    if target_tag in label:
                        is_included = '是'
                        break

            total_labels_count = process_result.get('total_labels_count', "") or  process_result.get("res", {}).get("total_labels_count", "")
            elapsed_time = process_result.get('elapsed_time', "") or process_result.get("res", {}).get("elapsed_time", "")
            token_cost = process_result.get('token_cost', "") or process_result.get("res", {}).get("token_cost", "")
            row = {
                '路径名': image_path,
                '路径URL': image_url,
                '预测标签': '|'.join(final_labels),
                '路径标签': target_tag,
                '是否包含': is_included,
                '标签数量': total_labels_count,
                '耗时': elapsed_time,
                'Token消耗': token_cost
            }
            processed_rows.append(row)

        df_original = pd.DataFrame(processed_rows)

        if df_original.empty:
            print("Warning: 数据为空")
            return None

        # --- 生成统计 Sheet ---
        stats = df_original.groupby('路径标签').agg(
            总数=('是否包含', 'count'),
            匹配数=('是否包含', lambda x: (x == '是').sum())
        ).reset_index()
        stats['识别率'] = stats['匹配数'] / stats['总数']
        stats['识别率(%)'] = (stats['识别率'] * 100).round(2).astype(str) + '%'

        # --- 生成概览 Sheet ---
        total_tags = len(stats)
        rates = stats['识别率']
        avg_acc = rates.mean() if total_tags > 0 else 0

        def safe_rate(cond):
            return (rates[cond].count() / total_tags) if total_tags > 0 else 0

        summary_data = {
            '统计项': ['平均准确度', '100%命中率占比', '70%-100%命中率占比', '50%-70%命中率占比', '0-50%命中率占比', '0命中率占比'],
            '数值': [
                avg_acc,
                safe_rate(rates == 1.0),
                safe_rate((rates >= 0.7) & (rates < 1.0)),
                safe_rate((rates >= 0.5) & (rates < 0.7)),
                safe_rate((rates > 0) & (rates < 0.5)),
                safe_rate(rates == 0)
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary['数值(格式化)'] = df_summary['数值'].apply(lambda x: f"{x:.2%}")

        # --- 保存 ---
        output_excel = os.path.splitext(json_path)[0] + '.xlsx'
        try:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                df_original.to_excel(writer, sheet_name='原始数据', index=False)
                stats.to_excel(writer, sheet_name='识别率统计', index=False)
                df_summary.to_excel(writer, sheet_name='整体概览', index=False)
            print(f"[√] Excel 已生成: {output_excel}")
            return output_excel
        except Exception as e:
            print(f"Error: 保存 Excel 失败 - {e}")
            return None

    # =========================================================================
    # 核心功能 2: 下载图片
    # =========================================================================
    def download_images(self, excel_path, save_root_dir='downloaded_images'):
        """
        从 Excel 下载图片到本地。
        """
        print(f"[-] 开始下载图片: {excel_path}")
        if not os.path.exists(excel_path):
            return

        try:
            # 关键：使用 engine='openpyxl'
            df = pd.read_excel(excel_path, sheet_name='原始数据', engine='openpyxl')
        except Exception as e:
            # 兼容如果只有一个 sheet 的情况
            df = pd.read_excel(excel_path, index_col=None, engine='openpyxl')

        if not os.path.exists(save_root_dir):
            os.makedirs(save_root_dir)

        count, fail = 0, 0
        for _, row in df.iterrows():
            url = row.get('路径URL')
            rel_path = row.get('路径名')

            if pd.isna(url) or pd.isna(rel_path): continue

            # 构建本地路径 (去掉开头的 / 防止变为根目录)
            local_path = os.path.join(save_root_dir, str(rel_path).lstrip('/'))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if os.path.exists(local_path): continue  # 跳过已下载

            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    with open(local_path, 'wb') as f:
                        f.write(resp.content)
                    count += 1
                    if count % 10 == 0: print(f"    已下载 {count} 张...")
                else:
                    fail += 1
            except:
                fail += 1

        print(f"[√] 下载完成: 成功 {count}, 失败 {fail}")

    # =========================================================================
    # 核心功能 3: 筛选低分标签并重测 (生成新 JSON)
    # =========================================================================
    def retest_low_accuracy(self, excel_path = "images_result_with_labels_20260129_match_result.xlsx", threshold=0.6):
        """
        筛选识别率 < threshold 的标签，调用接口重测，保存为新 JSON。
        """
        print(f"[-] 开始重测流程 (阈值 < {threshold:.0%})")
        if not os.path.exists(excel_path): return None

        try:
            # 读取统计表和原始数据，关键：engine='openpyxl'
            df_stats = pd.read_excel(excel_path, sheet_name='识别率统计', engine='openpyxl')

            # 尝试读取原始数据
            xls = pd.ExcelFile(excel_path, engine='openpyxl')
            sheet_name = '原始数据' if '原始数据' in xls.sheet_names else xls.sheet_names[0]
            df_data = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')

        except Exception as e:
            print(f"Error: 读取 Excel 失败 - {e}")
            return None

        # 筛选标签
        target_tags = df_stats[df_stats['识别率'] <= threshold]['路径标签'].tolist()
        if not target_tags:
            print("    没有低于该阈值的标签。")
            return None

        print(f"    目标标签 ({len(target_tags)}个): {target_tags[:5]}...")
        target_rows = df_data[df_data['路径标签'].isin(target_tags)]
        print(f"    共需重测 {len(target_rows)} 张图片。")

        results_list = []
        headers = {"Content-Type": "application/json"}

        for idx, row in target_rows.iterrows():
            image_url = row.get('路径URL')
            image_path = row.get('路径名')

            # 接口请求参数构造
            image_info = image_url if pd.notna(image_url) and str(image_url).strip() else image_path

            # 还原 except_tags 用于保持 JSON 结构
            except_tags = str(image_path).split('/')[:-1] if isinstance(image_path, str) else []

            payload = {
                "image_info": image_info,
                "task_id": str(uuid.uuid4())
            }

            entry = {
                "image_name": os.path.basename(str(image_path)),
                "image_path": image_path,
                "image_url": image_url,
                "except_tags": except_tags,
                "process_result": {}
            }

            try:
                # 调用接口
                # print(f"    POST {self.api_url}...") 
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    api_res = resp.json()
                    entry['process_result'] = api_res.get('result', api_res)
                else:
                    entry['error'] = f"HTTP {resp.status_code}"
            except Exception as e:
                entry['error'] = str(e)

            results_list.append(entry)
            time.sleep(0.05)  # 限流

            if len(results_list) % 10 == 0:
                print(f"    进度: {len(results_list)}/{len(target_rows)}")

        # 保存结果
        today = datetime.datetime.now().strftime('%Y%m%d')
        output_json = f"images_result_with_labels_{today}_retest.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=4)

        print(f"[√] 重测完成，结果已保存: {output_json}")
        return output_json

    # =========================================================================
    # 核心功能 4: 双 Excel 对比可视化
    # =========================================================================
    def compare_two_excels(self, old_excel_path, new_excel_path, top_n=30):
        print(f"[-] 开始对比分析: {old_excel_path} vs {new_excel_path}")

        if not (os.path.exists(old_excel_path) and os.path.exists(new_excel_path)):
            print("Error: 输入文件不存在")
            return

        try:
            df_old_stats = pd.read_excel(old_excel_path, sheet_name='识别率统计', engine='openpyxl')
            df_new_stats = pd.read_excel(new_excel_path, sheet_name='识别率统计', engine='openpyxl')

            try:
                df_old_summary = pd.read_excel(old_excel_path, sheet_name='整体概览', engine='openpyxl')
                df_new_summary = pd.read_excel(new_excel_path, sheet_name='整体概览', engine='openpyxl')
                has_summary = True
            except:
                has_summary = False
        except Exception as e:
            print(f"Error: 读取 Excel 失败 - {e}")
            return

        df_old_stats = df_old_stats[['路径标签', '识别率']].rename(columns={'识别率': 'acc_old'})
        df_new_stats = df_new_stats[['路径标签', '识别率']].rename(columns={'识别率': 'acc_new'})

        merged = pd.merge(df_old_stats, df_new_stats, on='路径标签', how='inner')
        merged['improvement'] = merged['acc_new'] - merged['acc_old']

        detail_file = "comparison_full_detail.xlsx"
        merged.to_excel(detail_file, index=False, engine='openpyxl')
        print(f"[√] 全量对比明细已保存: {detail_file}")

        # 图表 1: 红榜
        df_improve = merged.sort_values('improvement', ascending=False).head(top_n)
        self._plot_chart(
            data=df_improve,
            title=f'迭代效果最好的 Top {len(df_improve)} 标签',
            output_filename='comparison_improvement.png',
            summary_data=(df_old_summary, df_new_summary) if has_summary else None
        )

        # 图表 2: 黑榜
        df_stagnant = merged[
            (merged['acc_new'] < 0.5) &
            (merged['improvement'].abs() < 0.05)
            ].sort_values('acc_new', ascending=True).head(top_n)

        if not df_stagnant.empty:
            self._plot_chart(
                data=df_stagnant,
                title=f'准确率低(<50%)且无改善的 Top {len(df_stagnant)} 标签',
                output_filename='comparison_stagnant.png',
                summary_data=None,
                bar_colors=('#A9A9A9', '#696969')
            )
        else:
            print("[-] 太棒了！没有发现“准确率低且无变化”的标签。")

    def _plot_chart(self, data, title, output_filename, summary_data=None, bar_colors=('#87CEFA', '#FF7F50')):
        if data.empty: return

        fig_height = 8 + (len(data) // 10)
        height_ratios = [4, 1] if summary_data else [1]

        fig = plt.figure(figsize=(14, fig_height))
        gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.5)

        ax1 = fig.add_subplot(gs[0])
        labels = data['路径标签']
        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax1.bar(x - width / 2, data['acc_old'], width, label='旧版本', color=bar_colors[0])
        rects2 = ax1.bar(x + width / 2, data['acc_new'], width, label='新版本', color=bar_colors[1])

        ax1.set_ylabel('识别率 (Accuracy)')
        ax1.set_title(title)
        ax1.set_xticks(x)
        rotation = 45 if len(data) < 30 else 90
        ax1.set_xticklabels(labels, rotation=rotation, ha='right' if rotation == 45 else 'center')
        ax1.legend()

        # --- [新增] 显示百分比标签 ---
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                # 格式化为两位小数，例如 0.85
                # 如果数值为0，也可以选择不显示以保持整洁： if height > 0: ...
                ax1.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                         f'{height:.2f}',
                         ha='center', va='bottom', fontsize=8,
                         rotation=90 if len(data) > 20 else 0)  # 数量多时数值也竖着写
        autolabel(rects1)
        autolabel(rects2)
        # ---------------------------

        if summary_data:
            df_old_sum, df_new_sum = summary_data
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')

            summary_merge = pd.merge(
                df_old_sum[['统计项', '数值(格式化)']],
                df_new_sum[['统计项', '数值(格式化)']],
                on='统计项', suffixes=('_旧', '_新')
            )

            table_data = []
            columns = ['统计指标', '旧版本数据', '新版本数据', '变化趋势']
            for _, row in summary_merge.iterrows():
                val_old = row['数值(格式化)_旧']
                val_new = row['数值(格式化)_新']
                try:
                    diff = float(val_new.strip('%')) - float(val_old.strip('%'))
                    trend = f"{'↑' if diff > 0 else '↓' if diff < 0 else ''} {diff:+.2f}%"
                except:
                    trend = "-"
                table_data.append([row['统计项'], val_old, val_new, trend])

            table = ax2.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            for (r, c), cell in table.get_celld().items():
                if r == 0: cell.set_facecolor('#f0f0f0')

        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        print(f"[√] 图表已保存: {output_filename}")
        plt.close()

# =========================================================================
# 使用示例 (Main 流程)
# =========================================================================
if __name__ == "__main__":
    # 1. 实例化 Pipeline
    api_url = "http://10.136.234.255:8081/process_image"
    pipeline = ImageTagPipeline()

    # 2. 原始文件处理 (第一轮)
    first_round_json = "images_result_with_labels_20260129_match_result.json"


    # 2.1 JSON -> Excel
    old_excel = pipeline.json_to_excel(first_round_json)

    # 3. 开始重测流程;这个保存文件名字为retest不一致，用另一个py badcase_improve来测
    # 筛选识别率 < 60% 的标签进行重测

    retest_json = pipeline.retest_low_accuracy(threshold=1)
    # retest_json = "images_result_with_labels_20260211_retest.json"
    new_excel = pipeline.json_to_excel(retest_json)
    #
    # # 4. 将重测结果转为新 Excel
    # retest_json = "images_result_with_labels_20260211_match_result.json"
    new_excel = pipeline.json_to_excel(retest_json)

    # 5. 对比两个 Excel
    pipeline.compare_two_excels(old_excel, new_excel)