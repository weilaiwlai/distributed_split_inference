import json
import os

def process_json_data(file_path):
    """
    处理指定JSON文件，计算除第一条数据外所有数值字段的平均值
    
    Args:
        file_path (str): JSON数据文件的路径
    
    Returns:
        dict: 各数值字段的平均值结果
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：找不到文件 {file_path}")
    
    # 2. 读取并解析JSON文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"错误：{file_path} 不是有效的JSON文件")
    except Exception as e:
        raise Exception(f"读取文件时发生错误：{str(e)}")
    
    # 3. 验证数据格式（必须是列表）
    if not isinstance(data, list):
        raise TypeError("错误：JSON文件中的数据必须是列表格式")
    
    # 4. 检查数据量是否足够
    if len(data) <= 1:
        print("警告：数据条目数量小于等于1，无法计算除第一个外的平均值")
        return {}
    
    # 5. 提取除第一个外的所有数据并计算平均值
    target_data = data[1:]
    sums = {}
    count = len(target_data)
    
    for record in target_data:
        if not isinstance(record, dict):
            print(f"警告：跳过非字典格式的数据条目：{record}")
            continue
        
        for key, value in record.items():
            # 只处理数值类型
            if isinstance(value, (int, float)):
                if key not in sums:
                    sums[key] = 0.0
                sums[key] += value
    
    # 6. 计算平均值
    averages = {key: total / count for key, total in sums.items()}
    
    return averages

def save_result(result, output_path="average_result.json"):
    """
    将计算结果保存为JSON文件
    
    Args:
        result (dict): 计算得到的平均值结果
        output_path (str): 结果保存路径，默认当前目录下的average_result.json
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 统计结果已保存到：{os.path.abspath(output_path)}")
    except Exception as e:
        raise Exception(f"保存结果时出错：{str(e)}")

# ====================== 请修改这里的配置 ======================
# 替换为你的JSON数据文件的实际路径（绝对路径/相对路径都可以）
INPUT_JSON_PATH = "gsm8k_100mbit_client_layers_2.json"  # 例如："./metrics_data.json" 或 "C:/data/results.json"
# 结果保存路径（可选修改）
OUTPUT_JSON_PATH = "average_result.json"
# =============================================================

if __name__ == "__main__":
    try:
        # 处理数据并计算平均值
        print(f"正在处理文件：{os.path.abspath(INPUT_JSON_PATH)}")
        average_result = process_json_data(INPUT_JSON_PATH)
        
        if average_result:
            # 打印结果到控制台
            print("\n📊 计算结果（除第一条数据外的平均值）：")
            for key, value in average_result.items():
                print(f"  {key}: {value:.6f}")
            
            # 保存结果到文件
            save_result(average_result, OUTPUT_JSON_PATH)
        else:
            print("\n❌ 未计算出任何平均值（数据量不足）")
    
    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")