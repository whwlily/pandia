import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import os
import numpy as np
# 定义12个要提取的属性
fields = [
    "receiving_rate", "received_packets_num", "received_bytes", "queuing_delay", "delay",
    "mini_seen_delay", "delay_ratio", "delay_avg_mini_diff", "pkt_received_interval",
    "pkt_received_jitter", "pkt_loss_ratio", "pkt_loss_num"
]

with open('monitor_data_new.json', 'r') as f:
    json_data = json.load(f)

# 创建两个列表来保存分组数据
group1 = []  # duration 小于 1 的组
group2 = []  # duration 大于等于 1 的组

# 根据 duration 分组
for entry in json_data:
    if entry.get('duration', 0) == 0.06:
        group1.append({key: entry.get(key, None) for key in fields})
    else:
        group2.append({key: entry.get(key, None) for key in fields})

# 将数据写入 CSV 文件
def write_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

# 写入两个CSV文件
write_to_csv('group_short.csv', group1)  # duration 小于1的组
write_to_csv('group_long.csv', group2)  # duration 大于等于1的组

print("数据已成功保存到 group_short.csv 和 group_long.csv 文件中。")

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result_bwe_data')


# 创建单位类型映射
unit_categories = {
    'rate': ['receiving_rate'],
    'received_bytes': ['receiving_rate'],
    'received_packets_num': ['received_packets_num'],
    'delay': ['queuing_delay', 'delay', 'mini_seen_delay', 'delay_avg_mini_diff'],
    'delay_ratio': ['delay_ratio'],
    'jitter': ['pkt_received_interval', 'pkt_received_jitter'],
    'pkt_loss_num': ['pkt_loss_num'],
    'pkt_loss_ratio': ['pkt_loss_ratio']
}

df = []
time_interval = .06  # s
# 绘制同一单位类型的数据
def plot_same_unit(dataframe, columns, title, ylabel, save_name):
    for group_file in ['group_short.csv', 'group_long.csv']:
        rank = 0
        df = pd.read_csv(group_file)
        for delay_i in df["delay"]:
            if delay_i == 3000:
                rank += 1
        dataframe = df[rank:]
        time = np.arange(0, len(dataframe) * time_interval, time_interval)
        plt.figure(figsize=(10, 6))
        
        for col in columns:
            plt.plot(time, dataframe[col], label=col)

        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        # plt.xticks(np.arange(0, len(df) * time_interval, step=time_interval))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, group_file.split('.csv')[0], save_name))
        plt.close()

# 绘制各类图表
plot_same_unit(df, unit_categories['rate'], 'Rate Data', 'Rate', 'rate_plot.png')
plot_same_unit(df, unit_categories['received_bytes'], 'Received Bytes Data', 'Bytes', 'received_bytes_plot.png')
plot_same_unit(df, unit_categories['received_packets_num'], 'Received Packets Data', 'Packets', 'received_packets_plot.png')
plot_same_unit(df, unit_categories['delay'], 'Delay Data', 'Delay (ms)', 'delay_plot.png')
plot_same_unit(df, unit_categories['delay_ratio'], 'Delay Ratio Data', 'Delay_Ratio', 'Delay_Ratio.png')
plot_same_unit(df, unit_categories['jitter'], 'Jitter Data', 'Jitter (ms)', 'jitter_plot.png')
plot_same_unit(df, unit_categories['pkt_loss_num'], 'Packet Loss Number Data', 'Loss_Num', 'Loss_Num.png')
plot_same_unit(df, unit_categories['pkt_loss_ratio'], 'Packet Loss Ratio Data', 'Loss_Ratio', 'Loss_Ratio.png')



