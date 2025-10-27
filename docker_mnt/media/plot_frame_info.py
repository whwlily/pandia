import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # 正确导入 Line2D

# 定义不同帧类型的标记和颜色
frame_markers = {'I': 'o', 'P': 's', 'B': 'x'}
frame_colors = {'I': 'blue', 'P': 'green', 'B': 'red'}

def read_log_file(file_path):
    frame_sizes = []
    frame_types = []
    
    # 读取文件并解析帧大小和类型
    with open(file_path, 'r') as f:
        for line in f:
            size, frame_type = line.strip().split(',')
            frame_sizes.append(int(size))
            frame_types.append(frame_type)
    
    return frame_sizes, frame_types

def plot_frame_sizes(frame_sizes, frame_types):
    plt.figure(figsize=(10, 6))

    # 遍历帧数据，根据帧类型分别绘制
    for i, (size, frame_type) in enumerate(zip(frame_sizes, frame_types)):
        if frame_type != "I":
            plt.scatter(i, size, color=frame_colors[frame_type])
        else:
            plt.scatter(i, size, color=frame_colors[frame_type])
        
    # 设置图表的标题和标签
    plt.title('Frame Sizes by Type')
    plt.xlabel('Frame Number')
    plt.ylabel('Frame Size (Bytes)')
    
    # 创建图例
    # handles = [Line2D([0], [0], marker=frame_markers[t], color='w', label=t,
    #                   markerfacecolor=frame_colors[t], markersize=8) for t in frame_markers]
    # plt.legend(handles=handles, title="Frame Type")

    plt.grid(True)
    plt.savefig("frame_info.png")

# 主函数调用
file_path = '/data2/wuhw/Workspace/Pandia/media/info.log'  # 将此路径替换为你的日志文件路径
frame_sizes, frame_types = read_log_file(file_path)
plot_frame_sizes(frame_sizes, frame_types)