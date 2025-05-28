import os
import cv2
import numpy as np

# 目标分辨率 (720p)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 添加 180p 和 270p 支持的 YUV420p 分辨率列表
COMMON_RESOLUTIONS = [
    (320, 180),  # 180p
    (480, 270),  # 270p
    (640, 360),  # 360p
    (960, 540),  # 540p
    (1280, 720), # 720p (无需缩放)
]

# 计算 YUV420p 单帧大小
def calculate_yuv420_size(width, height):
    return int(width * height * 1.5)

# 根据文件大小推测分辨率
def infer_resolution(filepath):
    file_size = os.path.getsize(filepath)  # 获取文件大小 (字节)
    
    for width, height in COMMON_RESOLUTIONS:
        frame_size = calculate_yuv420_size(width, height)
        if file_size == frame_size:
            print(f"文件大小：{file_size}，推测 {filepath} 的分辨率为 {width}x{height}")
            return width, height

    print(f"无法推测 {filepath} 的分辨率，跳过。")
    return None, None

# # 读取 YUV 文件
# def read_yuv420(filename, width, height):
#     frame_size = calculate_yuv420_size(width, height)
#     with open(filename, "rb") as f:
#         raw = f.read(frame_size)
#         if len(raw) < frame_size:
#             print(f"Warning: {filename} 文件大小不匹配，可能损坏！")
#             return None, None, None

#     y_plane = np.frombuffer(raw[:width * height], dtype=np.uint8).reshape((height, width))
#     u_plane = np.frombuffer(raw[width * height:width * height + (width//2) * (height//2)], dtype=np.uint8).reshape((height//2, width//2))
#     v_plane = np.frombuffer(raw[width * height + (width//2) * (height//2):], dtype=np.uint8).reshape((height//2, width//2))

#     return y_plane, u_plane, v_plane

def read_yuv420(filename, width, height):
    frame_size = calculate_yuv420_size(width, height)
    with open(filename, "rb") as f:
        raw = f.read(frame_size)
        if len(raw) < frame_size:
            print(f"Warning: {filename} 文件大小不匹配，可能损坏！")
            return None, None, None

    y_plane = np.frombuffer(raw[:width * height], dtype=np.uint8).reshape((height, width))
    u_plane = np.frombuffer(raw[width * height:width * height + (width//2) * (height//2)], dtype=np.uint8).reshape((height//2, width//2))
    v_plane = np.frombuffer(raw[width * height + (width//2) * (height//2):], dtype=np.uint8).reshape((height//2, width//2))

    return y_plane, u_plane, v_plane

# # 重新调整 YUV420p 分辨率到 720p
# def resize_yuv420(y, u, v, src_width, src_height, target_width, target_height):
#     y_resized = cv2.resize(y, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
#     u_resized = cv2.resize(u, (target_width // 2, target_height // 2), interpolation=cv2.INTER_LINEAR)
#     v_resized = cv2.resize(v, (target_width // 2, target_height // 2), interpolation=cv2.INTER_LINEAR)
#     return y_resized, u_resized, v_resized
def resize_yuv420(y, u, v, src_width, src_height, target_width, target_height):
    y_resized = cv2.resize(y, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # UV 平面缩放时使用 INTER_NEAREST 避免色彩错误
    u_resized = cv2.resize(u, (target_width // 2, target_height // 2), interpolation=cv2.INTER_NEAREST)
    v_resized = cv2.resize(v, (target_width // 2, target_height // 2), interpolation=cv2.INTER_NEAREST)
    
    return y_resized, u_resized, v_resized

# 保存 YUV420p 数据
def save_yuv420(filename, y, u, v):
    with open(filename, "wb") as f:
        f.write(y.tobytes())
        f.write(u.tobytes())
        f.write(v.tobytes())

# 处理目录下的所有 YUV 文件
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".yuv"):
            filepath = os.path.join(input_dir, file)
            # 推测分辨率
            width, height = infer_resolution(filepath)
            if width is None or height is None:
                continue  # 无法推测，跳过
            
            # 读取 YUV 数据
            y, u, v = read_yuv420(filepath, width, height)
            if y is None:
                continue  # 读取失败，跳过

            # 如果小于 720p，进行插值
            if width < TARGET_WIDTH or height < TARGET_HEIGHT:
                y, u, v = resize_yuv420(y, u, v, width, height, TARGET_WIDTH, TARGET_HEIGHT)

            # 保存到新文件
            output_file = os.path.join(output_dir, f"resized_{file}")
            save_yuv420(output_file, y, u, v)
            print(f"已处理: {file} -> {output_file}")


# 示例：处理当前目录下的 yuv 文件，存到 output 目录
process_directory("/data2/kj/Workspace/Pandia/docker_mnt/media/res_video", "/data2/kj/Workspace/Pandia/docker_mnt/media/resized")

# import os
# import cv2
# import numpy as np

# # 目标分辨率
# TARGET_WIDTH = 1280
# TARGET_HEIGHT = 720

# def calculate_yuv420_size(width, height):
#     return int(width * height * 1.5)

# def read_yuv420(filename, width, height):
#     frame_size = calculate_yuv420_size(width, height)
#     with open(filename, "rb") as f:
#         raw = f.read(frame_size)
#         if len(raw) < frame_size:
#             print(f"Warning: {filename} 文件大小不匹配，可能损坏！")
#             return None, None, None

#     y_plane = np.frombuffer(raw[:width * height], dtype=np.uint8).reshape((height, width))
#     u_plane = np.frombuffer(raw[width * height:width * height + (width//2) * (height//2)], dtype=np.uint8).reshape((height//2, width//2))
#     v_plane = np.frombuffer(raw[width * height + (width//2) * (height//2):], dtype=np.uint8).reshape((height//2, width//2))

#     return y_plane, u_plane, v_plane

# def resize_yuv420(y, u, v, src_width, src_height, target_width, target_height):
#     y_resized = cv2.resize(y, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
#     # UV 平面缩放时使用 INTER_NEAREST 避免色彩错误
#     u_resized = cv2.resize(u, (target_width // 2, target_height // 2), interpolation=cv2.INTER_NEAREST)
#     v_resized = cv2.resize(v, (target_width // 2, target_height // 2), interpolation=cv2.INTER_NEAREST)
    
#     return y_resized, u_resized, v_resized

# def save_yuv420(filename, y, u, v):
#     with open(filename, "wb") as f:
#         f.write(y.tobytes())
#         f.write(u.tobytes())
#         f.write(v.tobytes())

# def yuv_to_rgb(y, u, v, width, height):
#     # UV 平面需要放大回 Y 分辨率大小
#     u = cv2.resize(u, (width, height), interpolation=cv2.INTER_NEAREST)
#     v = cv2.resize(v, (width, height), interpolation=cv2.INTER_NEAREST)
    
#     # 转换为 BGR 格式
#     yuv = cv2.merge([y, u, v])
#     bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
#     return bgr

# # 处理单个 YUV 文件
# def process_yuv_file(input_file, output_png):
#     width, height = 640, 360  # 这里手动指定原始分辨率
#     y, u, v = read_yuv420(input_file, width, height)
#     if y is None:
#         return

#     y, u, v = resize_yuv420(y, u, v, width, height, TARGET_WIDTH, TARGET_HEIGHT)
    
#     # 转换成 RGB 并保存
#     rgb_img = yuv_to_rgb(y, u, v, TARGET_WIDTH, TARGET_HEIGHT)
#     cv2.imwrite(output_png, rgb_img)
#     print(f"已保存: {output_png}")

# 测试
# process_yuv_file("/data2/kj/Workspace/Pandia/docker_mnt/media/res_video/received_719.yuv", "output.png")


