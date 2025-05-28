#!/bin/bash

# 定义源视频和接收视频目录
SOURCE_DIR="/data2/kj/Workspace/Pandia/docker_mnt/media/source"
RES_VIDEO_DIR="/data2/kj/Workspace/Pandia/docker_mnt/media/res_video"

# 定义视频格式
width=1280
height=720
pix_fmt="yuv420p"

# 输出 VMAF 值的文件
output_file="/data2/kj/Workspace/Pandia/docker_mnt/media/vmaf_scores.txt"

# 清空输出文件
echo "" > "$output_file"

# 遍历 res_video 目录下的 YUV 文件
for received_file in "$RES_VIDEO_DIR"/received_*.yuv; do
    # 获取文件名中的帧号 (假设文件名格式为 received_X.yuv)
    frame_number=$(basename "$received_file" .yuv | sed 's/received_//')

    # 获取对应的源文件
    source_file="$SOURCE_DIR/frame_$(printf "%04d" "$frame_number")".yuv

    # 检查源文件是否存在
    if [[ ! -f "$source_file" ]]; then
        echo "⚠️ Source frame $source_file does not exist, skipping..."
        continue
    fi

    # 使用 ffmpeg 计算 VMAF
    echo "Calculating VMAF for frame $frame_number..."

    # ffmpeg -f rawvideo -pix_fmt $pix_fmt -s ${width}x${height} -i "$received_file" \
    #     -f rawvideo -pix_fmt $pix_fmt -s ${width}x${height} -i "$source_file" \
    #     # -lavfi "libvmaf" -f null - 2>&1 | grep "VMAF score" | awk '{print "Frame " $6 " VMAF: " $8}' >> "$output_file"
    #     -lavfi "libvmaf" -f null - 2>&1 | \
    #     grep "VMAF score" | \
    #     awk -v frame="$frame_number" '{print "Frame " $frame " VMAF: " $6}' >> "$output_file"

    vmaf_score=$(ffmpeg -f rawvideo -pix_fmt $pix_fmt -s ${width}x${height} -i "$received_file" \
        -f rawvideo -pix_fmt $pix_fmt -s ${width}x${height} -i "$source_file" \
        -lavfi "libvmaf" -f null - 2>&1 | \
        grep "VMAF score" | \
        awk '{print $6}')

    # Write frame number and VMAF score to the output file
    echo "Frame $frame_number VMAF: $vmaf_score" >> "$output_file"

    echo "VMAF for frame $frame_number calculated and saved."
done

echo "VMAF calculation complete. Results saved to $output_file."
