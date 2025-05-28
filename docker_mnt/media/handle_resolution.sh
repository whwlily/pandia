#!/bin/bash

# 定义目录
SOURCE_DIR="/data2/kj/Workspace/Pandia/docker_mnt/media/source"
RES_VIDEO_DIR="/data2/kj/Workspace/Pandia/docker_mnt/media/res_video"

json_data="$1"

# 使用 jq 解析 JSON 数据
declare -A qp_map
while IFS="=" read -r key value; do
    qp_map["$key"]="$value"
done < <(echo "$json_data" | jq -r 'to_entries|map("\(.key)=\(.value|tostring)")|.[]')

# 定义目标分辨率
declare -A resolutions
resolutions["180p"]="320x180"
resolutions["270p"]="480x270"
# resolutions["360p"]="640x360"
resolutions["540p"]="960x540"
resolution_720p="1280x720"
# 选择插值算法
interpolation="bilinear"  # 你可以选择 "nearest", "bilinear", "bicubic", "lanczos"


# 遍历 res_video 目录下的文件
for received_file in "$RES_VIDEO_DIR"/received_*.yuv; do
    # 获取文件名中的帧号 (假设文件名格式为 received_X.yuv)
    frame_number=$(basename "$received_file" .yuv | sed 's/received_//')

    # 检查文件的分辨率 (通过文件名的帧号)
    size=$(stat -c %s "$received_file")
    
    if [ "$size" -eq 345600 ]; then
        tmp_file="temp_$(basename "$received_file")"
        ffmpeg -s 640x360 -pix_fmt yuv420p -i "$received_file" \
               -vf "scale=1280x720:sws_flags=$interpolation" \
               -pix_fmt yuv420p -y "$tmp_file" && mv -f "$tmp_file" "$received_file"
        rm -f "$tmp_file"
    fi
    
done

# 345600) # 480x270
#     # resolution="270p"
#     ffmpeg -s 640x360 -pix_fmt yuv420p -i "$received_file" \
#        -vf "scale=1280:720:sws_flags=$interpolation" \
#        -pix_fmt yuv420p -y "$received_file"
#     continue
#     ;;
# 遍历 res_video 目录下的文件
for received_file in "$RES_VIDEO_DIR"/received_*.yuv; do
    # 获取文件名中的帧号 (假设文件名格式为 received_X.yuv)
    frame_number=$(basename "$received_file" .yuv | sed 's/received_//')

    # 检查文件的分辨率 (通过文件名的帧号)
    size=$(stat -c %s "$received_file")

    case $size in
        86400)  # 320x180
            resolution="180p"
            ;;
        194400) # 480x270
            resolution="270p"
            ;;
        777600) # 960x540
            resolution="540p"
            ;;
        *)
            echo "⚠️ 360p 720p for $received_file, skipping..."
            continue
            ;;
    esac

    # 获取对应的源帧
    source_file="$SOURCE_DIR/frame_$(printf "%04d" "$frame_number")".yuv

    # 检查源文件是否存在
    if [[ ! -f "$source_file" ]]; then
        echo "⚠️ Source frame $source_file does not exist, skipping..."
        continue
    fi

    # 获取对应的分辨率
    target_resolution="${resolutions[$resolution]}"
    echo "Processing frame $frame_number with resolution $target_resolution"

     # 获取对应的 QP 值
    qp_value="${qp_map[$frame_number]}"
    if [[ -z "$qp_value" ]]; then
        echo "⚠️ QP value for frame $frame_number not found, skipping..."
        continue
    fi

    # 临时文件存储重新编码后的结果
    temp_file="$RES_VIDEO_DIR/temp_$(basename "$received_file")"
    temp_mp4="$RES_VIDEO_DIR/temp_mp4.mp4"

    # 使用 QP 值重新编码
    # ffmpeg -s 1280x720 -pix_fmt yuv420p -i "$source_file" \
    #    -vf "scale=$target_resolution:sws_flags=$interpolation" \
    #    -c:v libx264 -qp "$qp_value" -pix_fmt yuv420p -f rawvideo - | \
    # ffmpeg -f rawvideo -pix_fmt yuv420p -s $target_resolution -i - \
    #     -vf "scale=1280:720:sws_flags=$interpolation" \
    #     -pix_fmt yuv420p "$temp_file"

    ffmpeg -s 1280x720 -pix_fmt yuv420p -i "$source_file" \
       -vf "scale=$target_resolution:sws_flags=$interpolation" \
       -c:v libx264 -qp "$qp_value" -f mp4 "$temp_mp4"
    ffmpeg -i "$temp_mp4" -vf "scale=1280:720:sws_flags=$interpolation" -f rawvideo -pix_fmt yuv420p "$temp_file"
    rm -f "$temp_mp4"


    # ffmpeg -s 1280x720 -pix_fmt yuv420p -i ./source/frame_0101.yuv \
    #    -vf "scale=960x540:sws_flags=bilinear" \
    #    -c:v libx264 -qp 35 -f mp4 - | \
    # ffmpeg -i "$temp_mp4" -vf "scale=1280:720:sws_flags=bilinear" -f rawvideo -pix_fmt yuv420p test_101.yuv

    # ffmpeg -s 1280x720 -pix_fmt yuv420p -i "$source_file" \                                                          
    #    -vf "scale=$target_resolution:sws_flags=$interpolation" \
    #    -c:v libx264 -qp "$qp_value" test_output.mp4
    # ffmpeg -i pipe: -vf "scale=1280:720:sws_flags=bilinear" -f rawvideo -pix_fmt yuv420p "upsampled.yuv"

    # 替换原有的帧文件
    mv -f "$temp_file" "$received_file"
    echo "Updated $received_file with the interpolated 720p frame and QP value $qp_value."

    # ffmpeg -s 1280x720 -pix_fmt yuv420p -i "$source_file" \
    #    -vf "scale=$target_resolution:sws_flags=$interpolation,scale=1280:720:sws_flags=$interpolation" \
    #    -pix_fmt yuv420p "$temp_file"

    # # 替换原有的帧文件
    # mv -f "$temp_file" "$received_file"
    # echo "Updated $received_file with the interpolated 720p frame."
done

echo "Frame replacement complete."
