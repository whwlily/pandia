# frame_size=$((1280 * 720 * 3 / 2))
# mkdir frames
# split -b $frame_size drive_720p.yuv frames/temp_
# cd frames
# ls | awk '{printf("mv %s frame_%04d.yuv\n", $0, NR)}' | bash
# cd ..

for file in res_video/*.yuv; do
    size=$(stat -c%s "$file")  # 获取文件大小
    case $size in
        1382400) resolution="1280x720" ;;
        # 777600)  resolution="960x540"  ;;
        # 345600)  resolution="640x360"  ;;
        # 194400)  resolution="480x270"  ;;
        # 86400)   resolution="320x180"  ;;
        *) echo "⚠️ 无法确定 $file 的分辨率，跳过！"; continue ;;
    esac
    mkdir -p res_png
    ffmpeg -video_size $resolution -pixel_format yuv420p -i "$file" -vsync 0 "res_png/$(basename "$file" .yuv)_%04d.png"
done

# mkdir -p resized_png

# for file in resized/*.yuv; do
#   # 提取文件名
#   filename=$(basename "$file" .yuv)
  
#   # 使用 ffmpeg 转换 YUV 到 PNG，假设分辨率为 1280x720 和 yuv420p 格式
#   ffmpeg -f rawvideo -pix_fmt yuv420p -s 1280x720 -i "$file" "resized_png/${filename}.png"
# done