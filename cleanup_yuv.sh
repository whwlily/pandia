#!/bin/bash
# 清理env_emulator_offline产生的YUV文件

echo "正在停止相关的Docker容器..."
docker stop $(docker ps -q --filter "name=sb3_emulator") 2>/dev/null || true

echo "正在清理YUV文件..."
sudo rm -rf /data2/wuhw/Workspace/Pandia/docker_mnt/media/res_video/*

echo "清理完成！"
ls -la /data2/wuhw/Workspace/Pandia/docker_mnt/media/res_video/
