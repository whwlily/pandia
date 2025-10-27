#!/bin/bash

# 设置正确的环境变量和路径
export PYTHONPATH="/data2/wuhw/Workspace/Pandia:$PYTHONPATH"
export SUDO_PASSWORD="wuhw123456"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pandia

# 切换到项目目录
cd /data2/wuhw/Workspace/Pandia

# 运行离线测试
echo "开始运行离线测试..."
echo "PYTHONPATH: $PYTHONPATH"
echo "当前目录: $(pwd)"

python -m pandia.agent.env_emulator_offline
