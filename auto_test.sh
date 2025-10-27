#!/bin/bash

# 自动化离线测试脚本
# 自动输入sudo密码，无需人工干预
# 结果将保存到 /data2/wuhw/Workspace/Pandia/results/

echo "开始自动化离线测试..."
echo "密码已配置，无需手动输入"
echo "结果将保存到: /data2/wuhw/Workspace/Pandia/results/"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pandia

# 设置密码环境变量（用于sudo -S）
export SUDO_PASSWORD="wuhw123456"

# 确保结果目录存在
mkdir -p /data2/wuhw/Workspace/Pandia/results

# 运行离线测试
echo "启动离线测试程序..."
python -m pandia.agent.env_emulator_offline

echo "离线测试完成！"
echo "结果保存在: /data2/wuhw/Workspace/Pandia/results/"
