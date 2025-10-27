#!/bin/bash

# 配置sudo免密码脚本
# 只对特定的rm命令免密码，更安全

echo "配置sudo免密码..."

# 创建sudoers规则文件
sudo tee /etc/sudoers.d/pandia_cleanup > /dev/null << 'EOF'
# Pandia项目清理权限
wuhw ALL=(ALL) NOPASSWD: /bin/rm -rf /data2/wuhw/Workspace/Pandia/docker_mnt/media/res_video/*
wuhw ALL=(ALL) NOPASSWD: /usr/bin/docker stop *
EOF

echo "sudo免密码配置完成！"
echo "现在可以运行: ./auto_test.sh"
