#!/bin/bash

# 设置网络接口
INTERFACE="lo"

# 设置延迟时间（可以根据需求固定或者动态变化）
DELAY=50  # 延迟时间，单位：ms

# 带宽数组（单位：kbit/s）
BANDWIDTHS=(1000 2000 500 1500 3000 1000)  # 每秒钟更新的带宽值

# 清理现有的流量控制规则
tc qdisc del dev $INTERFACE root 2> /dev/null

# 添加初始的 netem 规则，用于设置延迟
tc qdisc add dev $INTERFACE root handle 1: netem delay ${DELAY}ms

# 开始动态调整带宽
for BW in "${BANDWIDTHS[@]}"; do
  echo "设置带宽为 ${BW} kbit/s"

  # 删除之前的 tbf 限制（如果存在）
  tc qdisc del dev $INTERFACE parent 1: handle 2: 2> /dev/null

  # 添加 tbf 规则，动态设置带宽
  tc qdisc add dev $INTERFACE parent 1: handle 2: tbf rate ${BW}kbit burst 1500 latency 100ms

  # 等待1秒
  sleep 1
done

# 清理流量控制规则
tc qdisc del dev $INTERFACE root
echo "流量控制规则已清理"