#!/bin/bash
# BANDWIDTH=$1  # 第一个参数为带宽
# DELAY=$2      # 第二个参数为延迟
# LOSS=$3       # 第三个参数为丢包率
# K=1000
# HIGH=$((4 * K))
# LOW=$((2 * K))

# # 带宽数组（单位：kbit/s）
# BANDWIDTHS=($HIGH $LOW)  # 每秒钟更新的带宽值

# # 清理现有的流量控制规则
# tc qdisc del dev lo root 2> /dev/null

# # 添加初始的 netem 规则，用于设置延迟
# # tc qdisc add dev lo root handle 1: netem delay ${DELAY}ms
# tc qdisc add dev lo root handle 1: netem delay 30ms

# # 添加初始的 tbf 规则
# tc qdisc add dev lo parent 1: handle 2: tbf rate ${BANDWIDTH}kbit burst 1500 latency 100ms

# while true; do
#     # 开始动态调整带宽
#     for BW in "${BANDWIDTHS[@]}"; do
#         echo "设置带宽为 ${BW} kbit/s"

#          # 修改现有的 tbf 规则，动态设置带宽
#         tc qdisc change dev lo parent 1: handle 2: tbf rate ${BW}kbit burst 1500 latency 100ms

#         # 等待1秒
#         sleep 5
#     done
# done

# # 清理流量控制规则
# tc qdisc del dev lo root
# echo "流量控制规则已清理"


# 检查是否提供了 JSON 文件路径
# $# 是一个特殊的 Shell 变量，表示脚本被调用时传递的命令行参数的个数。
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_json>"
    exit 1
fi

JSON_FILE=$1  # 第一个参数为带宽脚本路径
DELAY=$2      # 第二个参数为延迟
LOSS=$3       # 第三个参数为丢包率
JITTER=$4     # 第四个参数为抖动

# 使用 jq 从 JSON 文件中读取参数
BANDWIDTHS=($(jq -r '.true_capacity[] | . / 1024' "$JSON_FILE"))
DELAYS=($(jq -r '.observations[][35]' "$JSON_FILE"))
LOSS_trace=($(jq -r '.observations[][105]' "$JSON_FILE"))

# 打印读取的配置
echo "读取的配置："
# echo "带宽数组：${BANDWIDTHS[@]}"
# echo "延迟数组：${DELAYS[@]}"
echo "延迟：${DELAY}ms"
echo "丢包率：${LOSS}%"

# 清理现有的流量控制规则
tc qdisc del dev lo root 2> /dev/null

# 添加初始的 netem 规则，用于设置延迟
tc qdisc add dev lo root handle 1: netem delay ${DELAY}ms ${JITTER}ms loss ${LOSS}% 

# 添加初始的 tbf 规则
tc qdisc add dev lo parent 1: handle 2: tbf rate ${BANDWIDTHS[0]}kbit burst 1500 limit 75000

# # 添加 prio 规则管理优先级
# tc qdisc add dev lo parent 2: handle 3: prio

# # 小包（小于128字节）进入最高优先级队列 3:1
# tc filter add dev lo protocol ip parent 3: prio 1 u32 \
#     match u16 0 0 at 2 \
#     match u16 0x007f 0x007f at 2 \
#     flowid 3:1

# # 默认流量进入最低优先级队列 3:2
# tc filter add dev lo protocol ip parent 3: prio 2 u32 \
#     match ip protocol 0 0 \
#     flowid 3:2

# while true; do
#     # 开始动态调整带宽
#     for BW in "${BANDWIDTHS[@]}"; do
#         BW=$(printf "%.0f" "$BW")  # 将带宽值转换为整数
#         # NEW_BURST=$((BW * 50 / 8)) # 动态计算 burst (单位: bytes)
#         # echo "设置带宽为 ${BW} kbit/s, burst 为 ${NEW_BURST} bytes"
#         # 修改现有的 tbf 规则，动态设置带宽
#         tc qdisc change dev lo parent 1: handle 2: tbf rate ${BW}kbit burst 1500 limit 75000
#         # tc qdisc change dev lo parent 1: handle 2: tbf rate ${BW}kbit burst 1500 latency 200ms limit 75000
#         echo "${BW} kbit/s"
#         # tc -s qdisc show dev lo
#         # 等待1秒
#         sleep 0.06
#     done
# done

while true; do
    # 开始动态调整带宽和延迟
    for i in "${!BANDWIDTHS[@]}"; do

        # i 超过 250 就退出整个 while true 循环
        if (( i > 2000 )); then
            echo "i = $i，超过限制，退出脚本。"
            break 2
        fi

        BW=$(printf "%.0f" "${BANDWIDTHS[i]}")  # 将带宽值转换为整数
        DELAY0=$(printf "%.0f" "${DELAYS[i]}")  # 将延迟值转换为整数
        LOSS0=$(echo "${LOSS_trace[i]}" | awk '{printf "%d", $1 * 50}')


        # 检查 BW 是否在合法范围内
        if (( BW < 100 || BW > 9000 )); then
            echo "警告: 带宽值 ${BW} 不在合法范围 (100k - 9000k)，跳过该值"
            continue
        fi
        # 检查 DELAY0 是否在合法范围内
        if (( DELAY0 < 0 || DELAY0 > 1000 )); then
            echo "警告: 延迟值 ${DELAY0} 不在合法范围 (0 - 1000ms)，跳过该值"
            continue
        fi

        # 如果 DELAY0 小于 10，将其设置为 10
        if (( DELAY0 < 15 )); then
            DELAY0=15
        fi
        # 如果 DELAY0 超过 30，进行相应的计算
        if (( DELAY0 > 20 )); then
            DELAY0=$(printf "%.0f" $(( (DELAY0 - 20) * 10 / 10 + 20 )))
        fi
        # 修改现有的 netem 规则，动态设置延迟
        tc qdisc change dev lo root handle 1: netem delay ${DELAY0}ms ${JITTER}ms loss ${LOSS0}%
        # 修改现有的 tbf 规则，动态设置带宽
        tc qdisc change dev lo parent 1: handle 2: tbf rate ${BW}kbit burst 1500 limit 75000
        echo "带宽: ${BW} kbit/s, 延迟: ${DELAY0} ms, 丢包率: ${LOSS0}%" 
        # 等待0.06秒
        sleep 0.06
    done
done