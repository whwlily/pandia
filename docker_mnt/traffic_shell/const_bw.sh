BANDWIDTH=$1  # 第一个参数为带宽
DELAY=$2      # 第二个参数为延迟
LOSS=$3       # 第三个参数为丢包率

echo "设置带宽为 ${BANDWIDTH}kbit/s, 延迟为 ${DELAY}ms, 丢包率 ${LOSS}"

tc qdisc del dev lo root 2> /dev/null
tc qdisc add dev lo root handle 1: netem delay ${DELAY}ms
tc qdisc add dev lo parent 1: handle 2: tbf rate ${BANDWIDTH}kbit burst 1500 latency 100ms