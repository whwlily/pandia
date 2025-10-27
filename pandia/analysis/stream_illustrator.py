import os
import re
import time
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional
from pandia.agent.utils import divide
from pandia.constants import K, M
from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext
from pandia.context.streaming_context import StreamingContext

FIG_EXTENSION = 'png'
DPI = 600
PACKET_TYPES = ['audio', 'video', 'rtx', 'fec', 'padding']
kDeltaTick=.25  # In ms
kBaseTimeTick=kDeltaTick*(1<<8)  # In ms
kTimeWrapPeriod=kBaseTimeTick * (1 << 24)  # In ms

def illustrate_frame_ts(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    frames = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.decoding_at:
            frames.append((frame.captured_at_utc, 
                           frame.encoded_at - frame.captured_at,
                           frame.assembled_at_utc - frame.captured_at_utc,
                           frame.decoded_at_utc - frame.captured_at_utc))
    def plot(i, j):
        x = np.array([f[i] for f in frames])
        y = np.array([f[j] for f in frames])
        indexes = y > 0
        if not np.any(indexes):
            return
        plt.plot(x[indexes] - ts_min, y[indexes] * 1000)

    plt.close()
    for i in range(1, 4):
        plot(0, i)
    plt.ylim(0, 50)
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Delay (ms)')
    plt.legend(['Encoding Delay', 'Assembly Delay', 'Decoding Delay'])
    plt.savefig(os.path.join(path, 'frame-ts.pdf'))


def illustrate_frame_spec(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    encoded_size_data = []
    resolution_data = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.encoded_size > 0:
            encoded_size_data.append((frame.captured_at_utc, frame.encoded_size))
        resolution_data.append((frame.captured_at_utc, frame.height))
    if not encoded_size_data or not resolution_data:
        return
    encoded_size_data = np.array(encoded_size_data)
    resolution_data = np.array(resolution_data)
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(encoded_size_data[:, 0] - ts_min, encoded_size_data[:, 1] / K, 'b')
    ax1.set_xlabel('Frame capture time (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    indexes = resolution_data[:, 1] > 0
    ax2.plot(resolution_data[indexes, 0] - ts_min, resolution_data[indexes, 1], 'r')
    ax2.set_ylabel('Resolution (height)')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'frame-spec.pdf'))


def illustrate_frame_bitrate(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    bitrate_data = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.bitrate > 0:
            bitrate_data.append((frame.captured_at_utc, frame.bitrate))
    if not bitrate_data:
        return
    plt.close()
    plt.plot([f[0] - ts_min for f in bitrate_data], [f[1] / K for f in bitrate_data])
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Bitrate (Kbps)')
    plt.savefig(os.path.join(path, 'frame-bitrate.pdf'))

def parse_line(line, context: StreamingContext) -> dict:
    data = {}
    ts = 0
    if 'FrameCaptured' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] FrameCaptured, id: (\\d+), width: (\\d+), height: (\\d+), ts: (\\d+), utc ts: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        frame_id = int(m[2])
        width = int(m[3])
        height = int(m[4])
        utc_ts = int(m[6]) / 1000
        frame = FrameContext(frame_id, ts)
        frame.captured_at_utc = utc_ts
        context.last_captured_frame_id = frame_id
        context.frames[frame_id] = frame
        if context.utc_local_offset == 0:
            context.update_utc_local_offset(utc_ts - ts)
        [mb.on_frame_added(frame, ts) for mb in context.monitor_blocks.values()]
    elif 'UpdateFecRates' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] UpdateFecRates, fraction lost: ([0-9.]+).*'), line)
        ts = int(m[1]) / 1000
        loss_rate = float(m[2])
        context.packet_loss_data.append((ts, loss_rate))
    elif 'SetupCodec' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetupCodec, config, codec type: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        codec = int(m[2])
        if context.codec is None:
            context.codec = codec
            context.start_ts = ts
    elif 'Egress paused because of pacing rate constraint' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Egress paused because of pacing rate constraint, left packets: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        left_packets = int(m[2])
        context.pacing_queue_data.append((ts, left_packets))
    elif ' SendPacket,' in line:
        m = re.match('.*\\[(\\d+)\\].*SendPacket, id: (\\d+), seq: (\\d+)'
                     ', first in frame: (\\d+), last in frame: (\\d+), fid: (\\d+), type: (\\d+)'
                     ', rtx seq: (\\d+), allow rtx: (\\d+), size: (\\d+).*', line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        seq_num = int(m[3])
        first_in_frame = int(m[4]) == 1
        last_in_frame = int(m[5]) == 1
        frame_id = int(m[6])
        rtp_type = PACKET_TYPES[int(m[7])]  # 0: audio, 1: video, 2: rtx, 3: fec, 4: padding
        retrans_seq_num = int(m[8])
        allow_retrans = int(m[9]) != 0
        size = int(m[10])
        if rtp_id > 0:
            packet = PacketContext(rtp_id)
            packet.seq_num = seq_num
            packet.packet_type = rtp_type
            packet.frame_id = frame_id
            packet.first_packet_in_frame = first_in_frame
            packet.last_packet_in_frame = last_in_frame
            packet.allow_retrans = allow_retrans
            packet.retrans_ref = retrans_seq_num
            packet.size = size
            context.packets[rtp_id] = packet
            context.packet_id_map[seq_num] = rtp_id
            [mb.on_packet_added(packet, ts) for mb in context.monitor_blocks.values()]
            if rtp_type == 'rtx':
                packet.frame_id = context.packets[context.packet_id_map[retrans_seq_num]].frame_id
            if packet.frame_id > 0 and frame_id in context.frames:
                frame: FrameContext = context.frames[frame_id]
                frame.rtp_packets[rtp_id] = packet
                if rtp_type == 'rtx':
                    original_rtp_id = context.packet_id_map[retrans_seq_num]
                    frame.retrans_record.setdefault(original_rtp_id, []).append(rtp_id)
                if rtp_type == 'video':
                    frame.sequence_range[0] = min(frame.sequence_range[0], seq_num)
                    frame.sequence_range[1] = max(frame.sequence_range[1], seq_num)
    elif 'OnSentPacket' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] OnSentPacket, id: (-?\\d+), type: (\\d+), size: (\\d+), utc: (\\d+) ms.*'), line)
        ts = int(m[1]) / 1000
        rtp_id = int(m[2])
        payload_type = int(m[3])
        size = int(m[4])
        utc = int(m[5]) / 1000
        if rtp_id > 0:
            packet: PacketContext = context.packets[rtp_id]
            packet.payload_type = payload_type
            packet.size = size
            packet.sent_at = ts 
            packet.sent_at_utc = utc
            context.last_egress_packet_id = max(rtp_id, context.last_egress_packet_id)
            [mb.on_packet_sent(packet, context.frames.get(packet.frame_id, None), ts) for mb in context.monitor_blocks.values()]
    elif 'RTCP feedback,' in line:
        m = re.match(re.compile('.*\\[(\\d+)\\] RTCP feedback.*'), line)
        ts = int(m[1]) / 1000
        ms = re.findall(r'packet (acked|lost): (\d+) at (\d+) ms', line)
        pkt_pre = None
        for ack_type, rtp_id, received_at in ms:
            # The recv time is wrapped by kTimeWrapPeriod.
            # The fixed value 1570 should be calculated according to the current time.
            offset = int(time.time() * 1000 / kTimeWrapPeriod) - 1
            received_at = (int(received_at) + offset * kTimeWrapPeriod) / 1000
            rtp_id = int(rtp_id)
            # received_at = int(received_at) / 1000
            if rtp_id in context.packets:
                packet = context.packets[rtp_id]
                packet.received_at_utc = received_at
                packet.received = ack_type == 'acked'
                packet.acked_at = ts
                context.last_acked_packet_id = \
                    max(rtp_id, context.last_acked_packet_id)
                [mb.on_packet_acked(packet, pkt_pre, ts) for mb in context.monitor_blocks.values()]
                if packet.received:
                    pkt_pre = packet
    elif 'SetProtectionParameters' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] SetProtectionParameters, delta: (\\d+), key: (\\d+).*'), line)
        ts = int(m[1]) / 1000
        fec_delta = int(m[2])
        fec_key = int(m[3])
        context.fec.fec_key_data.append((ts, fec_key))
        context.fec.fec_delta_data.append((ts, fec_delta))
    elif 'NTP response' in line:
        m = re.match(re.compile(
            '.*NTP response: precision: (-?[.0-9]+), offset: (-?[.0-9]+), rtt: (-?[.0-9]+).*'), line)
        precision = float(m[1])
        offset = float(m[2])
        rtt = float(m[3])
        context.utc_offset = offset
    elif 'Update target bitrate' in line:
        m = re.match(re.compile(
            '.*\\[(\\d+)\\] Update target bitrate: (\\d+) kbps.*'), line)
        ts = int(m[1]) / 1000  # 提取时间戳
        target_bitrate = int(m[2]) / 1000  # 提取比特率
        context.target_bitrate_data.append((ts, target_bitrate))
    
    if line[0] == "[":
        m = re.match(re.compile('.*\\[(\\d+)\\].*'), line)
        ts = int(m[1]) / 1000
    if ts:
        context.update_ts(ts)
    return data

def analyze_frame(context: StreamingContext, output_dir: str, true_capacity) -> None:
    frame_id_list = list(sorted(context.frames.keys()))
    frames = [context.frames[i] for i in frame_id_list]
    frames_encoded: List[FrameContext] = list(filter(lambda f: f.encoded_size > 0, frames))
    frames_decoded: List[FrameContext] = list(filter(lambda f: f.decoded_at > context.start_ts, frames))
    frames_captured_ts = np.array([f.captured_at - context.start_ts for f in frames_encoded])
    frames_decoded_ts = np.array([f.decoded_at - context.start_ts for f in frames_decoded])
    frames_dropped = list(filter(lambda f: f.encoded_size <= 0, frames))
    frames_key = list(filter(lambda f: f.is_key_frame, frames))
    if (len(frames_encoded) == 0):
        print('ERROR: No frame encoded.')
        return

    plt.close()
    plt.plot([f.captured_at - context.start_ts for f in frames], [f.frame_id for f in frames], '.')
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Frame ID')
    plt.savefig(os.path.join(output_dir, f'mea-frame-id.{FIG_EXTENSION}'), dpi=DPI)

    # 绘制相邻两帧解码的间隔时间图
    frame_intervals = [(frames_decoded[i].decoded_at - frames_decoded[i-1].decoded_at) * 1000 for i in range(1, len(frames_decoded))]
    plt.close()
    plt.plot([frames_decoded[i].decoded_at - context.start_ts for i in range(1, len(frames_decoded))], frame_intervals)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Frame Interval (ms)')
    plt.title('Frame Decoding Intervals')

    frame_intervals = [x for x in frame_intervals if x < 1000 and x > 0] 
    np.save(os.path.join(output_dir, 'frame_intervals.npy'), np.array(frame_intervals))
    
    # for i in range(min(5, len(frame_intervals))):
    #     if frame_intervals[i] > 200:
    #         frame_intervals[i] = 200

    total_time = sum(frame_intervals)
    normal_time = len(frame_intervals) * 40
    frame_intervals_120 = [x for x in frame_intervals if x > 120] 
    frame_intervals_150 = [x for x in frame_intervals if x > 150]
    if total_time > 0:
        stall_rate = (total_time - normal_time) / total_time * 100
        stall_120 = (sum(frame_intervals_120) - len(frame_intervals_120) * 40) / total_time * 100
        stall_150 = (sum(frame_intervals_150) - len(frame_intervals_150) * 40) / total_time * 100

    # 标出 frame_intervals 中值大于 120 的个数
    count_above_120 = sum(1 for interval in frame_intervals if interval > 120) / len(frame_intervals) * 100
    count_above_150 = sum(1 for interval in frame_intervals if interval > 150) / len(frame_intervals) * 100
    plt.annotate(f'Freeze rate(120ms): {count_above_120:.2f}%', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    plt.annotate(f'Freeze rate(150ms): {count_above_150:.2f}%', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    with open(os.path.join(output_dir, 'freeze.txt'), 'a') as f:
        f.write(f"Freeze rate(120ms): {count_above_120:.2f}%\n")
        f.write(f"Freeze rate(150ms): {count_above_150:.2f}%\n")
        f.write(f"Stall rate: {stall_rate:.2f}%\n")
        f.write(f"Stall rate(120ms): {stall_120:.2f}%\n")
        f.write(f"Stall rate(150ms): {stall_150:.2f}%\n")
    plt.savefig(os.path.join(output_dir, 'frame_decoding_intervals.png'), dpi=DPI)

    # 绘制相邻两帧解码间隔时间的CDF图
    plt.close()
    cdf_x = list(sorted(frame_intervals))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Frame Interval (ms)')
    plt.ylabel('CDF')
    plt.title('CDF of Frame Decoding Intervals')
    # 标出10-90分位数的值
    for i in np.arange(95, 100, 0.5):
        pi = np.percentile(cdf_x, i)
        plt.axvline(pi, color='g', linestyle='--', label=f'{i}th: {pi:.2f}ms')

    plt.legend()
    plt.savefig(os.path.join(output_dir, 'frame_decoding_intervals_cdf.png'), dpi=DPI)

    decoding_delay = np.array([f.decoding_delay(context.utc_offset) * 1000 for f in frames_encoded])
    decoding_queue_delay = np.array([f.decoding_queue_delay(context.utc_offset) * 1000 for f in frames_encoded])
    assemble_delay = np.array([f.assemble_delay(context.utc_offset) * 1000 for f in frames_encoded])
    pacing_delay = np.array([f.pacing_delay() * 1000 for f in frames_encoded])
    pacing_rtx_delay = np.array([f.pacing_delay_including_rtx() * 1000 for f in frames_encoded])
    encoding_delay = np.array([f.encoding_delay() * 1000 for f in frames_encoded])
    encoding_delay0 = np.array([f.encoding_delay0() * 1000 for f in frames_encoded])
    decoding_delay0 = np.array([f.decoding_delay0() * 1000 for f in frames_encoded])
    stacked = np.vstack([decoding_delay, pacing_delay, pacing_rtx_delay, assemble_delay, decoding_queue_delay, decoding_delay0, encoding_delay0, encoding_delay])
    max_ylim = min(np.max(stacked), 3000)
    plt.close()
    plt.ylim(top=max_ylim)
    plt.plot(frames_captured_ts[encoding_delay >= 0], encoding_delay[encoding_delay >= 0], 'm')
    plt.plot(frames_captured_ts[pacing_delay >= 0], pacing_delay[pacing_delay >= 0], 'y')
    plt.plot(frames_captured_ts[pacing_rtx_delay >= 0], pacing_rtx_delay[pacing_rtx_delay >= 0], 'c')
    plt.plot(frames_captured_ts[assemble_delay >= 0], assemble_delay[assemble_delay >= 0], 'r')
    plt.plot(frames_captured_ts[decoding_queue_delay >= 0], decoding_queue_delay[decoding_queue_delay >= 0], 'g')
    plt.plot(frames_captured_ts[decoding_delay >= 0], decoding_delay[decoding_delay >= 0], 'b')
    plt.legend(['Encoded', 'Pacing', 'Pacing (RTX)', 'Transmission', 'Decoding queue', 'Decoded'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.savefig(os.path.join(output_dir, f'mea-delay-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.ylim(top=max_ylim)
    plt.plot(frames_captured_ts[decoding_delay >= 0], decoding_delay[decoding_delay >= 0], 'b')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('G2G Delay (ms)')
    plt.savefig(os.path.join(output_dir, f'mea-delay-g2g.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.ylim(top=max(np.max(decoding_delay0),np.max(encoding_delay0)))
    plt.plot(frames_captured_ts[encoding_delay0 >= 0], encoding_delay0[encoding_delay0 >= 0])
    plt.plot(frames_captured_ts[decoding_delay0 >= 0], decoding_delay0[decoding_delay0 >= 0])
    plt.legend(['Encoding', 'Decoding'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.savefig(os.path.join(output_dir, f'mea-delay-endec.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.ylim(top=max_ylim)
    plt.plot(frames_captured_ts[decoding_delay >= 0], decoding_delay[decoding_delay >= 0])
    plt.plot(frames_captured_ts[decoding_queue_delay >= 0], decoding_queue_delay[decoding_queue_delay >= 0])
    plt.plot(frames_captured_ts[encoding_delay >= 0], encoding_delay[encoding_delay >= 0])
    plt.legend(['Decoded', 'Decoding queue', 'Encoded'])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.savefig(os.path.join(output_dir, f'mea-delay-codec.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.ylim(top=max_ylim)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Delay (ms)')
    plt.plot(frames_captured_ts[assemble_delay >= 0], assemble_delay[assemble_delay >= 0], 'r')
    plt.plot(frames_captured_ts[pacing_rtx_delay >= 0], pacing_rtx_delay[pacing_rtx_delay >= 0], '-.g')
    plt.plot(frames_captured_ts[pacing_delay >= 0], pacing_delay[pacing_delay >= 0], '-b')
    plt.legend(['Transmission', 'Paced (RTX)', 'Paced'])
    plt.savefig(os.path.join(output_dir, f'mea-delay-trans.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.ylim(top=max_ylim)
    plt.plot(frames_captured_ts[1:], (frames_captured_ts[1:] - frames_captured_ts[:-1]) * 1000)
    plt.plot(frames_decoded_ts[1:], (frames_decoded_ts[1:] - frames_decoded_ts[:-1]) * 1000, alpha=0.7)
    plt.legend(['Encoding', 'Decoding'])
    plt.ylim([0, 100])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Interval (ms)')
    plt.savefig(os.path.join(output_dir, f'mea-frame-interval.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    res_list = [f.encoded_shape[1] for f in frames_encoded]
    res_list = list(sorted(set(res_list)))
    plt.plot(frames_captured_ts, [res_list.index(f.encoded_shape[1]) for f in frames_encoded])
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Resolution')
    plt.yticks(range(len(res_list)), [str(r) for r in res_list])
    plt.savefig(os.path.join(output_dir, f'mea-resolution-frame.{FIG_EXTENSION}'), dpi=DPI)
    
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(frames_captured_ts, [f.encoded_size / 1024 for f in frames_encoded], 'b.')
    ax1.plot([f.captured_at - context.start_ts for f in frames_dropped], [10 for _ in frames_dropped], 'xr')
    ax1.plot([f.captured_at - context.start_ts for f in frames_key], [f.encoded_size / 1024 for f in frames_key], '^g')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.legend(['Encoded size', 'Dropped frames', 'Key frames'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mea-size-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    frame_rtp_num_list = np.array([len(f.packets_video()) for f in frames_encoded])
    frame_rtp_lost_num_list = np.array([len([p for p in f.packets_video() if p.received is False]) for f in frames_encoded])
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(frames_captured_ts, divide(frame_rtp_lost_num_list, frame_rtp_num_list) * 100, 'b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.plot(frames_captured_ts, frame_rtp_lost_num_list, 'r.')
    plt.xlabel('Timestamp (s)')
    ax1.set_ylabel('Packet loss rate per frame (%)')
    ax2.set_ylabel('Number of lost packets per frame')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mea-loss-packet-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    for duration in [.1, .5, 1]:
        bucks = int((frames_encoded[-1].encoding_at - context.start_ts) / duration + 1)
        data = np.zeros((bucks, 1))
        for frame in frames_encoded:
            if frame.encoded_size > 0:
                i = int((frame.encoding_at - context.start_ts) / duration)
                data[i] += frame.encoded_size
        plt.plot(np.arange(bucks) * duration, data * 8 / duration / M)
    plt.ylabel('Rates (Mbps)')
    plt.xlabel('Timestamp (s)')
    plt.legend(['100ms', '500ms', '1s'])
    plt.savefig(os.path.join(output_dir, f'mea-bitrate.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    for duration in [1]:
        bucks = int((frames_encoded[-1].encoding_at - context.start_ts) / duration + 1)
        data = np.zeros((bucks, 1))
        for frame in frames_encoded:
            if frame.encoded_size > 0:
                i = int((frame.encoding_at - context.start_ts) / duration)
                data[i] += frame.encoded_size
        np.save(os.path.join(output_dir, 'bitrate.npy'), data * 8 / duration / M)
        plt.plot(np.arange(bucks) * duration, data * 8 / duration / M)
    # 对 true_capacity 进行抽样
    sampled_true_capacity = np.interp(np.arange(bucks) * duration, np.linspace(0, (bucks - 1) * duration, len(true_capacity)), true_capacity)
    # 确保 sampled_true_capacity 中没有零值
    sampled_true_capacity[sampled_true_capacity == 0] = np.nan

    # 计算带宽利用率
    bandwidth_utilization = np.minimum((data * 8 / duration / M) / sampled_true_capacity, 1)
    avg_bandwidth_utilization = np.nanmean(bandwidth_utilization) * 100 # 使用 nanmean 忽略 NaN 值
    # 将带宽利用率追加到txt文件
    with open(os.path.join(output_dir, 'score.txt'), 'a') as f:
        f.write(f"Bandwidth Utilization: {avg_bandwidth_utilization:.2f}%\n")

    np.save(os.path.join(output_dir, 'true_capacity.npy'), sampled_true_capacity)
    plt.plot(np.arange(bucks) * duration, sampled_true_capacity, label='True Capacity')
    plt.legend()
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Rate (Mbps)')
    plt.savefig(os.path.join(output_dir, f'mea-true_capacity&bitrate.{FIG_EXTENSION}'), dpi=DPI)


    plt.close()
    duration = .5
    bucks = int((frames_encoded[-1].encoding_at - context.start_ts) / duration + 1)
    data = np.zeros((bucks, 1))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Action bitrate (Mbps)')
    drl_bitrate = np.array(context.drl_bitrate)
    # ax1.plot((drl_bitrate[:, 0] - context.start_ts), drl_bitrate[:, 1] / 1024, 'b--')
    ax1.plot([f.encoding_at - context.start_ts for f in frames_encoded], [f.bitrate / M for f in frames_encoded])
    # ax1.plot(np.arange(bucks) * duration, data * 8 / duration / M, '.b')
    # ax1.legend(['Encoding bitrate', 'Encoded bitrate'])
    # ax1.tick_params(axis='y', labelcolor='b')
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Encoding FPS', color='r')
    # ax2.plot([f.encoding_at - context.start_ts for f in frames_encoded], [f.fps for f in frames_encoded], 'r')
    # ax2.tick_params(axis='y', labelcolor='r')
    plt.savefig(os.path.join(output_dir, f'set-codec-params.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.plot([f.encoded_at - context.start_ts for f in frames_encoded], [f.qp for f in frames_encoded], '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('QP')
    plt.savefig(os.path.join(output_dir, f'rep-qp-frame.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    for duration in [.1, .5, 1]:
        bucks = int((frames_encoded[-1].captured_at - context.start_ts) / duration + 1)
        data = np.zeros(bucks)
        for f in frames_encoded:
            data[int((f.captured_at - context.start_ts) / duration)] += 1
        plt.plot(np.arange(bucks) * duration, data / duration, '-x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FPS')
    plt.legend(['100ms', '200ms', '1s'])
    plt.savefig(os.path.join(output_dir, f'mea-fps-encoded.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    for duration in [.1, .5, 1]:
        bucks = int((frames_decoded[-1].decoded_at - context.start_ts) / duration + 1)
        data = np.zeros(bucks)
        for f in frames_decoded:
            data[int((f.decoded_at - context.start_ts) / duration)] += 1
        plt.plot(np.arange(bucks) * duration, data / duration, '-x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FPS')
    plt.legend(['100ms', '200ms', '1s'])
    plt.savefig(os.path.join(output_dir, f'mea-fps-decoded.{FIG_EXTENSION}'), dpi=DPI)


def analyze_packet(context: StreamingContext, output_dir: str, log_context: Optional[StreamingContext] = None) -> None:
    data_ack = []
    data_recv = []
    print(f'Analyzing {len(context.packets)} packets...')
    for pkt in sorted(context.packets.values(), key=lambda x: x.sent_at):
        pkt: PacketContext = pkt
        if pkt.sent_at < context.start_ts:
            continue
        if pkt.ack_delay() > 0:
            data_ack.append((pkt.sent_at, pkt.ack_delay()))
        if pkt.recv_delay() != -1:
            data_recv.append((pkt.sent_at, pkt.recv_delay() - context.utc_offset))
    if len(data_ack) == 0:
        print('ERROR: No packet ACKed.')
        return
    plt.close()
    x = [(d[0] - context.start_ts) for d in data_recv]
    y = [d[1] * 1000 for d in data_recv]

    # y_cal = [d[1] * 1000 for d in data_recv if 0 <= d[1] * 1000 <= 1000]
    np.save(os.path.join(output_dir, 'pkt_delay.npy'), y)
    y_cal = [d[1] * 1000 for d in data_recv if 0 <= d[1] * 1000 <= 2900]
    # 计算最大值、最小值和第95分位数值
    if y_cal:
        max_value = max(y_cal)
        min_value = min(y_cal)
        mean_value = np.mean(y_cal)
        percentile_95 = np.percentile(y_cal, 95)
        s_delay = 100 * (max_value - percentile_95) / (max_value - min_value)
        # 将数据追加到txt文件
        with open(os.path.join(output_dir, 'score.txt'), 'a') as f:
            print(f"Delay: Max: {max_value}, Min: {min_value}, 95th Percentile: {percentile_95}, S_Delay: {s_delay}")
            f.write(f"Max: {max_value}, Min: {min_value}, 95th Percentile: {percentile_95}, Mean: {mean_value}\n")
            f.write(f"Delay: {s_delay}\n")

    plt.plot(x, y, '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet transmission delay (ms)')
    if y:
        plt.ylim([min(y), min(max(y), 1000)])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet-recv-biased.{FIG_EXTENSION}'), dpi=DPI)
    plt.close()
    
    cdf_x = list(sorted([d[1] * 1000 for d in data_recv]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet receive delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, min(max(cdf_x),1000)])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet-recv-cdf.{FIG_EXTENSION}'), dpi=DPI)
    plt.close()
    
    plt.plot([(d[0] - context.start_ts) for d in data_ack], 
             [d[1] * 1000 for d in data_ack], 'x')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('RTT (ms)')
    plt.ylim([0, min(max(cdf_x), 1000)])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet-ack.{FIG_EXTENSION}'), dpi=DPI)
    plt.close()

    cdf_x = list(sorted([d[1] * 1000 for d in data_ack]))
    cdf_y = np.arange(len(cdf_x)) / len(cdf_x)
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Packet ACK delay (ms)')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, min(max(cdf_x), 1000)])
    plt.savefig(os.path.join(output_dir, f'mea-delay-packet-ack-cdf.{FIG_EXTENSION}'), dpi=DPI)

    packets = sorted(context.packets.values(), key=lambda x: x.sent_at)
    for duration in [1, .1, .01, .001]:
        plt.close()
        bucks = int((packets[-1].sent_at - context.start_ts) / duration + 1)
        data = np.zeros(bucks)
        for p in packets:
            if p.sent_at >= context.start_ts:
                data[int((p.sent_at - context.start_ts) / duration)] += p.size
        plt.plot(np.arange(bucks) * duration, data / duration * 8 / 1024 / 1024, '.')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Egress rate (Mbps)')
        plt.savefig(os.path.join(output_dir, f'mea-egress-rate-{int(duration * 1000)}.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    data = [[[], []], [[], []], [[], []]]  # [rtp_video, rtp_rtx, rtp_fec]
    for f in context.frames.values():
        for pkt in f.rtp_packets.values():
            d = None
            if pkt.packet_type == 'video':
                d = data[0]
            elif pkt.packet_type == 'rtx':
                d = data[1]
            elif pkt.packet_type == 'fec':
                d = data[2]
            if d and pkt.sent_at > 0:
                d[0].append(f.captured_at - context.start_ts)
                d[1].append((pkt.sent_at - f.encoded_at) * 1000)
    for d, m in zip(data, ['.', '8', '^']):
        plt.plot(d[0], d[1], m)
    plt.xlabel('Frame timestamp (s)')
    plt.ylabel('RTP egress timestamp (ms)')
    # plt.ylim([0, 150])
    plt.legend(['Video', 'RTX', 'FEC'])
    plt.savefig(os.path.join(output_dir, f'mea-rtp-pacing-ts.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    context = log_context if log_context else context
    x = [i[0] - context.start_ts for i in context.packet_loss_data]
    y = [i[1] * 100 for i in context.packet_loss_data]
    plt.plot(x, y, '.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet loss rate (%)')
    plt.savefig(os.path.join(output_dir, f'rep-loss-packet.{FIG_EXTENSION}'), dpi=DPI)
    
    duration = 1
    packets = sorted(context.packets.values(), key=lambda x: x.sent_at)
    bucks_sent = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    bucks_lost = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    print(f'Analyzing {len(packets)} packets for loss..., duration: {packets[-1].sent_at - packets[0].sent_at}, buckets: {bucks_sent.shape}')
    for p in packets:
        i = int((p.sent_at - packets[0].sent_at) / duration)
        if p.rtp_id >= 0 and p.received is not None:
            bucks_sent[i] += 1
            if not p.received:
                bucks_lost[i] += 1
    x = [i * duration for i in range(len(bucks_sent))]
    y = divide(bucks_lost, bucks_sent) * 100
    np.save(os.path.join(output_dir, 'packet_loss.npy'), y)
    with open(os.path.join(output_dir, 'score.txt'), 'a') as f:
        f.write(f"Loss: {np.mean(100 * (1 - y/100))}\n")
    plt.close()
    plt.plot(x, y)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet loss rate (%)')
    plt.savefig(os.path.join(output_dir, f'mea-loss-packet.{FIG_EXTENSION}'), dpi=DPI)

    duration = 1
    packets = sorted(context.packets.values(), key=lambda x: x.sent_at)
    bucks_sent = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    bucks_retrans = np.zeros(int((packets[-1].sent_at - packets[0].sent_at) / duration + 1))
    print(f'Analyzing {len(packets)} packets for retransmission..., duration: {packets[-1].sent_at - packets[0].sent_at}, buckets: {bucks_sent.shape}')
    for p in packets:
        i = int((p.sent_at - packets[0].sent_at) / duration)
        if p.rtp_id >= 0:
            bucks_sent[i] += 1
            if p.packet_type == 'rtx':
                bucks_retrans[i] += 1
    x = [i * duration for i in range(len(bucks_sent))]
    y = divide(bucks_retrans, bucks_sent) * 100
    np.save(os.path.join(output_dir, 'packet_retransmission.npy'), y)
    with open(os.path.join(output_dir, 'score.txt'), 'a') as f:
        f.write(f"Retrans: {np.mean(100 * (1 - y/100))}\n")
    plt.close()
    plt.plot(x, y)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Packet retransmission ratio (%)')
    plt.savefig(os.path.join(output_dir, f'mea-retrans-packet.{FIG_EXTENSION}'), dpi=DPI)




def analyze_network(context: StreamingContext, output_dir: str, log_context: Optional[StreamingContext] = None) -> None:
    if len(context.packets) == 0:
        print('ERROR: No packet sent.')
        return
    data = context.networking.pacing_rate_data
    x = [d[0] - context.start_ts for d in data]
    y = [d[1] / M for d in data]
    yy = [d[2] / M for d in data]
    if len(data) == 0:
        print('ERROR: No network data.')
        return
    plt.close()
    plt.plot(x, y, '.')
    plt.plot(x, yy, '.')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Rate (Mbps)")
    plt.legend(["Pacing rate", "Padding rate"])
    plt.savefig(os.path.join(output_dir, f'set-pacing-rate.{FIG_EXTENSION}'), dpi=DPI)
    ts_min = context.start_ts
    ts_max = max([p.sent_at for p in context.packets.values()])
    ts_range = ts_max - ts_min
    period = .1
    buckets = np.zeros(int(ts_range / period + 1))
    for p in context.packets.values():
        ts = (p.sent_at - ts_min)
        if ts >= 0:
            buckets[int(ts / period)] += p.size
    buckets = buckets / period * 8 / 1024 / 1024  # mbps
    plt.close()
    plt.plot(np.arange(len(buckets)) * period, buckets, '.')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTP egress rate (Mbps)")
    plt.savefig(os.path.join(output_dir, f'mea-sending-rate.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    x = [r[0] - context.start_ts for r in context.rtt_data]
    y = [r[1] * 1000 for r in context.rtt_data]
    plt.plot(x, y, '.')
    plt.xlabel("Timestamp (s)")
    plt.ylabel("RTT (ms)")
    plt.savefig(os.path.join(output_dir, f'rep-rtt.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    context = log_context if log_context else context
    x = [d[0] - context.start_ts for d in context.pacing_queue_data]
    y = [d[1] for d in context.pacing_queue_data]
    plt.plot(x, y)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Pacing queue size (packets)")
    plt.savefig(os.path.join(output_dir, f'mea-pacing-queue.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    context = log_context if log_context else context
    x = [d[0] - context.start_ts for d in context.target_bitrate_data]
    y = [d[1] for d in context.target_bitrate_data]
    plt.plot(x, y)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Target bitrate(mbps)")
    plt.savefig(os.path.join(output_dir, f'mea-target-bitrate.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_key_data],
             [d[1] for d in context.fec.fec_key_data], '--')
    plt.plot([d[0] - context.start_ts for d in context.fec.fec_delta_data],
             [d[1] for d in context.fec.fec_delta_data], '-.')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('FEC Ratio')
    plt.legend(['Key frame FEC', 'Delta frame FEC'])
    plt.savefig(os.path.join(output_dir, f'set-fec.{FIG_EXTENSION}'), dpi=DPI)



def setup_diagrams_path(path):
    os.makedirs(path, exist_ok=True)
    # for f in os.listdir(path):
    #     os.remove(os.path.join(path, f))

def retrieve_scores(output_dir: str) -> dict:
    scores = {}
    score_file = os.path.join(output_dir, 'score.txt')
    if not os.path.exists(score_file):
        print(f"ERROR: {score_file} does not exist.")
        return scores

    with open(score_file, 'r') as f:
        for line in f:
            if "Delay:" in line:
                match = re.search(r"Delay: (\d+\.?\d*)", line)
                if match:
                    scores['S_Delay'] = float(match.group(1))
            elif "Loss:" in line:
                match = re.search(r"Loss: (\d+\.?\d*)", line)
                if match:
                    scores['S_Loss'] = float(match.group(1))
            elif "Bandwidth Utilization:" in line:
                match = re.search(r"Bandwidth Utilization: (\d+\.?\d*)%", line)
                if match:
                    scores['S_U'] = float(match.group(1))

    with open(os.path.join(output_dir, 'score.txt'), 'a') as f:
        # 确保所有必需的分数都存在，如果不存在则使用默认值
        s_delay = scores.get('S_Delay', 0.0)
        s_loss = scores.get('S_Loss', 0.0)
        s_u = scores.get('S_U', 0.0)
        f.write(f"Score: {0.2 * s_delay + 0.3 * s_loss + 0.2 * s_u}\n")
    return scores

# 示例调用


def generate_diagrams(path, context, log_file, true_capacity):
    log_context = StreamingContext()
    for line in open(log_file).readlines():
        try:
            parse_line(line, log_context)
        except:
            pass
    setup_diagrams_path(path)
    # illustrate_frame_ts(path, context)
    # illustrate_frame_spec(path, context)
    # illustrate_frame_bitrate(path, context)
    analyze_frame(context, path, true_capacity)
    analyze_packet(context, path, log_context)
    analyze_network(context, path, log_context)
    retrieve_scores(path)




