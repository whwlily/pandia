from pandia.log_analyzer_sender import StreamingContext
from pandia.constants import M
import numpy as np

REWARD_MIN = -200
REWARD_MAX = 200


# def reward(context: StreamingContext, net_sample, terminated=False, actions=list()):
#     monitor_durations = list(sorted(context.monitor_blocks.keys()))
#     mb = context.monitor_blocks[monitor_durations[0]]
#     fps_score = 0
#     # fps_score = - (1 - np.clip(mb.frame_fps_decoded / 20, 0, 1)) ** 2
#     # if mb.frame_fps_decoded < 20:
#     #     return REWARD_MIN

#     delay_score = 0
#     mean_delay = np.mean([delay for delay in [mb.frame_decoded_delay]])
#     for delay in [mb.frame_decoded_delay]:
#         delay = max(0, delay) * 1000
#         delay_score += - (delay / 100) ** 2
#     quality_score = 2 * max(mb.frame_bitrate / M, 0)
#     res_score = 0
#     # if penalty == 0:
#     #     self.termination_ts = 0
#     # if penalty > 0 and self.termination_ts == 0:
#     #     # If unexpected situation lasts for 5s, terminate
#     #     self.termination_ts = self.context.last_ts + self.termination_timeout
#     # if mb.frame_fps < 1:
#     #     penalty = 100
#     stability_score = 0
#     # if len(actions) >= 2:
#     #     bitrates = [a.bitrate for a in actions[:-10]]
#     #     stability_score = -np.std(bitrates) / M * 16
#     score = res_score + quality_score + fps_score + delay_score + stability_score
#     score = np.clip(score, REWARD_MIN, REWARD_MAX)
#     return score, {"mean_delay": mean_delay, "bitrate": mb.frame_bitrate / M}


def reward(context: StreamingContext, net_sample, terminated=False, actions=list()):
    monitor_durations = list(sorted(context.monitor_blocks.keys()))
    # 安全地访问monitor_blocks，避免索引越界
    if len(monitor_durations) > 1:
        mb = context.monitor_blocks[monitor_durations[1]]
    else:
        mb = context.monitor_blocks[monitor_durations[0]]
    fps_score = 0
    # fps_score = - (1 - np.clip(mb.frame_fps_decoded / 20, 0, 1)) ** 2
    # if mb.frame_fps_decoded < 20:
    #     return REWARD_MIN
    mean_delay = np.mean([delay for delay in [mb.frame_decoded_delay]])
    mean_bitrate = max(mb.frame_bitrate / M, 0)
    # delay_score = 0
    # for delay in [mb.frame_decoded_delay]:
    #     delay = max(0, delay) * 1000
    #     delay_score += - (delay / 100) ** 2
    # quality_score = 2 * max(mb.frame_bitrate / M, 0)
    # res_score = 0
    # # if penalty == 0:
    # #     self.termination_ts = 0
    # # if penalty > 0 and self.termination_ts == 0:
    # #     # If unexpected situation lasts for 5s, terminate
    # #     self.termination_ts = self.context.last_ts + self.termination_timeout
    # # if mb.frame_fps < 1:
    # #     penalty = 100
    # stability_score = 0
    # # if len(actions) >= 2:
    # #     bitrates = [a.bitrate for a in actions[:-10]]
    # #     stability_score = -np.std(bitrates) / M * 16
    # score = res_score + quality_score + fps_score + delay_score + stability_score
    # score = np.clip(score, REWARD_MIN, REWARD_MAX)

    mean_pkt_bitrate = np.clip(mb.receiving_rate / M, 0 , 3)
    mean_pkt_delay =np.clip(np.mean([max(0, delay) for delay in [mb.queuing_delay]]), 0, 467.5)

    bit_score = np.log2(mean_pkt_bitrate + 1) * 4
    d_score = mean_pkt_delay / 100.0
    if  mean_pkt_delay >= 20 and mean_pkt_delay < 60: # 20-60
        d_score = mean_pkt_delay / 50.0 + 1     # 40
    elif mean_pkt_delay >= 60 and mean_pkt_delay < 150: # 60-120
        d_score = mean_pkt_delay / 25.0         # 20
    elif mean_pkt_delay >= 150:
        d_score = 10
    r = bit_score - d_score - 10 * mb.pkt_loss_rate

    # r = np.clip(r, REWARD_MIN, REWARD_MAX)
    # 确保返回的是float32类型的标量值
    return float(r)