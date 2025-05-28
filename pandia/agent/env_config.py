# All bitrates are in bps, all delays are in s,
# all loss rates are in %, all data size are in bytes

from pandia.constants import K, M

RESOLUTION_LIST = [144, 240, 360, 480, 720, 960, 1080, 1440, 2160]
MIN_BW = 100 * K

NORMALIZATION_RANGE = {
    'bandwidth': [0, 200 * M],
    'prediction_bandwidth':[0 * K, 100 * M],
    'bitrate': [0 * K, 10 * M],
    'fec_rate': [0, 255],
    'fps': [1, 60],
    'delay': [0, 3],
    'qp': [0, 255],
    'frame_size': [0, 100 * M],
    'loss_rate': [0, 1],
    'delay_ms': [0, 3000],
}

BOUNDARY = {
    # For action
    'fake': [0, 1], # placeholder
    'prediction_bandwidth': NORMALIZATION_RANGE['prediction_bandwidth'],
    'bitrate': NORMALIZATION_RANGE['bitrate'],
    'fps': NORMALIZATION_RANGE['fps'],
    # Limited by the software implementation,
    # The maximum egress rate of WebRTC is around 200 Mbps
    'pacing_rate': NORMALIZATION_RANGE['bandwidth'],
    'padding_rate': NORMALIZATION_RANGE['bandwidth'],
    'fec_rate_key': NORMALIZATION_RANGE['fec_rate'],
    'fec_rate_delta': NORMALIZATION_RANGE['fec_rate'],
    'resolution': RESOLUTION_LIST,

    # For observation
    'frame_encoding_delay': NORMALIZATION_RANGE['delay'],
    'frame_egress_delay': NORMALIZATION_RANGE['delay'],
    'frame_recv_delay': NORMALIZATION_RANGE['delay'],
    'frame_decoding_delay': NORMALIZATION_RANGE['delay'],
    'frame_decoded_delay': NORMALIZATION_RANGE['delay'],
    'frame_fps': NORMALIZATION_RANGE['fps'],
    'frame_fps_decoded': NORMALIZATION_RANGE['fps'],
    'frame_qp': NORMALIZATION_RANGE['qp'],
    'frame_height': RESOLUTION_LIST,
    'frame_encoded_height': RESOLUTION_LIST,
    'frame_size': NORMALIZATION_RANGE['frame_size'],
    'frame_bitrate': NORMALIZATION_RANGE['bandwidth'],
    'frame_key_count': [0, 10],
    'pkt_egress_rate': NORMALIZATION_RANGE['bandwidth'],
    'pkt_ack_rate': NORMALIZATION_RANGE['bandwidth'],
    'pkt_loss_rate': NORMALIZATION_RANGE['loss_rate'],
    'pkt_trans_delay': NORMALIZATION_RANGE['delay'],
    'pkt_delay_interval': [0, 10],
    'action_gap': [- 100 * M, 100 * M],
    'bandwidth': NORMALIZATION_RANGE['bandwidth'],
    # Bandwidth Estimation Challenge data
    "receiving_rate": [0, 200 * M],
    "received_packets_num": [0, 1000000],
    "received_bytes": [0, 200 * M],
    "queuing_delay": NORMALIZATION_RANGE['delay_ms'],
    "delay": NORMALIZATION_RANGE['delay_ms'],
    "mini_seen_delay": NORMALIZATION_RANGE['delay_ms'],
    "delay_ratio": [0, 1000000],
    "delay_avg_mini_diff": NORMALIZATION_RANGE['delay_ms'],
    "pkt_received_interval": NORMALIZATION_RANGE['delay_ms'],
    "pkt_received_jitter": NORMALIZATION_RANGE['delay_ms'],
    "pkt_loss_ratio": NORMALIZATION_RANGE['loss_rate'],
    "pkt_loss_num": [0, 1000000]
}

NETWORK_SETTING = {
    'bandwidth': [MIN_BW, 200 * M],
    'delay': [0, .02],
    'loss': [0, 0],
}

ACTION_LIMIT = {
    'bitrate': [MIN_BW, 200 * M],
}

VIDEO_SOURCE = {
    'resolution': 720,
    'fps': 25,
}

GYM_SETTING = {
    # 'step_duration': .6,
    'step_duration': .1,
    'startup_delay': 1,
    'action_cap': 1, # Cap the bitrate action by the bandwidth
    'duration': 10,
    # 'observation_durations' : [.06, .6],
    'observation_durations' : [.1],
    'history_size' : 5,
    # 'history_size' : 41,
    'print_step': False,
    'print_period': 1,
    'skip_slow_start': 0,
    'log_nml': True, # If false, use linear normalization
    'enable_own_logging': False,
    'enable_nvenc': True,
    'enable_nvdec': True,
}


ENV_CONFIG = {
    'action_keys' : list(sorted([
        'prediction_bandwidth',
        # 'bitrate',
        'pacing_rate',
        # 'resolution',
        # 'fps',
        # 'padding_rate',
        # 'fec_rate_key',
        # 'fec_rate_delta',
    ])),
    'observation_keys' : list([
        # 'frame_encoding_delay',
        # 'frame_egress_delay', 
        # 'frame_recv_delay',
        # 'frame_decoding_delay',
        # 'frame_fps', 
        # 'frame_fps_decoded', 
        # 'frame_qp',
        # 'frame_height', 
        # 'frame_encoded_height', 
        # 'frame_size',
        # 'frame_key_count',
        # 'pkt_egress_rate', 
        # 'pkt_ack_rate',
        # 'action_gap',

        # pandia data
        'frame_decoded_delay',
        'frame_bitrate',
        'bitrate',
        'pkt_trans_delay', 
        'pkt_delay_interval',
        'pkt_loss_rate',
        # Internal variable of the network.
        # Should only be used in the value function.
        # 'bandwidth',
        
        # Bandwidth Estimation Challenge data
        # "receiving_rate",
        # "received_packets_num",
        # "received_bytes",
        # "queuing_delay",
        # "delay",
        # "mini_seen_delay",
        # "delay_ratio",
        # "delay_avg_mini_diff",
        # "pkt_received_interval",
        # "pkt_received_jitter",
        # "pkt_loss_ratio",
        # "pkt_loss_num"
    ]), # 去除sorted
    'action_static_settings' : {
        'bitrate': 0,
        'prediction_bandwidth': 0,
        'pacing_rate': 0,
        'pacing_rate': 0,
        # 'fps': 25,
        'fps': 0,
        'fec_rate_key': 0,
        'fec_rate_delta': 0,
        'padding_rate': 0,
        # 'resolution': 720,
        'resolution': 0,
    },
    'action_invalid_value': {
        'bitrate': 0,
        'prediction_bandwidth': 0,
        'pacing_rate': 0,
        'fps': 0,
        'fec_rate_key': 256,
        'fec_rate_delta': 256,
        'padding_rate': 0,  # TODO: 0 is still a valid value
        'resolution': 0,
    },
    'boundary': BOUNDARY,
    'network_setting': NETWORK_SETTING,
    'action_limit': ACTION_LIMIT,  # It is used by curriculum learning that
                                   # constrains the action space by cliping
    'video_source': VIDEO_SOURCE,
    'gym_setting': GYM_SETTING,
}
