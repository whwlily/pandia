import json
import os
import socket
import docker
import time
import uuid
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from gymnasium import spaces
import re
import shutil
from pandia import BIN_PATH, RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env import WebRTCEnv
from pandia.agent.env_config_offline import ENV_CONFIG
from pandia.agent.observation_thread import ObservationThread
from pandia.agent.reward import reward
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, generate_diagrams
from pandia.constants import M, K
from datetime import datetime
import random

class WebRTCEmulatorEnv_pure(WebRTCEnv):
    def __init__(self, config=ENV_CONFIG, net_config={}, curriculum_level=0) -> None: 
        super().__init__(config, curriculum_level)
        # Exp settings
        self.uuid = str(uuid.uuid4())[:8]
        self.termination_timeout = 3
        # Logging settings
        self.logging_path = config['gym_setting'].get('logging_path', None)
        self.sb3_logging_path = config['gym_setting'].get('sb3_logging_path', None)
        self.obs_logging_path = config['gym_setting'].get('obs_logging_path', None)
        self.enable_own_logging = config['gym_setting'].get('enable_own_logging', False)
        if self.enable_own_logging:
            self.logging_path = f'{self.logging_path}.{self.uuid}'
            self.sb3_logging_path = f'{self.sb3_logging_path}.{self.uuid}'
        # RL settings
        self.init_timeout = 10
        self.action_space = spaces.Discrete(25)
        # Tracking
        self.bad_reward_count = 0
        self.docker_client = docker.from_env()
        self.control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.obs_socket = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket, logging_path=self.obs_logging_path)
        self.obs_thread.start()
        self.start_container()
        self.monitor_data = []
        # network settings
        self.net_config = net_config
        self.config = config

    def start_container(self):
        cmd = f'docker run -d --rm --name {self.container_name} '\
              f'--hostname {self.container_name} '\
              f'--runtime=nvidia --gpus all '\
              f'--cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all '\
              f'-v /tmp:/tmp '\
              f'-v /data2/kj/Workspace/Pandia/docker_mnt/media:/app/media '\
              f'-v /data2/kj/Workspace/Pandia/docker_mnt/traffic_shell:/app/traffic_shell '\
              f'--env PRINT_STEP=True -e SENDER_LOG=/tmp/sender.log --env BANDWIDTH=1000-3000 '\
              f'{"--env NVENC=1" if self.enable_nvenc else ""} '\
              f'{"--env NVDEC=1" if self.enable_nvdec else ""} '\
              f'--env OBS_SOCKET_PATH={self.obs_socket_path} '\
              f'--env LOGGING_PATH={self.logging_path} '\
              f'--env SB3_LOGGING_PATH={self.sb3_logging_path} '\
              f'--env CTRL_SOCKET_PATH={self.ctrl_socket_path} '\
              f'johnson163/pandia_emulator python -um sb3_client'
        print(cmd)
        print('============================================')
        os.system(cmd)
        self.container = self.docker_client.containers.get(self.container_name) # type: ignore
        ts = time.time()
        while time.time() - ts < 3:
            try:
                self.control_socket.connect(self.ctrl_socket_path)
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if time.time() - ts > 5:
            raise Exception(f'Cannot connect to {self.ctrl_socket_path}')
        self.container_start_ts = time.time()

    def stop_container(self):
        if self.container:
            os.system(f'docker stop {self.container.id} > /dev/null &')

    @property
    def container_name(self):
        return f'sb3_emulator_{self.uuid}'

    @property
    def ctrl_socket_path(self):
        return f'/tmp/{self.uuid}_ctrl.sock'

    @property
    def obs_socket_path(self):
        return f'/tmp/{self.uuid}_obs.sock'

    def log(self, msg):
        print(f'[{self.uuid}, {time.time() - self.start_ts:.02f}] {msg}', flush=True)

    def start_webrtc(self, bw="06251.json", delay=30, loss=0, jitter = 0):
        # self.sample_net_params()
        self.net_sample['bw'] = bw
        self.net_sample['delay'] = delay
        self.net_sample['loss'] = loss
        self.net_sample['jitter'] = jitter
        print(f'Starting WebRTC, {self.net_sample}', flush=True)
        buf = bytearray(1)
        buf[0] = 2
        buf += json.dumps(self.net_sample).encode()
        self.control_socket.send(buf)

    def stop_webrtc(self):
        print('Stopping WebRTC...', flush=True)
        buf = bytearray(1)
        buf[0] = 0  
        self.control_socket.send(buf)
        
    def create_observer(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(self.obs_socket_path)
        print(f'Listening on IPC socket {self.obs_socket_path}')
        return sock

    def reset(self, seed=None, options=None):
        ans = super().reset(seed, options)
        self.stop_webrtc()
        self.step_count = 0
        self.obs_thread.context = self.context
        self.last_print_ts = 0
        self.bad_reward_count = 0
        self.start_ts = time.time()
        sample_trace_file = sample_trace()
        if self.net_config["is_eval"]:
            if len(self.net_config["eval_list"]) > 0:
                self.net_config["bw_file_name"] = random.choice(self.net_config["eval_list"])
            else:
                self.net_config["bw_file_name"] = self.net_config["bw_file_name"]
        else:
            self.net_config["bw_file_name"] = sample_trace_file
        print(f'Reset WebRTC, trace: {self.net_config["bw_file_name"]}')
        return ans

    def close(self):
        self.stop_container()
        if os.path.exists(self.obs_socket_path):
            os.remove(self.obs_socket_path)
        self.obs_thread.stop()        
        # with open('monitor_data_new.json', 'w') as json_file:
        #     json.dump(self.monitor_data, json_file, indent=4)
        
    def step(self, action: np.ndarray):
        action_rtc = Action(self.config['action_keys'])
        # 用离散动作控制行为，例如映射到具体带宽
        bandwidth_levels = [0.1 * M, 0.2 * M, 0.3 * M, 0.4 * M, 0.5 * M, 0.6 * M, 0.7 * M, 0.8 * M, 0.9 * M, 1 * M, 1.1 * M, 1.2 * M, 1.3 * M, 1.4 * M, 1.5 * M, 1.6 * M, 1.7 * M, 1.8 * M, 1.9 * M, 2 * M, 2.1 * M, 2.2 * M, 2.3 * M, 2.4 * M, 2.5 * M]
        assert self.action_space.contains(action), f"Invalid action: {action}"
        action_rtc.prediction_bandwidth = bandwidth_levels[action]
        action = action_rtc.array()
        limit = Action.action_limit(self.action_keys, limit=self.action_limit)
        action = np.clip(action, limit.low, limit.high)
        self.context.reset_step_context()
        act = Action.from_array(action, self.action_keys)
        action_capped = False
        if self.action_cap:
            # Avoid over-sending by limiting the bitrate to the network bandwidth
            if act.bitrate > self.net_sample['bw'] * self.action_cap:
                act.bitrate = self.net_sample['bw'] * self.action_cap
                action_capped = True  
        self.actions.append(act)
        buf = bytearray(Action.shm_size() + 1)
        act.write(buf)
        buf[1:] = buf[:-1]
        buf[0] = 1
        self.control_socket.send(buf)

        if self.step_count == 0:
            self.start_webrtc(bw=self.net_config["bw_file_name"], delay=self.net_config["delay"], loss=0, jitter=self.net_config["jitter"])
        ts = time.time()
        while not self.context.codec_initiated:
            if time.time() - ts > self.init_timeout:
                print(f"WebRTC start timeout. Startover.", flush=True)
                self.reset()
                return self.step(act.array())
            time.sleep(.1)
        ts = time.time()
        if self.step_count == 0:
            self.start_ts = ts
            print(f'WebRTC is running.', flush=True)
        time.sleep(self.step_duration)
        if self.step_count == 0:
            time.sleep(self.skip_slow_start)

        for mb in self.context.monitor_blocks.values():
            mb.update_ts(time.time() - self.obs_thread.ts_offset)

        self.observation.append(self.context.monitor_blocks)
        r, r_detail = reward(self.context, self.net_sample)

        if self.print_step and time.time() - self.last_print_ts > self.print_period:
            self.last_print_ts = time.time()
            self.log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s, '
                    f'R.w.: {r:.02f}, '
                    f'Act.: {act}{"(C)" if action_capped else ""}, '
                    f'BW: {self.obs_thread.cur_capacity / K:.02f}Mbps, '
                    f'Obs.: {self.observation}')
            
        info = {
            'reward': r,
            'action': act.array(),
            'observation': self.observation.array_onrl(),
            'true_capacity': self.obs_thread.cur_capacity,
            'r_detail': r_detail,
        }
        self.step_count += 1
        if r <= -10:
            self.bad_reward_count += 1
        else:
            self.bad_reward_count = 0
        terminated = self.step_count > self.step_limit
        truncated = self.bad_reward_count >= 20
        return self.observation.array_onrl(), r, terminated, truncated, info


gymnasium.register('WebRTCEmulatorEnv_pure', entry_point='pandia.agent.env_emulator_pure:WebRTCEmulatorEnv_pure', 
                   nondeterministic=False)


def sample_trace():
    directory = "/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data"
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # 获取目录中的所有文件
    all_files = os.listdir(directory)
    file_list = [f for f in all_files if os.path.isfile(os.path.join(directory, f))]

    if not file_list:
        raise FileNotFoundError(f"No files found in directory: {directory}")

    # 随机选一个文件
    return random.choice(file_list)

def test_single(trace_file, is_gcc_model):
    model_path = ""
    is_gcc = is_gcc_model
    model_name = os.path.basename(model_path).replace('.onnx', '')
    net_config = {
        "bw_file_name": f"{trace_file}",
        "delay": 20,
        "jitter": 0,
        "loss": 0,
        "limit": 50,
        "is_gcc": is_gcc,
        "model_name": model_name if not is_gcc else "gcc",
        "model_path":  model_path,
        "std_factor": "-0.5, -0.5",
    }

    res_folder = os.path.join(RESULTS_PATH, f'trace_{net_config["bw_file_name"][-10:-5]}', f'{net_config["model_name"]}')
    res_folder = f"{res_folder}_{datetime.now().strftime('%m%d%H%M')}"
    os.makedirs(res_folder, exist_ok=True)
    with open(f'{res_folder}/net_config.json', 'w') as json_file:
        json.dump(net_config, json_file, indent=4)

    video_directory = '/data2/kj/Workspace/Pandia/docker_mnt/media/res_video'
    # 删除整个目录
    if os.path.exists(video_directory):
        shutil.rmtree(video_directory)
    # 重新创建一个空目录
    os.makedirs(video_directory, exist_ok=True)


    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 8 * M
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = .06
    config['gym_setting']['duration'] = 1000
    
    config['gym_setting']['step_duration'] = .06
    config['gym_setting']['observation_durations'] = [.06, .6]
    config['gym_setting']['history_size'] = 41
    
    config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['skip_slow_start'] = 0
    config['gym_setting']['enable_nvenc'] = False
    config['gym_setting']['enable_nvdec'] = False
    config['gym_setting']['action_cap'] = False
    env: WebRTCEmulatorEnv_offline = gymnasium.make("WebRTCEmulatorEnv_offline", config=config, net_config=net_config,curriculum_level=None) # type: ignore
    action = Action(config['action_keys'])
    actions = []
    rewards = []
    bwe_prediction = []
    true_capacity_json = []
    true_capacity = []
    true_capacity_record = []
    observation_log = []
    slope_log = []
    Q_std_log = []
    delay_log = []
    bitrate_log = []
    Pi_std_log = []  
    with open(f"/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data/{net_config['bw_file_name']}", "r") as file:
        true_capacity_json = json.load(file)["true_capacity"]
    try:
        env.reset()
        bitrates = [0 * K] if net_config['is_gcc'] else [100 * K]
        observation = None
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        pre_act = 150 * K
        counter = 5
        for i, bitrate in enumerate(bitrates):
            pre_bw = bitrates[i-1] if i > 0 else 1 * M
            
            action.prediction_bandwidth = bitrate
            # action.prediction_bandwidth = 0
            # action.pacing_rate = 0 if net_config['is_gcc'] else int(bitrate)
            # action.bitrate = bitrate
            obs, reward, terminated, truncated, r_detail = env.step(action.array())
            actions.append(action.prediction_bandwidth / M)
            # actions.append(action.bitrate / M)
            rewards.append(reward)
            delay_log.append(r_detail['mean_delay'])
            bitrate_log.append(r_detail['bitrate'])
            observation = obs
            feed_dict = {
                'obs': observation.reshape(1,1,-1),
                'hidden_states': hidden_state,
                'cell_states': cell_state
            }
            

            if net_config['is_gcc']:
                bitrates.append(0)
           
            # bwe_prediction.append(cur_act / M)

            # Q_std_log.append(q_std_val)
            observation_log.append(observation)
            slope_log.append(0)
            true_capacity_record.append(env.obs_thread.cur_capacity / K)
            # Pi_std_log.append(abs(bw_prediction[0,0,1]))
            # Pi_std_log.append(np.mean(np.squeeze(std)))


            if i == len(true_capacity_json) - 1 or i >= 1800:
                break
    except KeyboardInterrupt:
        pass
    env.close()
    true_capacity = true_capacity_record
    bwe_data = {
        'observations': np.array(observation_log).tolist(),
        'bandwidth_predictions': (np.array(bwe_prediction) * M).tolist(),
        'true_capacity': (np.array(true_capacity) * M).tolist(),
        'slope': np.array(slope_log).tolist(),
        'Q_std_log': np.array(Q_std_log).tolist(),
        'Pi_std_log': np.array(Pi_std_log).tolist(),
        'reward': np.array(rewards).tolist(),
        'delay': np.array(delay_log).tolist(),
        'bitrate': np.array(bitrate_log).tolist()
    }
    bandwidth_predictions = np.array(bwe_prediction)
    true_capacity = np.array(true_capacity)
    error_rate = np.nanmean(np.abs(bandwidth_predictions - true_capacity) / true_capacity)
    overestimation_values = bandwidth_predictions - true_capacity
    overestimation_rate = np.mean((overestimation_values[overestimation_values > 0] / true_capacity[overestimation_values > 0]))
    underestimation_values = true_capacity - bandwidth_predictions
    underestimation_rate = np.mean((underestimation_values[underestimation_values > 0] / true_capacity[underestimation_values > 0]))
    mse = np.mean((bandwidth_predictions - true_capacity) ** 2)
    bwe_data.update({
        'error_rate': error_rate,
        'overestimation_rate': overestimation_rate,
        'underestimation_rate': underestimation_rate,
        'mse': mse
    })
    print(f'Error rate: {error_rate:.4f}, Overestimation rate: {overestimation_rate:.4f}, Underestimation rate: {underestimation_rate:.4f}, MSE: {mse:.4f}')
    print(f"BW_predictions: {np.mean(bandwidth_predictions):.4f}")
    print(f"Reward: {np.mean(rewards):.4f}")
    with open(f'{res_folder}/{net_config["bw_file_name"][-10:]}', 'w') as json_file:
        json.dump(bwe_data, json_file, indent=4)

    os.system(f"cp {config['gym_setting']['logging_path']} {res_folder}")
    log_path = os.path.join(res_folder, 'pandia.log')
    fig_path = res_folder
    generate_diagrams(fig_path, env.context, log_path, true_capacity)


    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(len(actions)), actions, 'r')
    ax1.set_ylabel('Bitrate (Mbps)')
    ax1.set_xlabel('Step')
    ax1.spines['left'].set_color('r')
    ax1.yaxis.label.set_color('r')
    ax1.tick_params(axis='y', colors='r')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(rewards)), rewards, 'b')
    ax2.set_ylabel('Reward')
    ax2.spines['right'].set_color('b')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    plt.savefig(os.path.join(fig_path, f'bitrate_reward.{FIG_EXTENSION}'), dpi=DPI)

    rewards_cdf = np.sort(rewards)
    cdf = np.arange(1, len(rewards_cdf)+1) / len(rewards_cdf)
    # 绘制CDF图
    plt.close()
    plt.plot(rewards_cdf, cdf, marker='.', linestyle='none')
    plt.title('Rewards CDF')
    plt.xlabel('Rewards')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.xlim(-10, 10)  # 设置横轴范围
    plt.savefig(os.path.join(fig_path, f'cdf_reward.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    x = [i * 0.06 for i in range(len(bwe_prediction))]
    plt.plot(x, bwe_prediction, 'r', label='BWE Prediction')
    # plt.plot(x, true_capacity, 'g', label='True Capacity')
    plt.plot(x, true_capacity_record, 'b', label='True Capacity')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Bandwidth (Mbps)')
    plt.savefig(os.path.join(fig_path, f'bwe_prediction.{FIG_EXTENSION}'), dpi=DPI)

    plt.close()
    # cal_vmaf(res_folder)
    


if __name__ == '__main__':

    # for file in os.listdir("/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data"):
    #     if file.endswith(".json"):
    #         need = True
    #         if os.path.isdir(os.path.join(RESULTS_PATH, "trace_" + file.split('.')[0])):
    #             for dir_name in os.listdir(os.path.join(RESULTS_PATH, "trace_" + file.split('.')[0])):
    #                 if "gcc_" in dir_name:
    #                     need = False
    #                     break
    #         print(f"Trace file: {file}")
    #         if need:
    #             test_single(file, False)
    #             test_single(file, True)

    file = "02605.json"
    test_single(file, False)
    # test_single(file, True)

    # cal_vmaf("/data2/kj/Workspace/Pandia/results/trace_02683/actor_checkpoint_880000_03121942")
    # plot_vmaf("/data2/kj/Workspace/Pandia/results/trace_09401/actor_checkpoint_880000_03121632")



    
    
