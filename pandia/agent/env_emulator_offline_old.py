import json
import os
import socket
import docker
import time
import uuid
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
import onnxruntime as ort
from pandia import BIN_PATH, RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env import WebRTCEnv
from pandia.agent.env_config_offline import ENV_CONFIG
from pandia.agent.observation_thread import ObservationThread
from pandia.agent.reward import reward
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, generate_diagrams
from pandia.constants import M, K


class WebRTCEmulatorEnv_offline(WebRTCEnv):
    def __init__(self, config=ENV_CONFIG, curriculum_level=0) -> None: 
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
        # Tracking
        self.bad_reward_count = 0
        self.docker_client = docker.from_env()
        self.control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.obs_socket = self.create_observer()
        self.obs_thread = ObservationThread(self.obs_socket, logging_path=self.obs_logging_path)
        self.obs_thread.start()
        self.start_container()
        self.monitor_data = []

    def start_container(self):
        cmd = f'docker run -d --rm --name {self.container_name} '\
              f'--hostname {self.container_name} '\
              f'--runtime=nvidia --gpus all '\
              f'--cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all '\
              f'-v /tmp:/tmp '\
              f'-v /data2/wuhw/Workspace/Pandia/docker_mnt/media:/app/media '\
              f'-v /data2/wuhw/Workspace/Pandia/docker_mnt/traffic_shell:/app/traffic_shell '\
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

    def start_webrtc(self):
        self.sample_net_params()
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
        self.obs_thread.context = self.context
        self.last_print_ts = 0
        self.bad_reward_count = 0
        self.start_ts = time.time()
        return ans

    def close(self):
        self.stop_container()
        if os.path.exists(self.obs_socket_path):
            os.remove(self.obs_socket_path)
        self.obs_thread.stop()        
        with open('monitor_data_new.json', 'w') as json_file:
            json.dump(self.monitor_data, json_file, indent=4)
        

    def step(self, action: np.ndarray):
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
            self.start_webrtc()
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
            basic_attributes = {}
            for attr_name in dir(mb):
                # 忽略以__开头的属性，这些通常是Python内部使用的
                if not attr_name.startswith("__"):
                    attr_value = getattr(mb, attr_name)
                    # 检查是否为基础数据类型
                    if isinstance(attr_value, (int, float, str, bool)):
                        basic_attributes[attr_name] = attr_value
            self.monitor_data.append(basic_attributes)

        self.observation.append(self.context.monitor_blocks)
        r = reward(self.context, self.net_sample)

        if self.print_step and time.time() - self.last_print_ts > self.print_period:
            self.last_print_ts = time.time()
            self.log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s, '
                    f'R.w.: {r:.02f}, '
                    f'bw.: {self.net_sample["bw"] / M:.02f} Mbps, '
                    f'Act.: {act}{"(C)" if action_capped else ""}, '
                    f'Obs.: {self.observation}')
            
        self.step_count += 1
        if r <= -10:
            self.bad_reward_count += 1
        else:
            self.bad_reward_count = 0
        terminated = self.bad_reward_count > 1000
        truncated = self.step_count > self.step_limit
        return self.observation.array_bec(), r, terminated, truncated, {'action': act.array()} # self.observation.array()
        # return self.observation.array(), r, terminated, truncated, {'action': act.bitrate} # eval时使用


gymnasium.register('WebRTCEmulatorEnv_offline', entry_point='pandia.agent.env_emulator_offline:WebRTCEmulatorEnv_offline', 
                   nondeterministic=False)


def test_single():
    bw = 8 * M
    test_folder_name = 'offline_step_60ms_delay_30ms_bw_2-4'
    res_folder = os.path.join(RESULTS_PATH, test_folder_name)
    os.makedirs(res_folder, exist_ok=True)
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = bw
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = .06
    config['gym_setting']['duration'] = 1000
    
    config['gym_setting']['step_duration'] = .06
    config['gym_setting']['observation_durations'] = [.06, .6]
    config['gym_setting']['history_size'] = 41
    
    config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['skip_slow_start'] = 0
    config['gym_setting']['enable_nvenc'] = True
    config['gym_setting']['enable_nvdec'] = True
    config['gym_setting']['action_cap'] = False
    env: WebRTCEmulatorEnv_offline = gymnasium.make("WebRTCEmulatorEnv_offline", config=config, curriculum_level=None) # type: ignore
    action = Action(config['action_keys'])
    actions = []
    rewards = []
    bwe_prediction = []
    true_capacity = []
    time0 = time.time()
    try:
        env.reset()
        pd = 10
        # bitrates = [1 * M] * 50 + [2 * M] * 100 + [5 * M] * pd
        # bitrates = [1 * M] * pd + [2 * M] * pd + [3 * M] * pd + [3 * M] * pd+ [4 * M] * pd
        # bitrates = [500 * K]
        bitrates = [0 * K]
        observation = None
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        # model_path = "/data2/wuhw/Workspace/Pandia/bwe_model/checkpoint_580000.onnx"
        model_path = "/data2/wuhw/Workspace/Pandia/bwe_model/iql_checkpoint_140000.onnx"
        orts = ort.InferenceSession(model_path)
        for i, bitrate in enumerate(bitrates):
            pre_bw = bitrates[i-1] if i > 0 else 1 * M
            # min_bw = max(pre_bw * 0.6, 500 * K)
            # max_bw = min(pre_bw * 1.4, 15 * M) if pre_bw != 0 else 15 * M
            min_bw = 500*K
            max_bw = 15 * M
            
            # action.prediction_bandwidth = max(min(bitrate, max_bw), min_bw)
            action.prediction_bandwidth = bitrate
            obs, reward, terminated, truncated, _ = env.step(action.array())
            actions.append(action.prediction_bandwidth / M)
            rewards.append(reward)
            
            # action.pacing_rate= max(min(bitrate, max_bw), min_bw)
            # if i > 49:
            #     action.bitrate = bitrate * 0.5
            # else:
            #     action.bitrate = bitrate
            # obs, reward, terminated, truncated, _ = env.step(action.array())
            # actions.append(action.bitrate / M)
            # rewards.append(reward)
            
            observation = obs
            feed_dict = {
                'obs': observation.reshape(1,1,-1),
                'hidden_states': hidden_state,
                'cell_states': cell_state
            }
            bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
            # if i >= 49:
            predict = observation[5] * np.exp(bw_prediction[0,0,0])
            bitrates.append(predict)
            bwe_prediction.append(predict / M)
            print(f'{i}, 预测值: {predict / M:.02f} Mbps')

            if (time.time() - time0) % 10 <= 5:
                true_capacity.append(2)
            else:
                true_capacity.append(4)

            
            if i == 800:
                break
    except KeyboardInterrupt:
        pass
    env.close()
    os.system(f"cp {config['gym_setting']['logging_path']} {res_folder}")
    log_path = os.path.join(res_folder, 'pandia.log')
    fig_path = res_folder
    generate_diagrams(fig_path, env.context, log_path)

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

    plt.close()
    x = [i * 0.06 for i in range(len(bwe_prediction))]
    plt.plot(x, bwe_prediction, 'r', label='BWE Prediction')
    plt.plot(x, true_capacity, 'b', label='True Capacity')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Bandwidth (Mbps)')
    plt.savefig(os.path.join(fig_path, f'bwe_prediction.{FIG_EXTENSION}'), dpi=DPI)


if __name__ == '__main__':
    test_single()
