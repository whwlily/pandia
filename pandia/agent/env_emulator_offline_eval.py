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
from collections import deque
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
# from pandia.agent.utils import SlidingWindowQueue
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from pandia.agent.q_network import q_network, q_ensemble
from scipy.stats import norm
from scipy.optimize import minimize
import subprocess

class WebRTCEmulatorEnv_offline(WebRTCEnv):
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
        self.sample_net_params()
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
        # with open('monitor_data_new.json', 'w') as json_file:
        #     json.dump(self.monitor_data, json_file, indent=4)
        

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
            # basic_attributes = {}
            # for attr_name in dir(mb):
            #     # 忽略以__开头的属性，这些通常是Python内部使用的
            #     if not attr_name.startswith("__"):
            #         attr_value = getattr(mb, attr_name)
            #         # 检查是否为基础数据类型
            #         if isinstance(attr_value, (int, float, str, bool)):
            #             basic_attributes[attr_name] = attr_value
            # self.monitor_data.append(basic_attributes)

        self.observation.append(self.context.monitor_blocks)
        r, r_detail = reward(self.context, self.net_sample)

        if self.print_step and time.time() - self.last_print_ts > self.print_period:
            self.last_print_ts = time.time()
            self.log(f'#{self.step_count}@{int((time.time() - self.start_ts))}s, '
                    f'R.w.: {r:.02f}, '
                    # f'bw.: {self.net_sample["bw"] / M:.02f} Mbps, '
                    f'Act.: {act}{"(C)" if action_capped else ""}, '
                    f'Obs.: {self.observation}')
            
        self.step_count += 1
        if r <= -10:
            self.bad_reward_count += 1
        else:
            self.bad_reward_count = 0
        terminated = self.bad_reward_count > 1000
        truncated = self.step_count > self.step_limit
        return self.observation.array_bec(), r, terminated, truncated, r_detail # self.observation.array()
        # return self.observation.array_bec(), r, r_detail, terminated, truncated, {'action': act.array()} # self.observation.array()
        # return self.observation.array(), r, terminated, truncated, {'action': act.bitrate} # eval时使用


gymnasium.register('WebRTCEmulatorEnv_offline', entry_point='pandia.agent.env_emulator_offline:WebRTCEmulatorEnv_offline', 
                   nondeterministic=False)


def test_single(trace_file, is_gcc_model):
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/checkpoint_580000.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/iql_checkpoint_140000.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/iql_checkpoint_890000_wo5.onnx"
    # model_path = "/data2/kj/Schaferct/code/checkpoints_iql/new_act-beta-3.0-v2&v5_1_5-wo_5-new_reward_nonlinear-big-v14-26e77af4/actor_checkpoint_1000000.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/baseline.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/Schaferct.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/few_state_checkpoint_790000.onnx"
    # model_path = "/data2/kj/SRPO/BWE_policy_models/BWE_behavior-new_act-K1800--2-4/policy_ckpt180.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/actor_checkpoint_1000000_v2_v3_v4_80%.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/TEN-TMS_model.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/actor_checkpoint_1000000_riql_act80.onnx"
    # model_path = "/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-v2_v3_v4_80%-v14-4cf3978e/actor_checkpoint_1000000.onnx"
    # model_path = "/data2/kj/Workspace/Pandia/bwe_model/actor_checkpoint_1000000_log_act.onnx"
     # model_path = "/data2/kj/Workspace/Pandia/bwe_model/actor_checkpoint_1000000_act80.onnx"
    # model_path = "/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-v14-a5954cd3/actor_checkpoint_95000.onnx"
    model_path = "/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-gmm-argmax-v14-73f1c99c/actor_checkpoint_880000.onnx"
    orts = ort.InferenceSession(model_path)
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
        "std_factor": "-0.2, 0.1",
    }

    res_folder = os.path.join(RESULTS_PATH, f'trace_{net_config["bw_file_name"][:-5]}', f'{net_config["model_name"]}')
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
    # Q网络
    # q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-slope_reward_act80-v14-28227848/all_checkpoint_1000000.pt")
    # q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-v2_v3_v4_80%-v14-4eb63d30/all_checkpoint_1000000.pt")
    # q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-v2_v3_v4_80%-v14-4cf3978e/all_checkpoint_1000000.pt")
    # q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-log_action-v14-df63f4c1/all_checkpoint_1000000.pt")
    q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-gmm-argmax-v14-73f1c99c/all_checkpoint_880000.pt")

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
            # cur_slope = max(min(compute_slope(np.array(observation[85:90])), 100), -100)
            # cur_slope0 = max(min(compute_slope(np.array(observation[80:85])), 100), -100)
            # if i == 0:
            #     D, gamma = trend_detector(np.array(observation[85:90]), 15)
            # else:
            #     D, gamma = trend_detector(np.array(observation[85:90]), gamma)
            # print(f"D: {D:.02f}, gamma: {gamma:.02f}")
            # Q_std_log.append(D)
            # Pi_std_log.append(gamma)
            # if cur_slope < 3 or counter > 0:
            #     bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
            #     predict = observation[5] * np.exp(bw_prediction[0,0,0])
            #     counter -= 1
            # else:
            #     predict = pre_act * 0.8
            #     counter = 5

            # bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
            # q_mean_val, q_std_val = q_ensemble(q_net, observation, bw_prediction[0,0,0])
            # predict = observation[5] * np.exp(bw_prediction[0,0,0] - 0.2 * abs(bw_prediction[0,0,1])) if bw_prediction[0,0,0] > 0 else observation[5] * np.exp(bw_prediction[0,0,0] + 0.1 * abs(bw_prediction[0,0,1]))
            # print(f"Pi_std: {abs(bw_prediction[0,0,1]):.02f}")

            # predict = np.exp(bw_prediction[0,0,0] - 0.6 * abs(bw_prediction[0,0,1]))
            # predict = observation[5] * np.exp(bw_prediction[0,0,0] - 0.7 * abs(bw_prediction[0,0,1]))
            # predict = observation[5] * np.exp(bw_prediction[0,0,0]) if observation[0] != 0 else pre_act * 0.8
            # predict = observation[5] * np.exp(bw_prediction[0,0,0] * 3.0 + 1.0)
            # predict = bw_prediction[0,0,0]
            # if i > 10 and observation[0] == 0:
            #     predict = bitrates[i-1]

            # cur_act = min(max(predict, 100 * K), 8 * M) if observation[0] != 0 and observation[35] < 150 else 0
            # cur_act = min(max(predict, 150 * K), 8 * M) if cur_slope < 4 and np.sqrt(q_var_val) < 100 else 0
            # alpha = 0.08 if i < 100 else 0.04

            mean, std, pi, _, _ = orts.run(None, feed_dict)
            # 对于每个样本，找到权重最大的分支索引
            max_component_indices = np.argmax(pi, axis=1)
            # 利用高级索引选取对应分支的均值
            batch_indices = np.arange(mean.shape[0])
            selected_actions = mean[batch_indices, max_component_indices, :][0,0]
            selected_std = std[batch_indices, max_component_indices, :][0,0]

            mean = np.squeeze(mean)
            std = np.squeeze(std)
            pi = np.squeeze(pi)
            # act = find_gmm_mode(pi, mean, std)
            act, unc = find_mode_and_uncertainty(pi, mean, std, tol=1e-4, epsilon=1e-4)
            # print(f"pi: {pi}, mean: {mean}, std: {std}, act: {act}, selected_actions: {selected_actions}")
            if act:
                q_mean_val, q_std_val = q_ensemble(q_net, observation, act)
                predict = observation[5] * np.exp(act - 0.2 * unc)
            else:
                q_mean_val, q_std_val = q_ensemble(q_net, observation, selected_actions)
                predict = observation[5] * np.exp(selected_actions - 0.1 * selected_std)
            
            # predict = observation[5] * np.exp(selected_actions)

            alpha = 0.05
            if q_std_val < q_mean_val * alpha:
                cur_act = min(max(predict, 150 * K), 5 * M)
            else:
                cur_act = 0
            # cur_act = min(max(predict, 150 * K), 4 * M)
            # if i < 100 and q_std_val < q_mean_val * alpha * 2:
            #     cur_act = min(max(predict, 150 * K), 3 * M)
            # elif i >= 100 and q_std_val < q_mean_val * alpha:
            # # elif i >= 100 and q_std_val < q_mean_val * alpha:
            #     cur_act = min(max(predict, 150 * K), 3 * M)
            # else:
            #     cur_act = 0

            # cur_act = min(max(predict, 150 * K), 8 * M)

            if net_config['is_gcc']:
                bitrates.append(0)
            else:
                bitrates.append(int(cur_act))

            pre_act = cur_act
           
            bwe_prediction.append(cur_act / M)
            print(f'{i}, 预测值: {predict / M:.02f} Mbps')
            Q_std_log.append(q_std_val)
            observation_log.append(observation)
            slope_log.append(0)
            true_capacity_record.append(env.obs_thread.cur_capacity / K)
            # Pi_std_log.append(abs(bw_prediction[0,0,1]))
            Pi_std_log.append(np.mean(np.squeeze(std)))


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
    with open(f'{res_folder}/{net_config["bw_file_name"]}', 'w') as json_file:
        json.dump(bwe_data, json_file, indent=4)

    os.system(f"cp {config['gym_setting']['logging_path']} {res_folder}")
    log_path = os.path.join(res_folder, 'pandia.log')
    fig_path = res_folder
    generate_diagrams(fig_path, env.context, log_path, true_capacity)

    plt.close()
    plot_2d_res(delay_log, bitrate_log, fig_path)

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

    # 绘制平滑处理后的 reward 图
    plt.close()
    smoothed_rewards = uniform_filter1d(rewards, size=10)
    plt.plot(np.arange(len(smoothed_rewards)), smoothed_rewards, 'b')
    plt.ylabel('Smoothed Reward')
    plt.xlabel('Step')
    plt.savefig(os.path.join(fig_path, f'smoothed_reward.{FIG_EXTENSION}'), dpi=DPI)

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
    cal_vmaf(res_folder)
    

def cal_vmaf(res_folder):
    remove_pattern_lines()
    concatenate_lines()

    log_file_path = '/data2/kj/Workspace/Pandia/docker_mnt/media/sb3.log'
    seq_fid_map = {}
    qp_map = {} 
    # pattern = re.compile(r"seq: (\d+), first in frame: 1, last in frame: (\d+), fid: (\d+)")
    # pattern = re.compile(r".*seq: (\d+), first in frame: 1, last in frame: (\d+), fid: (\d+).*")
    # pattern = re.compile(r"seq:\s*(\d+),\s*first in frame:\s*(\d+),\s*last in frame:\s*(\d+),\s*fid:\s*(\d+),")
    pattern = re.compile(r"seq:\s*(\d+),\s*first\s*in\s*frame:\s*1,\s*last\s*in\s*frame:\s*(\d+),\s*fid:\s*(\d+),")
    pattern2 = re.compile(r"Finish\s*encoding,.*?frame\s*id:\s*(\d+),\s*frame\s*type:\s*\d+,\s*frame\s*size:\s*\d+,\s*is\s*key:\s*\d+,\s*qp:\s*(\d+)")
    i = 0
    j = 0
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            match2 = pattern2.search(line)
            if match:
                seq = int(match.group(1))
                fid = int(match.group(3))
                seq_fid_map[seq] = fid
                i += 1
            if match2:
                fid = str(match2.group(1))
                qp = int(match2.group(2))
                qp_map[fid] = qp
                j += 1
    for seq, fid in seq_fid_map.items():
        if str(fid) not in qp_map:
            print(f"seq: {seq}, fid: {fid} not found in qp_map")
    print(f"Total: {i} frames, {j} qp frames")
    directory = '/data2/kj/Workspace/Pandia/docker_mnt/media/res_video'
    print(f"Total: {len(os.listdir(directory))} frames")
    for filename in os.listdir(directory):
        match = re.match(r"received_(\d+)\.yuv", filename)
        if match:
            seq = int(match.group(1))
            if seq in seq_fid_map:
                new_filename = f"received_{seq_fid_map[seq]}.yuv"
                old_filepath = os.path.join(directory, filename)
                new_filepath = os.path.join(directory, new_filename)
                os.rename(old_filepath, new_filepath)

    os.system("chmod +x /data2/kj/Workspace/Pandia/docker_mnt/media/handle_resolution.sh")
    call_handle_resolution_script(qp_map, os.path.join(res_folder, "handle_resolution.log"))

    os.system("chmod +x /data2/kj/Workspace/Pandia/docker_mnt/media/cal_vmaf.sh")
    vmaf_log = os.path.join(res_folder, "cal_vmaf.log")
    os.system(f"bash /data2/kj/Workspace/Pandia/docker_mnt/media/cal_vmaf.sh > {vmaf_log} 2>&1")
    
    os.system(f"cp /data2/kj/Workspace/Pandia/docker_mnt/media/vmaf_scores.txt {res_folder}")

    plot_vmaf(res_folder)

def call_handle_resolution_script(qp_map, log_file_path):
    # 将字典转换为 JSON 字符串
    json_params = json.dumps(qp_map)
    
    # 调用 shell 脚本并传递 JSON 参数
    with open(log_file_path, "w") as log_file:
        subprocess.run(
            ["bash", "/data2/kj/Workspace/Pandia/docker_mnt/media/handle_resolution.sh", json_params],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True
        )

def plot_vmaf(res_folder):
    # Step 1: Read the txt file and extract the VMAF scores
    vmaf_scores = []
    frame_numbers = []

    # Read the file and extract relevant data
    with open(os.path.join(res_folder, "vmaf_scores.txt"), 'r') as file:
        for line in file:
            if line.startswith('Frame'):
                parts = line.split()
                try:
                    score, frame_number = float(parts[-1]), int(parts[1])
                    vmaf_scores.append(score)  # Extract VMAF score
                    frame_numbers.append(frame_number)  # Extract frame number
                except ValueError:
                    print(f"Error parsing line: {line}")  # If there is an error parsing the line, just
                    pass

   # Convert lists to numpy arrays for better manipulation
    frame_numbers = np.array(frame_numbers)
    vmaf_scores = np.array(vmaf_scores)

    # Step 2: Sort the data by frame number
    sorted_indices = np.argsort(frame_numbers)  # Get the indices that would sort frame_numbers
    frame_numbers_sorted = frame_numbers[sorted_indices]  # Sort frame numbers
    vmaf_scores_sorted = vmaf_scores[sorted_indices]  # Sort VMAF scores according to sorted frame numbers

    # Step 3: Calculate the mean of the VMAF scores
    mean_vmaf = np.mean(vmaf_scores_sorted)
    with open(os.path.join(res_folder, "vmaf_scores.txt"), 'a') as file:
        file.write(f"\nMean VMAF: {mean_vmaf:.2f}")
    # Step 4: Create a regular plot of VMAF scores
    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers_sorted, vmaf_scores_sorted, marker='o', linestyle='-', color='b', label='VMAF Scores')

    # Plot the mean as a horizontal line
    plt.axhline(mean_vmaf, color='r', linestyle='--', label=f'Mean VMAF: {mean_vmaf:.2f}')

    # Add title, labels, and grid
    plt.title('VMAF Scores vs Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('VMAF Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(res_folder, f'vmaf_scores.{FIG_EXTENSION}'), dpi=DPI)

    # Step 4: Create a CDF (Cumulative Distribution Function) plot of VMAF scores
    # Sort the VMAF scores to compute CDF
    sorted_vmaf_scores = np.sort(vmaf_scores)
    cdf = np.arange(1, len(sorted_vmaf_scores) + 1) / len(sorted_vmaf_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_vmaf_scores, cdf, marker='.', linestyle='-', color='r')
    plt.title('CDF of VMAF Scores')
    plt.xlabel('VMAF Score')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.savefig(os.path.join(res_folder, f'vmaf_cdf.{FIG_EXTENSION}'), dpi=DPI)


def remove_pattern_lines():
    log_file_path = '/data2/kj/Workspace/Pandia/docker_mnt/media/sb3.log'
    temp_file_path = log_file_path + ".tmp"
    pattern = re.compile(r"^(.*?)\(.*\):")

    with open(log_file_path, 'r') as file, open(temp_file_path, 'w') as temp_file:
        for line in file:
            match = pattern.match(line)
            if match:
                if match.group(1).strip() == "":
                    # 如果行以 \(.*\): 为起始，删除该整行
                    continue
                else:
                    # 保留该行 \(.*\): 前面的部分
                    temp_file.write(match.group(1) + '\n')
            else:
                temp_file.write(line)

    # 替换原始文件
    os.replace(temp_file_path, log_file_path)

def concatenate_lines():
    log_file_path = '/data2/kj/Workspace/Pandia/docker_mnt/media/sb3.log'
    temp_file_path = log_file_path + ".tmp"
    buffer = ""
    with open(log_file_path, 'r') as file, open(temp_file_path, 'w') as temp_file:
        for line in file:
            if line.startswith('['):
                if buffer:
                    temp_file.write(buffer + '\n')
                buffer = line.strip()
            else:
                buffer += line.strip()
        
        if buffer:
            temp_file.write(buffer + '\n')

    # 替换原始文件
    os.replace(temp_file_path, log_file_path)
    
def compute_slope(sample):
    # 时间序列：[0, 1, 2, 3, 4]
    x = np.arange(5, dtype=np.float32)
    
    # 使用最小二乘法计算斜率
    # y = sample (样本值)
    # 斜率计算公式：slope = (n * Σ(xy) - Σx * Σy) / (n * Σ(x^2) - (Σx)^2)
    n = sample.shape[0]
    xy_sum = np.sum(x * sample)
    x_sum = np.sum(x)
    y_sum = np.sum(sample)
    x_square_sum = np.sum(x**2)
    
    slope = - (n * xy_sum - x_sum * y_sum) / (n * x_square_sum - x_sum**2)
    
    return slope

def plot_2d_res(delay, bitrate, res_path):
    # 示例数据
    x = np.array(delay) * 1000
    y = np.array(bitrate)
    mask = x <= 1000
    x = x[mask]
    y = y[mask]

    # 计算均值和标准差
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)

    # 创建二维坐标图
    plt.scatter(x, y, color='blue', label='Data points')

    # 绘制均值点
    plt.scatter(x_mean, y_mean, color='red', s=100, zorder=5, label=f'Mean: ({x_mean:.2f}, {y_mean:.2f})')

    # 绘制标准差范围线
    plt.plot([x_mean - x_std, x_mean + x_std], [y_mean, y_mean], color='yellow', linestyle='-', label=f'delay std: ±{x_std:.2f}')
    plt.plot([x_mean, x_mean], [y_mean - y_std, y_mean + y_std], color='orange', linestyle='-', label=f'bitrate std: ±{y_std:.2f}')

    # 设置标签
    plt.xlabel('delay/ms')
    plt.ylabel('bitrate/mbps')
    plt.title('Delay and Bitrate')
    # 反转x轴
    plt.gca().invert_xaxis()
    # 显示图例
    plt.legend()
    plt.savefig(os.path.join(res_path, f'res_2d.{FIG_EXTENSION}'), dpi=DPI)

def trend_detector(d_arr, last_gamma = 20):
    k = 0.05
    e_arr = np.array([np.exp(-i) for i in range(5)])
    D = np.sum(d_arr * e_arr)
    gamma = last_gamma - k * (abs(D) - last_gamma)
    return D, gamma



# def negative_gmm_density(a, pi, mu, sigma):
#     """
#     目标函数：负的混合高斯密度
#     """
#     density = np.sum(pi * norm.pdf(a, loc=mu, scale=sigma))
#     return -density
    

# def find_gmm_mode(pi, mu, sigma, initial_guess=None):
#     """
#     利用数值优化方法求解一维混合高斯分布的众数（mode）。
    
#     参数：
#         pi: 混合分量权重数组，形状 (K,)
#         mu: 各分量的均值数组，形状 (K,)
#         sigma: 各分量的标准差数组，形状 (K,)
#         initial_guess: 初始猜测值（可选），默认为各分量均值的加权平均
        
#     返回：
#         mode: 求得的混合分布众数（标量）
#     """
#     if initial_guess is None:
#         initial_guess = np.sum(pi * mu)
#     res = minimize(negative_gmm_density, x0=initial_guess, args=(pi, mu, sigma), method='L-BFGS-B', tol=1e-4)
#     if res.success:
#         return res.x[0]
#     else:
#         return False


def gmm_pdf(x, pi, mu, sigma):
    """
    计算一维混合高斯分布在 x 处的密度
    参数:
        x: 标量
        pi: 混合分量权重数组，形状 (K,)
        mu: 各分量均值数组，形状 (K,)
        sigma: 各分量标准差数组，形状 (K,)
    返回:
        混合密度：sum_{i=1}^K pi_i * N(x; mu_i, sigma_i)
    """
    return np.sum(pi * norm.pdf(x, loc=mu, scale=sigma))

def negative_gmm_density(x, pi, mu, sigma):
    """
    目标函数：负的混合高斯密度，用于最小化
    """
    density = gmm_pdf(x, pi, mu, sigma)
    return -density

def find_gmm_mode(pi, mu, sigma, initial_guess=None, tol=1e-4):
    """
    利用 L-BFGS-B 数值优化方法求解一维混合高斯分布的众数（mode）。
    
    参数:
        pi: 混合分量权重数组，形状 (K,)
        mu: 各分量均值数组，形状 (K,)
        sigma: 各分量标准差数组，形状 (K,)
        initial_guess: 初始猜测值（可选），默认为加权均值 np.sum(pi*mu)
        tol: 优化收敛精度（如 1e-4）
    返回:
        众数（标量）
    """
    if initial_guess is None:
        initial_guess = np.sum(pi * mu)
    res = minimize(negative_gmm_density, x0=initial_guess, args=(pi, mu, sigma),
                   method='L-BFGS-B', tol=tol)
    if res.success:
        return res.x[0]
    else:
        return False

def estimate_uncertainty(mode, pi, mu, sigma, epsilon=1e-4):
    """
    利用二阶差分方法估计混合高斯分布在众数处的局部不确定性（近似标准差）。
    
    参数:
        mode: 众数（标量）
        pi, mu, sigma: 混合分量参数，形状均为 (K,)
        epsilon: 二阶差分步长（如 1e-4）
    返回:
        局部不确定性 sigma_eff = 1/sqrt(lambda)
        其中 lambda 为 log 密度的负二阶导数
    """
    f0 = np.log(gmm_pdf(mode, pi, mu, sigma) + 1e-12)
    f_plus = np.log(gmm_pdf(mode + epsilon, pi, mu, sigma) + 1e-12)
    f_minus = np.log(gmm_pdf(mode - epsilon, pi, mu, sigma) + 1e-12)
    lambda_est = - (f_plus - 2 * f0 + f_minus) / (epsilon ** 2)
    if lambda_est <= 0:
        return np.inf
    sigma_eff = 1 / np.sqrt(lambda_est)
    return sigma_eff

def find_mode_and_uncertainty(pi, mu, sigma, initial_guess=None, tol=1e-4, epsilon=1e-4):
    """
    综合求解混合高斯分布的众数以及在该点的局部不确定性（近似标准差），
    避免重复计算密度函数。
    
    参数:
        pi, mu, sigma: 混合分量参数（数组，形状 (K,)）
        initial_guess: 初始猜测值（可选）
        tol: 优化收敛精度
        epsilon: 数值二阶差分步长
    返回:
        mode: 众数（标量）
        sigma_eff: 估计的不确定性
    """
    mode = find_gmm_mode(pi, mu, sigma, initial_guess=initial_guess, tol=tol)
    uncertainty = estimate_uncertainty(mode, pi, mu, sigma, epsilon=epsilon)
    return mode, uncertainty

if __name__ == '__main__':

    # for file in os.listdir("/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data"):
        # if file.endswith(".json"):
        #     print(f"Trace file: {file}")
        #     test_single(file, False)
        #     test_single(file, True)

    # file = "09401.json"
    # test_single(file, False)
    # # test_single(file, True)

    # cal_vmaf("/data2/kj/Workspace/Pandia/results/trace_02683/actor_checkpoint_880000_03121942")
    # plot_vmaf("/data2/kj/Workspace/Pandia/results/trace_09401/actor_checkpoint_880000_03121632")

    trace_file = "/data2/kj/Schaferct/ALLdatasets/emulated_dataset_policy"
    for policy in os.listdir("/data2/kj/Schaferct/code/eval_list"):
        for bw_range in os.listdir(f"/data2/kj/Schaferct/code/eval_list/{policy}"):
            for trace_name in os.listdir(f"/data2/kj/Schaferct/code/eval_list/{policy}/{bw_range}"):
                trace = trace_file + f"/{policy}/{trace_name.split('.')[0]}.json"
                print(trace)



    
    
