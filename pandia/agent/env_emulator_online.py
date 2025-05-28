import json
import os

import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
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
from pandia.agent.env_emulator_pure import WebRTCEmulatorEnv_pure
import matplotlib.animation as animation

def test_single(trace_file, is_gcc_model):
    model_path = "onrl"
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
        "is_eval": True,
        "eval_list": []
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
    env: WebRTCEmulatorEnv_pure = gymnasium.make("WebRTCEmulatorEnv_pure", config=config, net_config=net_config, curriculum_level=None) # type: ignore
    # 加载保存的最优模型
    model = PPO.load("/data2/kj/Workspace/Pandia/ppo/checkpoints_reward12/model_step_15000.zip", env=env)
    # model = PPO.load("/data2/kj/Workspace/Pandia/ppo/checkpoints_reward13/model_step_45000.zip", env=env)
    # model = PPO.load("/data2/kj/Workspace/Pandia/ppo/checkpoints_reward14/model_step_15000.zip", env=env)

    # action = Action(config['action_keys'])
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

    time_steps = []
    bw_queue = []
    bitrate_queue = []

    with open(f"/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data/{net_config['bw_file_name']}", "r") as file:
        true_capacity_json = json.load(file)["true_capacity"]
    try:
        obs, _ = env.reset()
        bitrates = [0 * K] if net_config['is_gcc'] else [100 * K]
        observation = None
       
        for i, bitrate in enumerate(bitrates):

            action, _ = model.predict(obs, deterministic=True)
            # 用离散动作控制行为，例如映射到具体带宽
            bandwidth_levels = [0.1 * M, 0.2 * M, 0.3 * M, 0.4 * M, 0.5 * M, 0.6 * M, 0.7 * M, 0.8 * M, 0.9 * M, 1 * M, 1.1 * M, 1.2 * M, 1.3 * M, 1.4 * M, 1.5 * M, 1.6 * M, 1.7 * M, 1.8 * M, 1.9 * M, 2 * M, 2.1 * M, 2.2 * M, 2.3 * M, 2.4 * M, 2.5 * M]
            cur_act = bandwidth_levels[action]

            obs, reward, terminated, truncated, info = env.step(action)

            r_detail = info['r_detail']

            actions.append(cur_act / M)
            rewards.append(reward)
            delay_log.append(r_detail['mean_delay'])
            bitrate_log.append(r_detail['bitrate'])
            observation = obs

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


            if net_config['is_gcc']:
                bitrates.append(0)
            else:
                bitrates.append(int(cur_act))

           
            bwe_prediction.append(cur_act / M)

            observation_log.append(observation)
            slope_log.append(0)
            true_capacity_record.append(env.obs_thread.cur_capacity / K)

            time_steps.append(i * 0.06)
            bw_queue.append(env.obs_thread.cur_capacity / K)
            bitrate_queue.append(r_detail['bitrate'])

            if i == len(true_capacity_json) - 1 or i >= 1800:
                break
    except KeyboardInterrupt:
        pass
    env.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    def animate(j):
        ax.clear()
        ax.plot(time_steps[:j], bw_queue[:j], label='True capacity')
        ax.plot(time_steps[:j], bitrate_queue[:j], label='Bitrate')
        ax.set_ylabel("Mbps")
        ax.set_xlabel("Time (s)")
        ax.set_title("Bandwidth vs Bitrate over Time")
        ax.set_ylim(0, max(max(bw_queue), max(bitrate_queue)) * 1.2)
        ax.legend()

    ani = animation.FuncAnimation(fig, animate, frames=len(time_steps), interval=50)
    ani.save(f'{res_folder}/bandwidth_visualization.mp4', writer='ffmpeg', fps=20)

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
    mask = true_capacity > 0
    bandwidth_predictions = bandwidth_predictions[mask]
    true_capacity = true_capacity[mask]
    error_rate = np.nanmean(np.abs(bandwidth_predictions - true_capacity) / true_capacity)

    overestimation_values = bandwidth_predictions - true_capacity
    overestimation_rate = np.nanmean((overestimation_values[overestimation_values > 0] / true_capacity[overestimation_values > 0]))
    underestimation_values = true_capacity - bandwidth_predictions
    underestimation_rate = np.nanmean((underestimation_values[underestimation_values > 0] / true_capacity[underestimation_values > 0]))
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
    # cal_vmaf(res_folder)
    

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




if __name__ == '__main__':
    # for file in os.listdir("/data2/kj/Workspace/Pandia/docker_mnt/traffic_shell/trace_data"):
    #     if file.endswith(".json"):
    #         need = True
    #         if os.path.isdir(os.path.join(RESULTS_PATH, "trace_" + file.split('.')[0])):
    #             for dir_name in os.listdir(os.path.join(RESULTS_PATH, "trace_" + file.split('.')[0])):
    #                 if "onrl" in dir_name:
    #                     need = False
    #                     break
    #         print(f"Trace file: {file}")
    #         if need:
    #             try:
    #                 test_single(file, False)
    #             except:
    #                 print(f"Error processing {file}")
                    # test_single(file, True)

    file = "06512.json"
    test_single(file, False)
    # test_single(file, True)

    # cal_vmaf("/data2/kj/Workspace/Pandia/results/trace_02683/actor_checkpoint_880000_03121942")
    # plot_vmaf("/data2/kj/Workspace/Pandia/results/trace_09401/actor_checkpoint_880000_03121632")



    
    
