import argparse
import copy
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from pandia import RESOURCE_PATH, RESULTS_PATH
from pandia.agent.action import Action
from pandia.agent.env_emulator import WebRTCEmulatorEnv
from pandia.agent.curriculum_level import CURRICULUM_LEVELS
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.utils import deep_update
from pandia.analysis.stream_illustrator import generate_diagrams, setup_diagrams_path
from pandia.constants import M
from pandia.train.train_sb3_simple_simulator import CustomPolicy


def run(model_path, config):
    config = copy.deepcopy(config)
    if model_path is None:
        config['action_keys'] = ['fake']
    env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    if model_path:
        model = PPO.load(model_path, env, custom_objects={'policy_class': CustomPolicy})
        print(f'Loading model from {model_path}')
    else:
        model = None
    obs, _ = env.reset()
    print(f'Eval with bw: {env.net_sample["bw"] / M:.02f} Mbps')
    rewards = []
    bitrates = []
    delays = []
    for i in range(1000):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_obj = env.observation
        delays.append(obs_obj.get_data(obs_obj.data[0][0], 'frame_decoded_delay', True))
        if model:
            act_obj = Action.from_array(action, env.action_keys)
            # bitrates.append(act_obj.bitrate)
            bitrates.append(info["action"])
            # if act_obj.bitrate / M < 5:
            #     print(f"error！！！！,act_obj.bitrate:{act_obj.bitrate / M}")
        else:
            bitrates.append(float(obs_obj.get_data(obs_obj.data[0][0], 'bitrate', True)))
        rewards.append(reward)
        if terminated or truncated:
            break
    print(f'bw: {env.net_sample["bw"] / M:.02f}Mbps, bitrate: {np.mean(bitrates) / M:.02f}Mbps, '
            f'delay: {np.mean(delays) * 1000:.02f}ms, reward: {np.mean(rewards):.02f}')
    env.close()
    return rewards, bitrates


def main(model_id=None, bw=7 * M):
    if model_id is None:
        log_dir = os.path.expanduser(f'~/sb3_logs/ppo')
        models = [int(d[18:]) for d in os.listdir(log_dir) if d.startswith('WebRTCEmulatorEnv_')]
        model_id = max(models)
    config = ENV_CONFIG
    deep_update(config, CURRICULUM_LEVELS[0])
    config['network_setting']['bandwidth'] = bw
    config['network_setting']['delay'] = .01
    config['gym_setting']['print_step'] = True
    config['gym_setting']['step_duration'] = .02
    config['gym_setting']['action_cap'] = False
    config['gym_setting']['print_period'] = 0
    config['gym_setting']['duration'] = 10
    config['gym_setting']['skip_slow_start'] = 1
    # env = WebRTCEmulatorEnv(config=config, curriculum_level=None) # type: ignore
    path = os.path.expanduser(f"~/sb3_logs/ppo/WebRTCEmulatorEnv_{model_id}/best_model")
    r1, a1 = run(None, config)
    r2, a2 = run(path, config) 
    output_dir = os.path.join(RESOURCE_PATH, 'eval_sb3_emulator_vs_gcc')
    os.makedirs(output_dir, exist_ok=True)
    dump_path = os.path.join(output_dir, 'rewards.json')
    with open(dump_path, 'w') as f:
        json.dump([r1, r2, a1, a2], f)


def analysis():
    resource_dir = os.path.join(RESOURCE_PATH, 'eval_sb3_emulator_vs_gcc')
    os.makedirs(resource_dir, exist_ok=True)
    dump_path = os.path.join(resource_dir, 'rewards.json')
    with open(dump_path) as f:
        r1, r2, a1, a2 = json.load(f)
    output_dir = os.path.join(RESULTS_PATH, 'eval_sb3_emulator_vs_gcc')
    setup_diagrams_path(output_dir)

    plt.close()
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(np.arange(len(r1)), r1, label='GCC')
    plt.plot(np.arange(len(r2)), r2, label='DRL')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rewards.pdf'))

    plt.close()
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    plt.plot(np.arange(len(a1)), np.array(a1) / M, label='GCC')
    plt.plot(np.arange(len(a2)), np.array(a2) / M, label='DRL')
    plt.xlabel('Step')
    plt.ylabel('Bitrate (Mbps)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bitrate.pdf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_id', type=int, default=None, help='model id')
    parser.add_argument('-b', '--bandwidth', type=float, default=3, help='bandwidth in mbps')
    parser.add_argument('-a', '--analyze', action='store_true', help='analyze')
    args = parser.parse_args()
    if args.analyze:
        analysis()
    else:
        main(model_id=args.model_id, bw=args.bandwidth * M)
