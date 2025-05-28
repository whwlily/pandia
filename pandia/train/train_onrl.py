import json
import gymnasium
from stable_baselines3 import PPO
from pandia.agent.env_config_offline import ENV_CONFIG
from pandia.constants import M, K
from pandia.agent.env_emulator_pure import WebRTCEmulatorEnv_pure
from pandia.model.policies import CustomPolicy
from pandia.model.schedules import linear_schedule
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class PeriodicCheckpoint(BaseCallback):
    def __init__(self, save_freq: int, start_step: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.start_step = start_step
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 开始 checkpoint 的起点条件
        if self.num_timesteps >= self.start_step and self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model_step_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"✅ Saved model at step {self.num_timesteps} to {path}")
        return True


def main():
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 8 * M
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = .06
    config['gym_setting']['duration'] = 100
    
    config['gym_setting']['step_duration'] = .06
    config['gym_setting']['observation_durations'] = [.06, .6]
    config['gym_setting']['history_size'] = 41
    
    config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['skip_slow_start'] = 0
    config['gym_setting']['enable_nvenc'] = False
    config['gym_setting']['enable_nvdec'] = False
    config['gym_setting']['action_cap'] = False

    model_path = ""
    is_gcc = False
    model_name = os.path.basename(model_path).replace('.onnx', '')
    trace_file = "09401.json"
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
        "is_eval": False,
        "eval_list": []
    }
    env: WebRTCEmulatorEnv_pure  = gymnasium.make("WebRTCEmulatorEnv_pure", config=config, net_config=net_config, curriculum_level=None) # type: ignore

    # # 实例化模型
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=3e-4,
    #     clip_range=0.2,
    #     n_steps=1024,
    #     batch_size=64,
    #     verbose=1,
    #     tensorboard_log="./ppo/tensorboard/"
    # )
    # 加载模型（确保模型路径存在）
    model = PPO.load("/data2/kj/Workspace/Pandia/ppo/checkpoints_reward11/model_step_105000.zip", env=env)

    net_config_eval = {
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
        "eval_list": ["08682.json", "05842.json", "01722.json", "01493.json", "02572.json", "00113.json", "00472.json", "05940.json", "08312.json", "07495.json", "05865.json", "01196.json", "01667.json", "01483.json"]
    }
    eval_env: WebRTCEmulatorEnv_pure  = gymnasium.make("WebRTCEmulatorEnv_pure", config=config, net_config=net_config_eval, curriculum_level=None) # type: ignore

    eval_callback = EvalCallback(
        eval_env,                      # 验证环境
        best_model_save_path="./ppo/best_models/",
        log_path="./ppo/eval_logs/",
        eval_freq=5000,
        n_eval_episodes=5, 
        deterministic=True,
        render=False
    )

    checkpoint_callback = PeriodicCheckpoint(
        save_freq=5000,
        start_step=5000,
        save_path="./ppo/checkpoints_reward14/",
        verbose=1
    )

    model.learn(total_timesteps=500000, callback=[eval_callback, checkpoint_callback])


def evaluat():
    config = ENV_CONFIG
    config['network_setting']['bandwidth'] = 8 * M
    config['gym_setting']['print_step'] = True
    config['gym_setting']['print_period'] = .06
    config['gym_setting']['duration'] = 100
    
    config['gym_setting']['step_duration'] = .06
    config['gym_setting']['observation_durations'] = [.06, .6]
    config['gym_setting']['history_size'] = 41
    
    config['gym_setting']['logging_path'] = '/tmp/pandia.log'
    config['gym_setting']['skip_slow_start'] = 0
    config['gym_setting']['enable_nvenc'] = False
    config['gym_setting']['enable_nvdec'] = False
    config['gym_setting']['action_cap'] = False

    model_path = ""
    is_gcc = False
    model_name = os.path.basename(model_path).replace('.onnx', '')
    trace_file = "08690.json" 
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
    env: WebRTCEmulatorEnv_pure  = gymnasium.make("WebRTCEmulatorEnv_pure", config=config, net_config=net_config,curriculum_level=None) # type: ignore

    # 加载保存的最优模型
    model = PPO.load("/data2/kj/Workspace/Pandia/ppo/checkpoints_reward11/model_step_105000.zip", env=env)
    n_episodes = 1
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    print(f"\n✅ Mean reward over {n_episodes} episodes: {mean_reward:.2f}")
    return mean_reward


if __name__ == "__main__":
    main()
    # evaluat()
