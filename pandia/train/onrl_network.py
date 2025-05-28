from typing import Callable, Tuple
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class OnRLPolicyNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(OnRLPolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class CustomMLPExtractor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.policy_net = OnRLPolicyNet(input_dim, action_dim)
        self.value_net = OnRLPolicyNet(input_dim, 1)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)
    

class CustomOnRLPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],                # 禁用默认结构
            activation_fn=nn.Tanh,      # 可不写，备用
            **kwargs
        )

        input_dim = self.features_dim
        action_dim = self.action_dist.proba_distribution.param_shape[0]

        # 替换掉默认 mlp_extractor
        self.mlp_extractor = CustomMLPExtractor(input_dim, action_dim)

        # 重构模型参数
        self._build(late_init=True)
