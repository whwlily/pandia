import torch
import torch.nn as nn
import math
import numpy as np
import time
import torch.nn.functional as F
MAX_ACTION = 20  # Mbps
STATE_DIM = 150
ACTION_DIM = 1

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_critics: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(VectorizedLinear(dims[i], dims[i + 1], num_critics))
            model.append(nn.ReLU())
        model.append(VectorizedLinear(dims[-1], 1, num_critics))
        self.critic = nn.Sequential(*model)

        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
        self.num_critics = num_critics
        self.encoder = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                #  1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                #  1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder.requires_grad_(False)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state = state * self.encoder
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values
    

class GMMPolicy_NewAct(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        num_components: int = 4,  # 混合分量数
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_components = num_components
        self.act_dim = act_dim
        self.max_action = max_action

        # 固定的 encoder0 参数（与你原来一样）
        self.encoder0 = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder0.requires_grad_(False)

        # 公共编码部分
        self.encoder1 = nn.Sequential(
            nn.Linear(150, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, 2)
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --------------------------
        # 均值分支：输出维度为 act_dim * num_components
        self.rb1_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rb2_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.final_mean = nn.Sequential(
            nn.Linear(hidden_dim, act_dim * num_components),
            nn.Tanh()
        )
        
        # 标准差分支：输出维度为 act_dim * num_components
        self.rb1_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rb2_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.final_std = nn.Sequential(
            nn.Linear(hidden_dim, act_dim * num_components),
            nn.Tanh()
        )
        
        # 混合权重分支：输出 num_components 个分量的权重
        self.pi_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_components)
        )
        
    def forward(self, obs: torch.Tensor, h, c):
        # 对输入做处理：squeeze、乘以 encoder0
        obs_ = torch.squeeze(obs, 0)
        obs_ = obs_ * self.encoder0

        # 公共编码部分
        x = self.encoder1(obs_)
        x, _ = self.gru(x)
        x = self.fc_mid(x)
        
        batch_size = x.shape[0]
        
        # 均值分支
        mean_branch = x
        mem1_mean = mean_branch
        mean_branch = self.rb1_mean(mean_branch) + mem1_mean
        mem2_mean = mean_branch
        mean_branch = self.rb2_mean(mean_branch) + mem2_mean
        mean_out = self.final_mean(mean_branch)
        mean_out = mean_out * 6.0  # 调整尺度（例如 Mbps -> bps）
        # 重塑为 (batch_size, num_components, act_dim)
        mean = mean_out.view(batch_size, self.num_components, self.act_dim)
        
        # 标准差分支
        std_branch = x
        mem1_std = std_branch
        std_branch = self.rb1_std(std_branch) + mem1_std
        mem2_std = std_branch
        std_branch = self.rb2_std(std_branch) + mem2_std
        std_out = self.final_std(std_branch)
        std_out = std_out * 5.0
        std_out = torch.exp(std_out.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX))
        # 重塑为 (batch_size, num_components, act_dim)
        std = std_out.view(batch_size, self.num_components, self.act_dim)
        
        # 混合权重：对每个分量计算权重，并用 softmax 归一化
        pi = self.pi_layer(x)  # (batch_size, num_components)
        pi = F.softmax(pi, dim=-1)

        # 返回一个元组：(mean, std, pi)
        # 如果后续需要构造混合分布，可以根据这三个输出构造相应分布对象
        return (mean, std, pi), h, c

def q_network(state_dict_path, num_critics=5):
    state_dim = 150
    action_dim = 1
    q_net = VectorizedQ(state_dim, action_dim, num_critics)
    state_dict = torch.load(state_dict_path)["qf"]
    q_net.load_state_dict(state_dict)
    q_net.eval()
    return q_net

def q_ensemble(q_net, obs, act):
    test_input = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    test_action = torch.tensor(act, dtype=torch.float32).unsqueeze(0)
    q_values = q_net(test_input, test_action)
    std = torch.std(q_values).item()
    mean = torch.mean(q_values).item()
    return mean, std


def gmm_policy(model_path):
    actor = GMMPolicy_NewAct(state_dim=150, act_dim=1, num_components=4, max_action=MAX_ACTION)
    state_dict = torch.load(model_path)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


if __name__ == "__main__":
#     q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-wo_huber-act_80-v14-a60d1b36/all_checkpoint_1000000.pt")
#     test_input = torch.rand(1, 150)
#     test_action = torch.rand(1, 1)
#     tim1 = time.time()
#     q_values = q_net(test_input, test_action)
#     print(time.time()-tim1)
#     print(torch.var(q_values).item())
    # print(q_values, time.time()-tim1)   

    state_dim = 150
    action_dim = 1
    # state_dict_path = "/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-r-delay-jitter_1-gmm_4-argmax-emulated_tested-v14-737c533e/all_checkpoint_860000.pt"
    # # 构建模型
    # model = VectorizedQ(state_dim, action_dim, 10)
    # state_dict = torch.load(state_dict_path)["qf"]
    # model.load_state_dict(state_dict)
    # model.eval()

    # 创建 dummy inputs
    dummy_state = torch.randn(1, state_dim)
    dummy_action = torch.randn(1, action_dim)
    # onnx_path = "/data2/kj/Workspace/Pandia/pandia/agent/q_network.onnx"
    # # 导出 ONNX
    # torch.onnx.export(
    #     model,
    #     (dummy_state, dummy_action),
    #     onnx_path,
    #     input_names=['state', 'action'],
    #     output_names=['q_values'],
    #     opset_version=11,
    #     dynamic_axes={
    #         'state': {0: 'batch_size'},
    #         'action': {0: 'batch_size'},
    #         'q_values': {0: 'batch_size'}
    #     }
    # )

    # print(f"ONNX model saved to: {onnx_path}")
    # onnx_path = "/data2/kj/Workspace/Pandia/pandia/agent/q_network.onnx"
    # state = np.random.randn(1, 150).astype(np.float32)
    # action = np.random.randn(1, 1).astype(np.float32)

    # time1 = time.time()
    # session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # outputs = session.run(None, {"state": state, "action": action})
    # q_values = outputs[0]  # shape: [1, num_critics]
    # print(q_values)
    # mean = np.mean(q_values)
    # std = np.std(q_values)
    # print(f"Mean: {mean}, Std: {std}")
    # print(f"ONNX inference time: {time.time() - time1:.6f} seconds")


    # q_net = q_network("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-r-delay-jitter_1-gmm_4-argmax-emulated_tested-v14-737c533e/all_checkpoint_860000.pt", num_critics=10)
    # # dummy_state = torch.randn(1, state_dim)
    # # dummy_action = torch.randn(1, action_dim)
    # time2 = time.time()
    # mean, std = q_ensemble(q_net, state, action)

    # print(f"Mean: {mean}, Std: {std}")
    # print(f"PyTorch inference time: {time.time() - time2:.6f} seconds")
    # actor = GMMPolicy_NewAct(state_dim=150, act_dim=1, num_components=4, max_action=MAX_ACTION)
    # state_dict = torch.load("/data2/kj/Schaferct/code/checkpoints_iql/riql-new_act-beta_3.0-quantile_0-sigma_0.5-K1800-few_state-act_80-r-delay-jitter_1-gmm_4-argmax-emulated_tested-v14-737c533e/actor_checkpoint_860000.pt")
    # actor.load_state_dict(state_dict)

    # time1 = time.time()
    # (mean, std, pi), h, c = actor(
    #     dummy_state.unsqueeze(0), 
    #     torch.zeros((1, 1)), 
    #     torch.zeros((1, 1))
    # )
    # print(mean, std, pi)
    # print(f"PyTorch inference time: {time.time() - time1:.6f} seconds")

    


