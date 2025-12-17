import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, args):
        self.N = args.N
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        # 初始化经验回放池存储为 bfloat16 张量
        self.buffer_obs_n = []
        self.buffer_a_n = []
        self.buffer_r_n = []
        self.buffer_s_next_n = []
        self.buffer_done_n = []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(torch.empty((self.buffer_size, args.obs_dim_n[agent_id]), dtype=torch.bfloat16))
            self.buffer_a_n.append(torch.empty((self.buffer_size, args.action_dim_n[agent_id]), dtype=torch.bfloat16))
            self.buffer_r_n.append(torch.empty((self.buffer_size, 1), dtype=torch.bfloat16))
            self.buffer_s_next_n.append(torch.empty((self.buffer_size, args.obs_dim_n[agent_id]), dtype=torch.bfloat16))
            self.buffer_done_n.append(torch.empty((self.buffer_size, 1), dtype=torch.bfloat16))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        # 存储单步转移（将输入转换为 torch.bfloat16）
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = torch.as_tensor(obs_n[agent_id], dtype=torch.bfloat16)
            self.buffer_a_n[agent_id][self.count] = torch.as_tensor(a_n[agent_id], dtype=torch.bfloat16)
            self.buffer_r_n[agent_id][self.count] = torch.as_tensor(r_n[agent_id], dtype=torch.bfloat16)
            self.buffer_s_next_n[agent_id][self.count] = torch.as_tensor(obs_next_n[agent_id], dtype=torch.bfloat16)
            self.buffer_done_n[agent_id][self.count] = torch.as_tensor(done_n[agent_id], dtype=torch.bfloat16)
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        # 随机采样一个批次的转移（无放回）
        idx = torch.randperm(self.current_size)[:self.batch_size]
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(self.buffer_obs_n[agent_id][idx])
            batch_a_n.append(self.buffer_a_n[agent_id][idx])
            batch_r_n.append(self.buffer_r_n[agent_id][idx])
            batch_obs_next_n.append(self.buffer_s_next_n[agent_id][idx])
            batch_done_n.append(self.buffer_done_n[agent_id][idx])
        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n
