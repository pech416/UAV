import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, args):
        self.N = args.N
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id]), dtype=np.float32))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id]), dtype=np.float32))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1), dtype=np.float32))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id]), dtype=np.float32))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1), dtype=np.float32))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 修改：设置用于采样的设备

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        # 使用 torch.from_numpy 创建张量并转换为 bfloat16 (在指定设备上)
        for agent_id in range(self.N):
            batch_obs_n.append(torch.from_numpy(self.buffer_obs_n[agent_id][index]).to(device=self.device, dtype=torch.float32))         # 修改：obs 转换为 bfloat16 张量
            batch_a_n.append(torch.from_numpy(self.buffer_a_n[agent_id][index]).to(device=self.device, dtype=torch.float32))             # 修改：action 转换为 bfloat16 张量
            batch_r_n.append(torch.from_numpy(self.buffer_r_n[agent_id][index]).to(device=self.device, dtype=torch.float32))             # 修改：reward 转换为 bfloat16 张量
            batch_obs_next_n.append(torch.from_numpy(self.buffer_s_next_n[agent_id][index]).to(device=self.device, dtype=torch.float32)) # 修改：next_obs 转换为 bfloat16 张量
            batch_done_n.append(torch.from_numpy(self.buffer_done_n[agent_id][index]).to(device=self.device, dtype=torch.float32))       # 修改：done 转换为 bfloat16 张量
        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n
