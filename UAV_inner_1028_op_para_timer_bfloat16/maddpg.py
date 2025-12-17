import copy
import numpy as np
import torch
import torch.nn.functional as F

# 兼容两种命名：Critic_MADDPG / Critic_MATD3，并导入 orthogonal_init
try:
    from networks import Actor, Critic_MADDPG as CriticNet, orthogonal_init
except Exception:  # noqa
    from networks import Actor, Critic_MATD3 as CriticNet, orthogonal_init

class MADDPG(object):
    def __init__(self, args, agent_id: int):
        # ---------- 基本超参 ----------
        self.args = args
        self.N = args.N
        self.agent_id = agent_id

        self.max_action = float(args.max_action)
        self.action_dim = int(args.action_dim_n[agent_id])
        self.obs_dim = int(args.obs_dim_n[agent_id])

        self.lr_a = float(args.lr_a)
        self.lr_c = float(args.lr_c)
        self.gamma = float(args.gamma)
        self.tau = float(args.tau)

        # 可选开关（若 args 没有相应字段，给默认值）
        self.use_grad_clip = bool(getattr(args, "use_grad_clip", True))
        self.use_orth_init = bool(getattr(args, "use_orthogonal_init", True))

        # ---------- 网络 ----------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 修改：选择运行设备
        self.device = device  # 修改：保存设备用于张量转换
        self.actor = Actor(args, agent_id).to(device=device, dtype=torch.bfloat16)          # 修改：Actor 模型使用 bfloat16 精度
        self.actor_target = copy.deepcopy(self.actor).to(device=device, dtype=torch.bfloat16)   # 修改：Actor_target 模型使用 bfloat16
        self.critic = CriticNet(args).to(device=device, dtype=torch.bfloat16)            # 修改：Critic 模型使用 bfloat16 精度
        self.critic_target = copy.deepcopy(self.critic).to(device=device, dtype=torch.bfloat16)  # 修改：Critic_target 模型使用 bfloat16

        # 将 gamma 和 tau 转换为 bfloat16 张量
        self.gamma = torch.tensor(self.gamma, dtype=torch.bfloat16, device=device)  # 修改：gamma 用 bfloat16 张量表示
        self.tau = torch.tensor(self.tau, dtype=torch.bfloat16, device=device)      # 修改：tau 用 bfloat16 张量表示

        # ---------- 权重初始化（可选） ----------
        if self.use_orth_init:
            for m in self.actor.modules():
                if hasattr(m, "weight") and hasattr(m, "bias"):
                    try:
                        orthogonal_init(m)
                    except Exception:
                        pass
            for m in self.critic.modules():
                if hasattr(m, "weight") and hasattr(m, "bias"):
                    try:
                        orthogonal_init(m)
                    except Exception:
                        pass
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        # ---------- 优化器 ----------
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        # 训练/评估模式
        self.actor.train()
        self.critic.train()
        self.actor_target.eval()
        self.critic_target.eval()

    # --------------------- 动作选择（含高斯噪声） ---------------------
    def choose_action(self, obs, noise_std: float = 0.1):
        """
        obs: np.ndarray shape=(obs_dim,)
        return: np.ndarray shape=(action_dim,), clipped to [-max_action, max_action]
        """
        self.actor.eval()
        with torch.no_grad():
            # 将观测转换为 bfloat16 张量并送入设备
            obs_tensor = torch.as_tensor(obs, dtype=torch.bfloat16, device=self.device).unsqueeze(0)  # 修改：使用 bfloat16 dtype 的输入张量
            a_tensor = self.actor(obs_tensor)   # bfloat16 forward
            # 将输出转换回 float32 CPU numpy 数组
            a = a_tensor.float().cpu().numpy().reshape(-1)  # 修改：输出转换为 float32 用于环境
        self.actor.train()

        if noise_std is not None and noise_std > 0:
            noise = np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)
            a = a + noise

        a = np.clip(a, -self.max_action, self.max_action).astype(np.float32)
        return a

    # --------------------- （废弃）一体化入口 ---------------------
    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Deprecated: use update_critic / update_actor / update_targets inside the env loop."
        )

    # --------------------- Critic 更新 ---------------------
    def update_critic(self, replay_buffer=None, agent_n=None, cached_actions=None, batch_data=None):
        """
        batch_data: (batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n)
        - *_n 是长度为 N 的 list，元素张量形状均为 (batch, dim)
        """
        assert agent_n is not None, "agent_n (list of agents) is required."

        if batch_data is None:
            assert replay_buffer is not None, "Either batch_data or replay_buffer must be provided."
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        else:
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = batch_data

        # 计算 s', a' 以及 target 动作（可缓存传入以避免重复前向）
        if cached_actions is None:
            with torch.no_grad():
                cached_actions = [ag.actor_target(batch_obs_next_n[i]) for i, ag in enumerate(agent_n)]

        s_next_cat = torch.cat(batch_obs_next_n, dim=1)    # (B, sum_obs)
        a_next_cat = torch.cat(cached_actions, dim=1)      # (B, sum_act)
        s_a_next = torch.cat([s_next_cat, a_next_cat], dim=1)

        # === 修复处：确保形状匹配 (B,1) ===
        r_i = batch_r_n[self.agent_id]                     # (B,1)
        done_i = batch_done_n[self.agent_id].to(r_i.dtype)  # (B,1) -> 转换为与 r_i 相同 dtype (bfloat16)  # 修改：done 张量转换为 bf16

        with torch.no_grad():
            q_next = self.critic_target.forward_sa(s_a_next)          # (B,1) bfloat16
            one_minus_done = torch.ones_like(done_i) - done_i         # 修改：计算 (1 - done_i) 在 bfloat16
            target_Q = r_i + one_minus_done * self.gamma * q_next     # (B,1) 目标 Q 值 (bf16 计算)

        # 当前 Q(s,a)
        s_cur_cat = torch.cat(batch_obs_n, dim=1)
        a_cur_cat = torch.cat(batch_a_n, dim=1)
        s_a_cur = torch.cat([s_cur_cat, a_cur_cat], dim=1)
        current_Q = self.critic.forward_sa(s_a_cur)  # (B,1) bfloat16

        # MSE 损失并更新
        # 现在 current_Q.shape == target_Q.shape == (B,1)
        critic_loss = F.mse_loss(current_Q.float(), target_Q.float())  # 修改：在 float32 中计算 MSE 损失
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        return float(critic_loss.item())

    # --------------------- Actor 更新 ---------------------
    def update_actor(self, replay_buffer=None, agent_n=None, batch_data=None):
        """
        - 本 agent 用 actor(obs_i) 输出 a_i
        - 其他 agent 的动作固定为 batch 中的 a_j（detach）
        - 目标：最大化集中式 critic 的 Q(s, a_1..a_N)（即最小化 -Q）
        """
        assert agent_n is not None, "agent_n (list of agents) is required."

        if batch_data is None:
            assert replay_buffer is not None, "Either batch_data or replay_buffer must be provided."
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        else:
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = batch_data

        a_cur_list = []
        for i, obs in enumerate(batch_obs_n):
            if i == self.agent_id:
                a_cur_list.append(self.actor(obs))
            else:
                a_cur_list.append(batch_a_n[i].detach())

        s_cur_cat = torch.cat(batch_obs_n, dim=1)
        a_cur_cat = torch.cat(a_cur_list, dim=1)
        s_a_cur = torch.cat([s_cur_cat, a_cur_cat], dim=1)

        actor_loss = - self.critic.forward_sa(s_a_cur).float().mean()  # 修改：在 float32 中计算策略梯度损失
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        return float(actor_loss.item())

    # --------------------- 软更新 target 网络 ---------------------
    def update_targets(self):
        with torch.no_grad():
            # 在 bfloat16 上执行软更新
            one = torch.tensor(1.0, dtype=torch.bfloat16, device=self.device)  # 修改：bfloat16 标量 1
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(tp.data * (one - self.tau) + p.data * self.tau)  # 修改：Critic 网络软更新 (bfloat16)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(tp.data * (one - self.tau) + p.data * self.tau)  # 修改：Actor 网络软更新 (bfloat16)
