import copy
import numpy as np
import torch
import torch.nn.functional as F

try:
    from networks import Actor, Critic_MADDPG as CriticNet, orthogonal_init
except Exception:
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

        # 可选开关
        self.use_grad_clip = bool(getattr(args, "use_grad_clip", True))
        self.use_orth_init = bool(getattr(args, "use_orthogonal_init", True))

        # ---------- 网络 ----------
        self.actor = Actor(args, agent_id)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNet(args)
        self.critic_target = copy.deepcopy(self.critic)

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

        # ---------- 确保网络使用 bfloat16 精度 ----------
        self.actor = self.actor.to(torch.bfloat16)
        self.actor_target = self.actor_target.to(torch.bfloat16)
        self.critic = self.critic.to(torch.bfloat16)
        self.critic_target = self.critic_target.to(torch.bfloat16)

        # ---------- 优化器 ----------
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        # 训练/评估模式
        self.actor.train()
        self.critic.train()
        self.actor_target.eval()
        self.critic_target.eval()

    def choose_action(self, obs, noise_std: float = 0.1):
        """
        obs: np.ndarray shape=(obs_dim,)
        return: np.ndarray shape=(action_dim,), clipped to [-max_action, max_action]
        """
        self.actor.eval()
        with torch.no_grad():
            # 将观察转换为 bfloat16 张量
            obs_tensor = torch.as_tensor(obs, dtype=torch.bfloat16).unsqueeze(0)
            a_tensor = self.actor(obs_tensor)
            # 转换为 float32 以便与 numpy 交互（numpy 不支持 bfloat16）
            a = a_tensor.to(torch.float32).cpu().numpy().reshape(-1)
        self.actor.train()

        if noise_std is not None and noise_std > 0:
            noise = np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)
            a = a + noise

        # 将动作裁剪到有效范围
        a = np.clip(a, -self.max_action, self.max_action)
        return a.astype(np.float32)

    def update_critic(self, replay_buffer=None, agent_n=None, cached_actions=None, batch_data=None):
        """
        batch_data: (batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n)
        - *_n 是长度为 N 的 list，元素张量形状均为 (batch_size, dim)
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

        # === 确保形状匹配 (B,1) ===
        r_i = batch_r_n[self.agent_id]                     # (B,1)
        done_i = batch_done_n[self.agent_id].to(r_i.dtype)  # (B,1) -> float

        with torch.no_grad():
            q_next = self.critic_target.forward_sa(s_a_next)  # (B,1)
            gamma_bf16 = torch.tensor(self.gamma, dtype=torch.bfloat16, device=q_next.device)
            target_Q = r_i + (1.0 - done_i) * gamma_bf16 * q_next  # (B,1)

        # 当前 Q(s,a)
        s_cur_cat = torch.cat(batch_obs_n, dim=1)
        a_cur_cat = torch.cat(batch_a_n, dim=1)
        s_a_cur = torch.cat([s_cur_cat, a_cur_cat], dim=1)
        current_Q = self.critic.forward_sa(s_a_cur)  # (B,1)

        # MSE 损失并更新
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        return float(critic_loss.item())

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

        actor_loss = - self.critic.forward_sa(s_a_cur).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        return float(actor_loss.item())

    def update_targets(self):
        with torch.no_grad():
            one_minus_tau = torch.tensor(1.0 - self.tau, dtype=torch.bfloat16)
            tau_bf = torch.tensor(self.tau, dtype=torch.bfloat16)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                # 用 bfloat16 执行软更新
                tp.data.mul_(one_minus_tau)
                tp.data.add_(p.data * tau_bf)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(one_minus_tau)
                tp.data.add_(p.data * tau_bf)
