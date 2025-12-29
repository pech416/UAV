# maddpg.py
import copy
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from networks import Actor, Critic_MADDPG as CriticNet, orthogonal_init
except Exception:  # noqa
    from networks import Actor, Critic_MATD3 as CriticNet, orthogonal_init


def _set_many(d: Dict[str, Any], keys, value: float):
    for k in keys:
        try:
            d[k] = float(value)
        except Exception:
            pass


class MADDPG(object):
    def __init__(self, args, agent_id: int):
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

        self.use_grad_clip = bool(getattr(args, "use_grad_clip", True))
        self.use_orth_init = bool(getattr(args, "use_orthogonal_init", True))

        # device / bf16
        dev_str = getattr(args, "device", None)
        if dev_str is None:
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev_str)

        # self.use_bfloat16 = bool(getattr(args, "use_bfloat16", True))
        # self.dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        self.dtype = torch.float32


        self.actor = Actor(args, agent_id).to(self.device, dtype=self.dtype)
        self.actor_target = copy.deepcopy(self.actor).to(self.device, dtype=self.dtype)

        self.critic = CriticNet(args).to(self.device, dtype=self.dtype)
        self.critic_target = copy.deepcopy(self.critic).to(self.device, dtype=self.dtype)

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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.actor.train()
        self.critic.train()
        self.actor_target.eval()
        self.critic_target.eval()

    # --------------------- choose_action (supports timing) ---------------------
    def choose_action(self, obs, noise_std: float = 0.1, timing: Optional[Dict[str, Any]] = None, **kwargs):
        """
        timing 兼容策略：
        - 不带 _ms 的 key -> 写入“秒”
        - 带 _ms 的 key   -> 写入“毫秒”
        这样无论环境是 (value*1000) 还是直接打印 ms，都能对上。
        """
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype).unsqueeze(0)

        timing_enabled = timing is not None
        if timing_enabled:
            t_total0 = time.perf_counter()

        self.actor.eval()
        with torch.no_grad():
            # FC1
            t0 = time.perf_counter()
            x = F.relu(self.actor.fc1(obs_tensor))
            t1 = time.perf_counter()

            # FC2
            x = F.relu(self.actor.fc2(x))
            t2 = time.perf_counter()

            # FC3 + tanh + to_cpu_numpy（把转换开销并入 FC3，避免“子项加和 << total”）
            a_tensor = self.max_action * torch.tanh(self.actor.fc3(x))
            a = a_tensor.to(dtype=torch.float32).cpu().numpy().reshape(-1)
            t3 = time.perf_counter()

        self.actor.train()

        # Noise + clip
        if timing_enabled:
            t_noise0 = time.perf_counter()

        noise_added = False
        if noise_std is not None and noise_std > 0:
            noise = np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)
            a = a + noise
            noise_added = True

        a = np.clip(a, -self.max_action, self.max_action).astype(np.float32)

        if timing_enabled:
            t_noise1 = time.perf_counter()
            t_total1 = time.perf_counter()

            fc1_s = (t1 - t0)
            fc2_s = (t2 - t1)
            fc3_s = (t3 - t2)
            noise_s = (t_noise1 - t_noise0) if noise_added else 0.0
            total_s = (t_total1 - t_total0)

            fc1_ms = fc1_s * 1000.0
            fc2_ms = fc2_s * 1000.0
            fc3_ms = fc3_s * 1000.0
            noise_ms = noise_s * 1000.0
            total_ms = total_s * 1000.0

            # ---- 写入“秒”的 key（环境若 *1000 -> ms 正确）----
            _set_many(timing, ["fc1", "actor_fc1", "Actor FC1 (NN)", "Actor FC1", "2.1", "2_1"], fc1_s)
            _set_many(timing, ["fc2", "actor_fc2", "Actor FC2 (NN)", "Actor FC2", "2.2", "2_2"], fc2_s)
            _set_many(timing, ["fc3", "actor_fc3", "Actor FC3 (NN)", "Actor FC3", "2.3", "2_3"], fc3_s)

            # noise：写入尽可能多的兼容 key（都用“秒”）
            _set_many(
                timing,
                ["noise", "noise_add", "noise_addition", "Noise Addition", "Noise Addition (NN)", "2.4", "2_4"],
                noise_s
            )

            _set_many(
                timing,
                ["total", "act_total", "action_select", "Action Select", "action_select_total", "act_total_s"],
                total_s
            )

            # ---- 写入“毫秒”的 key（环境若直接打印 _ms -> ms 正确）----
            _set_many(timing, ["fc1_ms", "actor_fc1_ms", "2.1_ms", "2_1_ms"], fc1_ms)
            _set_many(timing, ["fc2_ms", "actor_fc2_ms", "2.2_ms", "2_2_ms"], fc2_ms)
            _set_many(timing, ["fc3_ms", "actor_fc3_ms", "2.3_ms", "2_3_ms"], fc3_ms)
            _set_many(timing, ["noise_ms", "noise_addition_ms", "2.4_ms", "2_4_ms"], noise_ms)
            _set_many(timing, ["total_ms", "act_total_ms", "action_select_ms"], total_ms)

            # 辅助信息（不影响打印）
            timing["noise_std"] = float(noise_std if noise_std is not None else 0.0)
            timing["noise_added"] = bool(noise_added)
            timing["_units_non_ms_keys"] = "seconds"
            timing["_units_ms_keys"] = "milliseconds"

        return a

    # --------------------- deprecated train() ---------------------
    def train(self, *args, **kwargs):
        raise NotImplementedError("Deprecated: use update_critic / update_actor / update_targets inside the env loop.")

    # --------------------- Critic update ---------------------
    def update_critic(self, replay_buffer=None, agent_n=None, cached_actions=None, batch_data=None):
        assert agent_n is not None, "agent_n (list of agents) is required."

        if batch_data is None:
            assert replay_buffer is not None, "Either batch_data or replay_buffer must be provided."
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        else:
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = batch_data

        batch_obs_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_obs_n]
        batch_a_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_a_n]
        batch_r_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_r_n]
        batch_obs_next_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_obs_next_n]
        batch_done_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_done_n]

        if cached_actions is None:
            with torch.no_grad():
                cached_actions = [ag.actor_target(batch_obs_next_n[i]) for i, ag in enumerate(agent_n)]
        else:
            cached_actions = [torch.as_tensor(a, device=self.device, dtype=self.dtype) for a in cached_actions]

        s_next_cat = torch.cat(batch_obs_next_n, dim=1)
        a_next_cat = torch.cat(cached_actions, dim=1)
        s_a_next = torch.cat([s_next_cat, a_next_cat], dim=1)

        r_i = batch_r_n[self.agent_id]
        done_i = batch_done_n[self.agent_id].to(r_i.dtype)

        with torch.no_grad():
            q_next = self.critic_target.forward_sa(s_a_next)
            target_Q = r_i + (1.0 - done_i) * self.gamma * q_next

        s_cur_cat = torch.cat(batch_obs_n, dim=1)
        a_cur_cat = torch.cat(batch_a_n, dim=1)
        s_a_cur = torch.cat([s_cur_cat, a_cur_cat], dim=1)
        current_Q = self.critic.forward_sa(s_a_cur)

        critic_loss = F.mse_loss(current_Q.float(), target_Q.float())
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        return float(critic_loss.item())

    # --------------------- Actor update ---------------------
    def update_actor(self, replay_buffer=None, agent_n=None, batch_data=None):
        assert agent_n is not None, "agent_n (list of agents) is required."

        if batch_data is None:
            assert replay_buffer is not None, "Either batch_data or replay_buffer must be provided."
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        else:
            batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = batch_data

        batch_obs_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_obs_n]
        batch_a_n = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in batch_a_n]

        a_cur_list = []
        for i, obs in enumerate(batch_obs_n):
            if i == self.agent_id:
                a_cur_list.append(self.actor(obs))
            else:
                a_cur_list.append(batch_a_n[i].detach())

        s_cur_cat = torch.cat(batch_obs_n, dim=1)
        a_cur_cat = torch.cat(a_cur_list, dim=1)
        s_a_cur = torch.cat([s_cur_cat, a_cur_cat], dim=1)

        actor_loss = -self.critic.forward_sa(s_a_cur).float().mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        return float(actor_loss.item())

    # --------------------- Soft update targets ---------------------
    def update_targets(self):
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
