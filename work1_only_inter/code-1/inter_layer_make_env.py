# inter_layer_make_env.py
import math
import random
import copy

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from maddpg import MADDPG
import myConfig
import myUtils


# -------------------- Running mean & std for normalization --------------------
class RunningMeanStd:
    """Running mean and std for online normalization (supports vector inputs)."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype='float64')
        self.var = np.ones(shape, dtype='float64')
        self.count = 1e-4

    def update(self, x):
        """
        Update with a batch x (shape: (batch, *shape) or (n,) for vector).
        """
        x = np.array(x, dtype='float64')
        if x.size == 0:
            return
        if x.ndim == 1:
            batch_mean = np.mean(x)
            batch_var = np.var(x)
            batch_count = x.shape[0]
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / (tot_count + 1e-12)
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (tot_count + 1e-12)
        new_var = M2 / (tot_count + 1e-12)
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


# ------------------------------ Runner class ----------------------------------
# inter_layer_make_env.py (节选)
class Runner:
    def __init__(self, args, seed, uavs, users, tb_logdir=None):
        self.args = args
        self.seed = seed
        self.uavs = uavs
        self.users = users
        self.total_task_num = myConfig.users_num * myConfig.user_tasks_num

        # 初始状态
        self.inter_layer_state_init, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)

        # UAV 数量和维度
        self.args.N = len(uavs)
        self.args.obs_dim_n = [6 for _ in range(self.args.N)]
        self.args.action_dim_n = [1 for _ in range(self.args.N)]

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # 创建智能体
        print("Algorithm: MADDPG + dual reward logging")
        self.agent_n = [MADDPG(args, agent_id) for agent_id in range(self.args.N)]

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.args)

        # 状态 & 奖励归一化
        self.obs_rms = [RunningMeanStd(shape=(self.args.obs_dim_n[i],)) for i in range(self.args.N)]
        self.reward_rms = RunningMeanStd(shape=())

        # 计数器
        self.total_steps = 0
        self.train_step = 0
        self.noise_std = self.args.noise_std_init

        # reward 平滑器
        self.last_global_reward = 0.0
        self.smooth_alpha = 0.9

        # TensorBoard
        self.writer = SummaryWriter(log_dir=tb_logdir) if tb_logdir else None

    def _normalize_obs(self, obs_list):
        normed = []
        for i, obs in enumerate(obs_list):
            arr = np.array(obs, dtype=np.float64)
            self.obs_rms[i].update(arr.reshape(1, -1))
            arr_norm = (arr - self.obs_rms[i].mean) / (self.obs_rms[i].std + 1e-8)
            normed.append(np.clip(arr_norm, -5.0, 5.0).tolist())
        return normed

    def run(self, lock=True):
        return self.inter_layer_env(lock)

    def inter_layer_env(self, lock):
        inter_layer_state = copy.deepcopy(self.inter_layer_state_init)

        have_dealt_task_num = 0
        inter_episode_total_time = 0
        total_profits = -math.inf
        inter_layer_end_number = 0

        per_agent_rewards_episode = [0.0 for _ in range(self.args.N)]

        while True:
            # ---------- 状态归一化 ----------
            norm_state = self._normalize_obs(inter_layer_state)

            # ---------- 动作选择 ----------
            if lock:
                inter_layer_actions = [ag.choose_action(obs, noise_std=self.noise_std)
                                       for ag, obs in zip(self.agent_n, norm_state)]
            else:
                inter_layer_actions = [np.random.uniform(-1, 1, size=(self.args.action_dim_n[i],))
                                       for i in range(self.args.N)]

            # ---------- UAV 移动 ----------
            move_distance = 10
            for i, uav in enumerate(self.uavs):
                act_val = float(np.array(inter_layer_actions[i]).flatten()[0])
                move_direction = 180 * act_val + 180
                uav.uav_move(math.radians(move_direction), move_distance)

            for _ in range(5):
                for uav in self.uavs:
                    uav.position[0] += random.randint(-5, 5)
                    uav.position[1] += random.randint(-5, 5)
                myUtils.get_uav_min_distance_restraint(self.uavs)

            # ---------- 任务分配 ----------
            for user in self.users:
                for task in user.tasks:
                    if task[-1]:
                        continue
                    best_time, best_idx = math.inf, -1
                    for idx, uav in enumerate(self.uavs):
                        d_u_v, d_u_cloud = uav.get_uv_distance(user.position, myConfig.cloud_position)
                        trans_user, trans_cloud = uav.get_uav_transmission_rate(d_u_v, d_u_cloud)
                        t = uav.get_uav_transmission_computing_time(task, trans_user, trans_cloud)
                        if t < best_time:
                            best_time, best_idx = t, idx
                    if best_idx != -1:
                        uav = self.uavs[best_idx]
                        if (task[0] in uav.cached_service_type or task[0] in uav.cached_content_type) and not task[-1]:
                            uav.storage_space -= task[0]
                            profit = (task[3] / best_time)
                            uav.profits += profit
                            uav.tasked_number += 1
                            task[-1] = True
                            have_dealt_task_num += 1
                            uav.storage_space += task[0]
                            uav.uav_cope_tasks_time += best_time
                            inter_episode_total_time += best_time

            # ---------- 奖励计算 ----------
            agent_rewards_raw = [(u.profits - u.base_cost) / (u.base_cost + 1e-5) for u in self.uavs]

            # 原始全局奖励
            outer_layer_reward_raw = np.mean(agent_rewards_raw) \
                                     + 0.01 * have_dealt_task_num \
                                     + 0.001 * self.users_number_covered
            reward_var = np.var(agent_rewards_raw)
            outer_layer_reward_raw -= 0.05 * reward_var

            # 平滑
            self.last_global_reward = self.smooth_alpha * self.last_global_reward \
                                      + (1 - self.smooth_alpha) * outer_layer_reward_raw
            smoothed_reward = self.last_global_reward

            # 归一化（训练用）
            self.reward_rms.update(np.array([smoothed_reward]))
            outer_layer_reward_norm = float(
                np.clip((smoothed_reward - self.reward_rms.mean) / (self.reward_rms.std + 1e-8), -5, 5)
            )
            agent_rewards_norm = [
                float(np.clip((r - self.reward_rms.mean) / (self.reward_rms.std + 1e-8), -5, 5))
                for r in agent_rewards_raw
            ]

            total_profits = max(sum(agent_rewards_raw), total_profits)

            # ---------- 下一状态 ----------
            inter_layer_state_, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)
            norm_state_next = []
            for i, obs in enumerate(inter_layer_state_):
                arr = np.array(obs, dtype=np.float64)
                arr_norm = (arr - self.obs_rms[i].mean) / (self.obs_rms[i].std + 1e-8)
                norm_state_next.append(np.clip(arr_norm, -5, 5).tolist())

            # ---------- 存储 & 训练 ----------
            if lock:
                self.replay_buffer.store_transition(norm_state, inter_layer_actions, agent_rewards_norm,
                                                    norm_state_next, [True] * self.args.N)
                inter_layer_state = inter_layer_state_
                self.total_steps += 1
                self.train_step += 1

                if self.args.use_noise_decay:
                    self.noise_std = max(self.noise_std * self.args.noise_std_decay, self.args.noise_std_min)

                if self.replay_buffer.current_size > self.args.batch_size:
                    # target 动作 + smoothing
                    batch_obs_n, _, _, batch_obs_next_n, _ = self.replay_buffer.sample()
                    with torch.no_grad():
                        target_actions = [ag.actor_target(batch_obs_next_n[i]) for i, ag in enumerate(self.agent_n)]
                        noisy_actions = []
                        for a in target_actions:
                            noise = (torch.randn_like(a) * self.args.target_noise_std).clamp(
                                -self.args.target_noise_clip, self.args.target_noise_clip)
                            noisy_actions.append((a + noise).clamp(-self.args.max_action, self.args.max_action))

                    # critic 更新
                    critic_losses = []
                    for _ in range(self.args.num_critic_updates):
                        for ag in self.agent_n:
                            loss_c = ag.update_critic(self.replay_buffer, self.agent_n, cached_actions=noisy_actions)
                            critic_losses.append(loss_c)

                    # actor 更新 (policy_delay)
                    actor_losses = []
                    if self.train_step % self.args.policy_delay == 0:
                        for ag in self.agent_n:
                            loss_a = ag.update_actor(self.replay_buffer, self.agent_n)
                            ag.update_targets()
                            actor_losses.append(loss_a)

                    # TensorBoard step 日志
                    if self.writer:
                        if critic_losses:
                            self.writer.add_scalar("train/critic_loss", np.mean(critic_losses), self.total_steps)
                        if actor_losses:
                            self.writer.add_scalar("train/actor_loss", np.mean(actor_losses), self.total_steps)
                        self.writer.add_scalar("train/global_reward_norm", outer_layer_reward_norm, self.total_steps)
                        self.writer.add_scalar("train/global_reward_raw", outer_layer_reward_raw, self.total_steps)

            # ---------- per-agent 累积奖励 ----------
            for i in range(self.args.N):
                per_agent_rewards_episode[i] += agent_rewards_norm[i]

            # ---------- 结束条件 ----------
            if have_dealt_task_num >= self.total_task_num * 0.9 or inter_layer_end_number >= 8:
                if self.writer:
                    self.writer.add_scalar("episode/outer_reward_norm", outer_layer_reward_norm, self.total_steps)
                    self.writer.add_scalar("episode/outer_reward_raw", outer_layer_reward_raw, self.total_steps)
                    self.writer.add_scalar("episode/total_profits", total_profits, self.total_steps)
                    for i, r in enumerate(per_agent_rewards_episode):
                        self.writer.add_scalar(f"episode/agent{i}_reward_sum", r, self.total_steps)
                break
            else:
                inter_layer_end_number += 1

        return have_dealt_task_num, total_profits, inter_episode_total_time, \
               self.users_number_covered, outer_layer_reward_norm, outer_layer_reward_raw