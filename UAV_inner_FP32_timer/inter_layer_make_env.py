# inter_layer_make_env.py
# 仅应用方案A：使用 args.max_episode_steps 作为每个episode的步数上限，
# 取代原先 inter_layer_end_number >= 8 的固定9步限制。

import math
import random
import copy
import time  # <-- 新增：高精度计时

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
import myUtils

# 初始化环境参数
users_num = 500
user_tasks_num = 50
cloud_position = [30000, 30000]  # 远程云的位置
d_uav_min = 400                  # 任意两个无人机之间的最小距离


# -------------------- Running mean & std for normalization --------------------
class RunningMeanStd:
    """Running mean and std for online normalization (支持向量输入)."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype='float64')
        self.var = np.ones(shape, dtype='float64')
        self.count = 1e-4

    def update(self, x):
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
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (tot_count + 1e-12)
        new_var = M2 / (tot_count + 1e-12)
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


# ------------------------------ Runner 类 ----------------------------------
class Runner:
    def __init__(self, args, seed, uavs, users, tb_logdir=None):
        self.args = args
        self.seed = seed
        self.uavs = uavs
        self.users = users
        self.total_task_num = users_num * user_tasks_num

        # 初始状态
        self.inter_layer_state_init, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)

        # UAV 数量和维度
        self.args.N = len(uavs)
        self.args.obs_dim_n = [6 for _ in range(self.args.N)]
        self.args.action_dim_n = [1 for _ in range(self.args.N)]

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # 创建智能体（Agent）
        print("Algorithm: MADDPG + dual reward logging")
        self.agent_n = [MADDPG(args, agent_id) for agent_id in range(self.args.N)]

        # 初始化多智能体经验回放池
        self.replay_buffer = ReplayBuffer(self.args)

        # 状态 & 奖励归一化
        self.obs_rms = [RunningMeanStd(shape=(self.args.obs_dim_n[i],)) for i in range(self.args.N)]
        self.reward_rms = RunningMeanStd(shape=())

        # 计数器
        self.total_steps = 0
        self.train_step = 0
        self.noise_std = self.args.noise_std_init

        # 奖励平滑器
        self.last_global_reward = 0.0
        self.smooth_alpha = 0.9

        # 日志记录
        self.writer = SummaryWriter(log_dir=tb_logdir) if tb_logdir else None

        # 打印控制
        self._print_every = getattr(self.args, "print_every", 1)
        self._avg_window = max(1, int(getattr(self.args, "print_avg_window", 20)))
        self._buf_reward_raw = []
        self._buf_reward_norm = []
        self._buf_profit = []

        # 训练开始提示只打印一次
        self.training_started = False

    def _normalize_obs(self, obs_list):
        # 利用 numpy 矢量化进行状态归一化
        obs_array = np.array(obs_list, dtype=np.float64)
        for i in range(len(obs_list)):
            self.obs_rms[i].update(obs_array[i:i+1])
        mean_array = np.stack([r.mean for r in self.obs_rms])
        std_array = np.stack([r.std for r in self.obs_rms])
        arr_norm = (obs_array - mean_array) / (std_array + 1e-8)
        arr_norm = np.clip(arr_norm, -5.0, 5.0).astype(np.float32)
        return [arr_norm[i] for i in range(len(obs_list))]

    def run(self, lock=True):
        # 运行一次环境模拟（lock=True 表示策略行动，False 表示随机探索）
        return self.inter_layer_env(lock)

    def inter_layer_env(self, lock):
        inter_layer_state = copy.deepcopy(self.inter_layer_state_init)

        have_dealt_task_num = 0
        inter_episode_total_time = 0
        total_profits = -math.inf

        per_agent_rewards_episode = [0.0 for _ in range(self.args.N)]
        step_idx = 0  # 显式步数计数器

        # ---- 分阶段计时累加器（单位：秒） ----
        t_action_select = 0.0
        t_move_constraint = 0.0
        t_task_assign = 0.0
        t_reward_calc = 0.0
        t_buffer_store = 0.0
        t_train_step = 0.0

        while True:
            # ---------- 状态归一化 ----------
            norm_state = self._normalize_obs(inter_layer_state)

            # ---------- 动作选择 ----------
            _t0 = time.perf_counter()
            if lock:
                inter_layer_actions = [ag.choose_action(obs, noise_std=self.noise_std) for ag, obs in zip(self.agent_n, norm_state)]
            else:
                inter_layer_actions = [np.random.uniform(-1, 1, size=(self.args.action_dim_n[i],)) for i in range(self.args.N)]
            t_action_select += (time.perf_counter() - _t0)

            # ---------- UAV 移动 + 约束 ----------
            _t0 = time.perf_counter()
            move_distance = 10
            for i, uav in enumerate(self.uavs):
                act_val = float(np.array(inter_layer_actions[i]).flatten()[0])
                move_direction = 180 * act_val + 180
                uav.uav_move(math.radians(move_direction), move_distance)

            # 最小距离扰动+约束
            for _ in range(5):
                for uav in self.uavs:
                    uav.position[0] += random.randint(-5, 5)
                    uav.position[1] += random.randint(-5, 5)
                myUtils.get_uav_min_distance_restraint(self.uavs)
            t_move_constraint += (time.perf_counter() - _t0)

            # ---------- 任务分配（向量化） ----------
            _t0 = time.perf_counter()
            pos_array = np.array([uav.position for uav in self.uavs], dtype=np.float32)
            for user in self.users:
                user_x, user_y = user.position[0], user.position[1]
                dx = pos_array[:, 0] - user_x
                dy = pos_array[:, 1] - user_y
                d_level = np.sqrt(dx**2 + dy**2)
                d_u_v_arr = np.sqrt(d_level**2 + pos_array[:, 2]**2)
                d_u_cloud_arr = np.sqrt((pos_array[:, 0] - cloud_position[0])**2 + (pos_array[:, 1] - cloud_position[1])**2)
                B_arr = np.array([uav.B for uav in self.uavs], dtype=np.float32)
                B_cloud_arr = np.array([uav.B_cloud for uav in self.uavs], dtype=np.float32)
                P_u_arr = np.array([uav.P_u for uav in self.uavs], dtype=np.float32)
                P_v_w_arr = np.array([uav.P_v_w for uav in self.uavs], dtype=np.float32)
                r_arr = np.array([uav.r for uav in self.uavs], dtype=np.float32)
                sigma2_arr = np.array([uav.sigma2 for uav in self.uavs], dtype=np.float32)
                N_arr = np.array([uav.N for uav in self.uavs], dtype=np.float32)

                trans_cloud_arr = 10 * B_cloud_arr * np.log2(1 + 10000 * P_u_arr * np.power(d_u_cloud_arr/100.0, -r_arr) / (sigma2_arr + N_arr))
                trans_user_arr = 100 * B_arr * np.log2(1 + 5000 * (P_v_w_arr * np.power(d_u_v_arr/10.0, -r_arr) / (sigma2_arr + N_arr)))

                for task in user.tasks:
                    if task[-1]:
                        continue
                    size = task[1]
                    computing_delay = np.zeros(len(self.uavs), dtype=np.float32)

                    service_mask = np.array([task[0] in uav.cached_service_type for uav in self.uavs])
                    content_mask = np.array([task[0] in uav.cached_content_type for uav in self.uavs])
                    if service_mask.any():
                        rand_factors = np.random.randint(3, 9, size=service_mask.sum()).astype(np.float32)
                        F_cpu_arr = np.array([uav.F_cpu for uav in self.uavs], dtype=np.float32)
                        computing_delay[service_mask] = size / (rand_factors * (F_cpu_arr[service_mask] / 30.0))
                    rest_mask = ~(service_mask | content_mask)
                    computing_delay[rest_mask] = 2 * (size / trans_cloud_arr[rest_mask])

                    UV_rate_arr = np.array([uav.U_V_transmission_rate for uav in self.uavs], dtype=np.float32)
                    total_time_arr = computing_delay + (size / trans_user_arr) + (size * 0.6 / UV_rate_arr)
                    best_idx = int(np.argmin(total_time_arr))
                    best_time = float(total_time_arr[best_idx])
                    if (service_mask[best_idx] or content_mask[best_idx]) and not task[-1]:
                        uav = self.uavs[best_idx]
                        uav.storage_space -= task[0]
                        profit = task[3] / best_time
                        uav.profits += profit
                        uav.tasked_number += 1
                        task[-1] = True
                        have_dealt_task_num += 1
                        uav.storage_space += task[0]
                        uav.uav_cope_tasks_time += best_time
                        inter_episode_total_time += best_time
            t_task_assign += (time.perf_counter() - _t0)

            # ---------- 奖励计算 ----------
            _t0 = time.perf_counter()
            agent_rewards_raw = [(u.profits - u.base_cost) / (u.base_cost + 1e-5) for u in self.uavs]
            outer_layer_reward_raw = np.mean(agent_rewards_raw) + 0.01 * have_dealt_task_num + 0.001 * self.users_number_covered
            reward_var = np.var(agent_rewards_raw)
            outer_layer_reward_raw -= 0.05 * reward_var

            # 平滑/归一化
            self.last_global_reward = self.smooth_alpha * self.last_global_reward + (1 - self.smooth_alpha) * outer_layer_reward_raw
            smoothed_reward = self.last_global_reward
            self.reward_rms.update(np.array([smoothed_reward]))
            outer_layer_reward_norm = float(np.clip((smoothed_reward - self.reward_rms.mean) / (self.reward_rms.std + 1e-8), -5, 5))
            agent_rewards_norm = [float(np.clip((r - self.reward_rms.mean) / (self.reward_rms.std + 1e-8), -5, 5)) for r in agent_rewards_raw]

            # 统计 & 打印用
            profit_sum = float(np.sum([u.profits for u in self.uavs]))
            self._buf_reward_raw.append(float(outer_layer_reward_raw))
            self._buf_reward_norm.append(float(outer_layer_reward_norm))
            self._buf_profit.append(profit_sum)
            if len(self._buf_reward_raw) > self._avg_window:
                self._buf_reward_raw = self._buf_reward_raw[-self._avg_window:]
                self._buf_reward_norm = self._buf_reward_norm[-self._avg_window:]
                self._buf_profit = self._buf_profit[-self._avg_window:]
            avg_reward_raw = float(np.mean(self._buf_reward_raw))
            avg_reward_norm = float(np.mean(self._buf_reward_norm))
            avg_profit = float(np.mean(self._buf_profit))

            total_profits = max(sum(agent_rewards_raw), total_profits)
            t_reward_calc += (time.perf_counter() - _t0)

            # ---------- 下一状态 ----------
            inter_layer_state_, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)
            mean_array = np.array([r.mean for r in self.obs_rms], dtype=np.float64)
            std_array = np.array([r.std for r in self.obs_rms], dtype=np.float64)
            arr_next_full = np.array(inter_layer_state_, dtype=np.float64)
            arr_norm_next = (arr_next_full - mean_array) / (std_array + 1e-8)
            arr_norm_next = np.clip(arr_norm_next, -5, 5).astype(np.float32)
            norm_state_next = [arr_norm_next[i] for i in range(self.args.N)]

            # ---------- 步进 & 结束条件 ----------
            step_idx += 1
            done_condition = (
                have_dealt_task_num >= self.total_task_num * 0.9
                or step_idx >= self.args.max_episode_steps
            )
            done_list = [bool(done_condition)] * self.args.N

            # ---------- 经验存储 ----------
            _t0 = time.perf_counter()
            self.replay_buffer.store_transition(norm_state, inter_layer_actions, agent_rewards_norm, norm_state_next, done_list)
            t_buffer_store += (time.perf_counter() - _t0)

            # ---------- 训练触发 ----------
            if lock:
                self.train_step += 1
                self.total_steps += 1
                if self.args.use_noise_decay:
                    self.noise_std = max(self.noise_std * self.args.noise_std_decay, self.args.noise_std_min)

                if self.replay_buffer.current_size >= self.args.batch_size:
                    if not self.training_started:
                        self.training_started = True
                        print("=== MADDPG Training Started ===")

                    _t0_train = time.perf_counter()

                    # 采样一个批次
                    batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = self.replay_buffer.sample()

                    # 目标动作 + smoothing（TD3风格）
                    with torch.no_grad():
                        target_actions = [ag.actor_target(batch_obs_next_n[i]) for i, ag in enumerate(self.agent_n)]
                        noisy_actions = []
                        for a in target_actions:
                            noise = (torch.randn_like(a) * self.args.target_noise_std).clamp(
                                -self.args.target_noise_clip, self.args.target_noise_clip
                            )
                            noisy_actions.append((a + noise).clamp(-self.args.max_action, self.args.max_action))

                    # 更新 Critic
                    critic_losses = []
                    for _ in range(self.args.num_critic_updates):
                        for ag in self.agent_n:
                            loss_c = ag.update_critic(
                                batch_data=(batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n),
                                agent_n=self.agent_n,
                                cached_actions=noisy_actions
                            )
                            critic_losses.append(loss_c)

                    # 延迟更新 Actor（policy_delay）
                    actor_losses = []
                    if self.train_step % self.args.policy_delay == 0:
                        for ag in self.agent_n:
                            loss_a = ag.update_actor(
                                batch_data=(batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n),
                                agent_n=self.agent_n
                            )
                            ag.update_targets()
                            actor_losses.append(loss_a)

                    # TensorBoard 记录（可选）
                    if self.writer:
                        if critic_losses:
                            self.writer.add_scalar("train/critic_loss", np.mean(critic_losses), self.total_steps)
                        if actor_losses:
                            self.writer.add_scalar("train/actor_loss", np.mean(actor_losses), self.total_steps)
                        self.writer.add_scalar("train/global_reward_norm", outer_layer_reward_norm, self.total_steps)
                        self.writer.add_scalar("train/global_reward_raw", outer_layer_reward_raw, self.total_steps)
                        self.writer.add_scalar("train/avg_reward_norm", avg_reward_norm, self.total_steps)
                        self.writer.add_scalar("train/avg_reward_raw", avg_reward_raw, self.total_steps)
                        self.writer.add_scalar("train/avg_profit", avg_profit, self.total_steps)

                    t_train_step += (time.perf_counter() - _t0_train)
            else:
                # warm-up 阶段也累计总步数（仅用于打印频率控制）
                self.total_steps += 1

            # ---------- 累积每个 agent 的奖励 ----------
            for i in range(self.args.N):
                per_agent_rewards_episode[i] += agent_rewards_norm[i]

            # ---------- 收尾 ----------
            if done_condition:
                if self.writer:
                    self.writer.add_scalar("episode/outer_reward_norm", outer_layer_reward_norm, self.total_steps)
                    self.writer.add_scalar("episode/outer_reward_raw", outer_layer_reward_raw, self.total_steps)
                    self.writer.add_scalar("episode/total_profits", total_profits, self.total_steps)
                    for i, r in enumerate(per_agent_rewards_episode):
                        self.writer.add_scalar(f"episode/agent{i}_reward_sum", r, self.total_steps)
                break
            else:
                inter_layer_state = inter_layer_state_

        # 返回：原有6项 + 步数 + 分阶段计时累加（秒）
        timing_sums = {
            "action_select_s": t_action_select,
            "move_constraint_s": t_move_constraint,
            "task_assign_s": t_task_assign,
            "reward_calc_s": t_reward_calc,
            "buffer_store_s": t_buffer_store,
            "train_step_s": t_train_step,
        }
        return (
            have_dealt_task_num, total_profits, inter_episode_total_time,
            self.users_number_covered, outer_layer_reward_norm, outer_layer_reward_raw,
            step_idx, timing_sums
        )
