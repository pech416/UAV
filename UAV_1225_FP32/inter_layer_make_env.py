# inter_layer_make_env.py
# 使用向量化算子（vectorized_ops）替换原始循环实现，并确保 episode 具有明确终止条件。

import math
import random
import copy
import time
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
import myUtils
from vectorized_ops import VectorizedOps as VO

# ================== 环境常量 ==================
users_num = 100
user_tasks_num = 10
cloud_position = [30000, 30000]  # 远程云
d_uav_min = 400                  # 任意两机最小间距

# -------------------- Running mean & std --------------------
class RunningMeanStd:
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

class EpisodeTiming:
    """Accumulates per-episode timings (seconds) and provides ms views."""
    __slots__ = ("sec", "_fc1_ms_sum", "_fc2_ms_sum", "_fc3_ms_sum", "_noise_ms_sum", "_action_count")
    def __init__(self):
        self.sec = {
            'state_norm': 0.0,
            'action_select': 0.0,
            'uav_move': 0.0,
            'task_assign': 0.0,
            'distance': 0.0,
            'trans_rate': 0.0,
            'cache_check': 0.0,
            'comp_time': 0.0,
            'reward_calc': 0.0,
            'next_state': 0.0,
            'store_train': 0.0,
        }
        self._fc1_ms_sum = 0.0
        self._fc2_ms_sum = 0.0
        self._fc3_ms_sum = 0.0
        self._noise_ms_sum = 0.0
        self._action_count = 0
    def add(self, key, seconds):
        if key in self.sec:
            self.sec[key] += float(seconds)
    def add_action_sub_ms(self, fc1_ms=0.0, fc2_ms=0.0, fc3_ms=0.0, noise_ms=0.0):
        self._fc1_ms_sum += float(fc1_ms)
        self._fc2_ms_sum += float(fc2_ms)
        self._fc3_ms_sum += float(fc3_ms)
        self._noise_ms_sum += float(noise_ms)
        self._action_count += 1
    def action_sub_avgs_ms(self):
        c = max(1, int(self._action_count))
        return (self._fc1_ms_sum / c, self._fc2_ms_sum / c, self._fc3_ms_sum / c, self._noise_ms_sum / c)
    def total_ms(self):
        return 1000.0 * sum(self.sec.values())
    def _ms(self, key):
        return 1000.0 * float(self.sec.get(key, 0.0))
    def print_console(self):
        fc1, fc2, fc3, noise = self.action_sub_avgs_ms()
        print("===== Episode Timing (ms) =====")
        print(f"1. State Norm:        {self._ms('state_norm'):.4f}")
        print(f"2. Action Select:     {self._ms('action_select'):.4f}")
        print(f"   - 2.1 Actor FC1 (NN):      {fc1:.4f}")
        print(f"   - 2.2 Actor FC2 (NN):      {fc2:.4f}")
        print(f"   - 2.3 Actor FC3 (NN):      {fc3:.4f}")
        print(f"   - 2.4 Noise Addition:      {noise:.4f}")
        print(f"3. UAV Move:          {self._ms('uav_move'):.4f}")
        print(f"4. Task Assign:       {self._ms('task_assign'):.4f}")
        print(f"   - 4.1 Distance (sqrt):     {self._ms('distance'):.4f}")
        print(f"   - 4.2 Trans Rate (log/pow):{self._ms('trans_rate'):.4f}")
        print(f"   - 4.3 Cache Check (loop):  {self._ms('cache_check'):.4f}")
        print(f"   - 4.4 Comp Time (scalar):  {self._ms('comp_time'):.4f}")
        print(f"5. Reward Calc:       {self._ms('reward_calc'):.4f}")
        print(f"6. Next State:        {self._ms('next_state'):.4f}")
        print(f"7. Store & Train:     {self._ms('store_train'):.4f}")
        print(f"Total Episode Time:   {self.total_ms():.4f}")

# ------------------------------ Runner ----------------------------------
class Runner:
    def __init__(self, args, seed, uavs, users, tb_logdir=None):
        self.args = args
        self.seed = seed
        self.uavs = uavs
        self.users = users
        self.total_task_num = users_num * user_tasks_num

        # 初始状态
        self.inter_layer_state_init, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)

        # UAV 数量与维度
        self.args.N = len(uavs)
        self.args.obs_dim_n = [6 for _ in range(self.args.N)]
        self.args.action_dim_n = [1 for _ in range(self.args.N)]

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        print("Algorithm: MADDPG + dual reward logging")
        self.agent_n = [MADDPG(args, agent_id) for agent_id in range(self.args.N)]

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.args)

        # 归一化器
        self.obs_rms = [RunningMeanStd(shape=(self.args.obs_dim_n[i],)) for i in range(self.args.N)]
        self.reward_rms = RunningMeanStd(shape=())

        # 计数与噪声
        self.total_steps = 0
        self.train_step = 0
        self.noise_std = self.args.noise_std_init

        # 平滑
        self.last_global_reward = 0.0
        self.smooth_alpha = 0.9

        # 日志
        self.writer = SummaryWriter(log_dir=tb_logdir) if tb_logdir else None

        # 打印缓存
        self._print_every = getattr(self.args, "print_every", 1)
        self._avg_window = max(1, int(getattr(self.args, "print_avg_window", 20)))
        self._buf_reward_raw, self._buf_reward_norm, self._buf_profit = [], [], []

        # 只打印一次“开始训练”
        self.training_started = False

    def _normalize_obs(self, obs_list):
        arr = np.array(obs_list, dtype=np.float64)
        for i in range(len(obs_list)):
            self.obs_rms[i].update(arr[i:i+1])
        mean = np.stack([r.mean for r in self.obs_rms])
        std = np.stack([r.std for r in self.obs_rms])
        arr_norm = (arr - mean) / (std + 1e-8)
        arr_norm = np.clip(arr_norm, -5.0, 5.0).astype(np.float32)
        return [arr_norm[i] for i in range(len(obs_list))]

    def run(self, lock=True):
        return self.inter_layer_env(lock)

    def inter_layer_env(self, lock):
        inter_layer_state = copy.deepcopy(self.inter_layer_state_init)

        have_dealt_task_num = 0
        inter_episode_total_time = 0.0
        total_profits = -math.inf

        per_agent_rewards_episode = [0.0 for _ in range(self.args.N)]
        step_idx = 0
        timing = EpisodeTiming()
        episode_time_ms = 0.0

        while True:
            # -------- 归一化状态 --------
            _t0 = time.perf_counter()
            norm_state = self._normalize_obs(inter_layer_state)
            timing.add('state_norm', time.perf_counter() - _t0)

            # -------- 选择动作 --------
            _t0 = time.perf_counter()
            if lock:
                inter_layer_actions = []
                for ag, obs in zip(self.agent_n, norm_state):
                    _sub = {}
                    try:
                        a = ag.choose_action(obs, noise_std=self.noise_std, timing=_sub)
                    except TypeError:
                        a = ag.choose_action(obs, noise_std=self.noise_std)
                    inter_layer_actions.append(a)
                    fc1 = _sub.get('actor_fc1_ms', _sub.get('fc1_ms', 0.0))
                    fc2 = _sub.get('actor_fc2_ms', _sub.get('fc2_ms', 0.0))
                    fc3 = _sub.get('actor_fc3_ms', _sub.get('fc3_ms', 0.0))
                    noise = _sub.get('noise_ms', _sub.get('noise_addition_ms', 0.0))
                    timing.add_action_sub_ms(fc1, fc2, fc3, noise)
            else:
                inter_layer_actions = [
                    np.random.uniform(-1, 1, size=(self.args.action_dim_n[i],))
                    for i in range(self.args.N)
                ]
            timing.add('action_select', time.perf_counter() - _t0)
            # -------- UAV 移动（向量化） --------
            _t0 = time.perf_counter()
            uav_positions = np.array([uav.position for uav in self.uavs], dtype=np.float32)  # (N,3)
            actions = np.array([float(np.array(a).reshape(-1)[0]) for a in inter_layer_actions], dtype=np.float32)  # (N,)
            new_positions = VO.move_uavs_batch(uav_positions, actions, move_distance=10.0)
            # 最小间距约束（向量化）
            new_positions = VO.enforce_min_distance_batch(
                new_positions, min_distance=d_uav_min, max_range=self.uavs[0].uav_movable_range
            )
            for i, u in enumerate(self.uavs):
                u.position = new_positions[i].tolist()

            timing.add('uav_move', time.perf_counter() - _t0)
            # -------- 任务分配（向量化） --------
            _t0 = time.perf_counter()
            # 使用 find_best_uav_for_tasks 统一考虑“缓存”和“云端”两类路径的耗时；得到 (best_uav, min_time)
            assignments = VO.find_best_uav_for_tasks([user.tasks for user in self.users], self.uavs, self.users, cloud_position, timing=timing)
            # 将分配结果落地：统一标记完成并更新统计 —— 无需再额外判断是否缓存命中
            for (user_idx, task_idx), (best_idx, best_time) in assignments.items():
                user = self.users[user_idx]
                task = user.tasks[task_idx]
                if task[-1]:
                    continue
                uav = self.uavs[best_idx]

                # 保持与原实现一致的记账（存取空间按原逻辑对 task[0] 操作）
                uav.storage_space -= task[0]
                profit = task[3] / max(best_time, 1e-6)
                uav.profits += profit
                uav.tasked_number += 1
                task[-1] = True
                have_dealt_task_num += 1
                uav.storage_space += task[0]
                uav.uav_cope_tasks_time += best_time
                inter_episode_total_time += best_time

            timing.add('task_assign', time.perf_counter() - _t0)
            # -------- 奖励计算（向量化） --------
            _t0 = time.perf_counter()
            profits_arr = np.array([u.profits for u in self.uavs], dtype=np.float32)
            base_cost_arr = np.array([u.base_cost for u in self.uavs], dtype=np.float32)
            tasked_arr = np.array([u.tasked_number for u in self.uavs], dtype=np.int32)
            sill_arr = np.array([u.task_num_sill for u in self.uavs], dtype=np.int32)

            rewards_arr, updated_base, updated_sill = VO.calculate_uav_rewards_batch(
                profits_arr, base_cost_arr, tasked_arr, sill_arr
            )
            # 回写动态成本与阈值
            for i, u in enumerate(self.uavs):
                u.base_cost = float(updated_base[i])
                u.task_num_sill = int(updated_sill[i])

            # 归一化到“raw ratio”与全局 reward
            agent_rewards_raw = (rewards_arr / (base_cost_arr + 1e-5)).tolist()
            outer_layer_reward_raw = float(np.mean(agent_rewards_raw)) \
                                     + 0.01 * have_dealt_task_num \
                                     + 0.001 * self.users_number_covered
            outer_layer_reward_raw -= 0.05 * np.var(agent_rewards_raw)

            # 平滑 + 标准化
            self.last_global_reward = self.smooth_alpha * self.last_global_reward + \
                                      (1 - self.smooth_alpha) * outer_layer_reward_raw
            smoothed = self.last_global_reward
            self.reward_rms.update(np.array([smoothed], dtype=np.float64))
            outer_layer_reward_norm = float(np.clip(
                (smoothed - self.reward_rms.mean) / (self.reward_rms.std + 1e-8), -5, 5
            ))
            agent_rewards_norm = [float(np.clip(
                (r - self.reward_rms.mean) / (self.reward_rms.std + 1e-8), -5, 5
            )) for r in agent_rewards_raw]

            # 打印用缓存
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

            timing.add('reward_calc', time.perf_counter() - _t0)
            # -------- 下一状态 --------
            _t0 = time.perf_counter()
            inter_layer_state_, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)
            mean = np.array([r.mean for r in self.obs_rms], dtype=np.float64)
            std = np.array([r.std for r in self.obs_rms], dtype=np.float64)
            arr_next = np.array(inter_layer_state_, dtype=np.float64)
            arr_next = np.clip((arr_next - mean) / (std + 1e-8), -5, 5).astype(np.float32)
            norm_state_next = [arr_next[i] for i in range(self.args.N)]

            timing.add('next_state', time.perf_counter() - _t0)
            # -------- 训练/记录 --------
            _t0 = time.perf_counter()
            step_idx += 1
            done_condition = (have_dealt_task_num >= self.total_task_num * 0.9) or \
                             (step_idx >= self.args.max_episode_steps)
            done_list = [bool(done_condition)] * self.args.N

            # 经验
            self.replay_buffer.store_transition(norm_state, inter_layer_actions, agent_rewards_norm, norm_state_next, done_list)

            if lock:
                self.train_step += 1
                self.total_steps += 1
                if getattr(self.args, "use_noise_decay", True):
                    self.noise_std = max(self.noise_std * self.args.noise_std_decay, self.args.noise_std_min)

                if self.replay_buffer.current_size >= self.args.batch_size:
                    if not self.training_started:
                        self.training_started = True
                        print("=== MADDPG Training Started ===")

                    # 采样
                    batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = self.replay_buffer.sample()

                    # 目标动作（TD3-style smoothing）
                    with torch.no_grad():
                        target_actions = [ag.actor_target(batch_obs_next_n[i]) for i, ag in enumerate(self.agent_n)]
                        noisy_actions = []
                        for a in target_actions:
                            noise = (torch.randn_like(a) * self.args.target_noise_std).clamp(
                                -self.args.target_noise_clip, self.args.target_noise_clip
                            )
                            noisy_actions.append((a + noise).clamp(-self.args.max_action, self.args.max_action))

                    # 更新 critic
                    critic_losses = []
                    for _ in range(self.args.num_critic_updates):
                        for ag in self.agent_n:
                            lc = ag.update_critic(
                                batch_data=(batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n),
                                agent_n=self.agent_n,
                                cached_actions=noisy_actions
                            )
                            critic_losses.append(lc)

                    # 延迟更新 actor
                    actor_losses = []
                    if self.train_step % self.args.policy_delay == 0:
                        for ag in self.agent_n:
                            la = ag.update_actor(
                                batch_data=(batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n),
                                agent_n=self.agent_n
                            )
                            ag.update_targets()
                            actor_losses.append(la)

                    # TensorBoard
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
            else:
                self.total_steps += 1  # warmup 期间也递增，便于统一打印节奏

            timing.add('store_train', time.perf_counter() - _t0)
            # -------- 收尾或继续 --------
            if done_condition:
                if self.writer:
                    self.writer.add_scalar("episode/outer_reward_norm", outer_layer_reward_norm, self.total_steps)
                    self.writer.add_scalar("episode/outer_reward_raw", outer_layer_reward_raw, self.total_steps)
                    self.writer.add_scalar("episode/total_profits", total_profits, self.total_steps)
                    for i, r in enumerate(per_agent_rewards_episode):
                        self.writer.add_scalar(f"episode/agent{i}_reward_sum", r, self.total_steps)
                timing.print_console()
                episode_time_ms = timing.total_ms()
                break
            else:
                inter_layer_state = inter_layer_state_

        # 返回签名保持不变，兼容主程序 main_sweep_dual_reward.py 调用
        return have_dealt_task_num, total_profits, inter_episode_total_time, self.users_number_covered, \
               outer_layer_reward_norm, outer_layer_reward_raw, episode_time_ms