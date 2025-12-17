import math
import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from make_env import make_env
import argparse
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
import copy
import myConfig
import myUtils


class Runner:
    def __init__(self, args, seed, uavs, users):
        self.args = args
        self.seed = seed
        self.uavs = uavs
        self.users = users
        self.total_task_num = myConfig.users_num * myConfig.user_tasks_num
        self.inter_layer_state_init, self.users_number_covered = inter_layer_state, users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)
        self.args.N = len(uavs) # The number of agents
        self.args.obs_dim_n = [6 for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [1 for i in range(self.args.N)]  # actions dimensions of N agents

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        print("Algorithm: MADDPG")
        self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]

        self.replay_buffer = ReplayBuffer(self.args)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, lock=True):
        # self.evaluate_policy()
        have_dealt_task_num, total_profits, inter_episode_total_time, users_number_covered, outer_layer_reward = self.inter_layer_env(lock)
        print("have_dealt_task_num", have_dealt_task_num)
        print("total_profits", total_profits)
        print("inter_episode_total_time", inter_episode_total_time)
        print("users_number_covered", users_number_covered)
        print("outer_layer_reward", outer_layer_reward)
        return have_dealt_task_num, total_profits, inter_episode_total_time, users_number_covered, outer_layer_reward

    def evaluate_policy(self, lock=False):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            inter_layer_state = self.inter_layer_state_init
            episode_reward = 0
            inter_rewards_profits = self.inter_layer_env(lock)
            evaluate_reward += inter_rewards_profits

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward, self.noise_std))


    def inter_layer_env(self, lock):
        inter_layer_state = copy.deepcopy(self.inter_layer_state_init)
        users_number_covered_temp = self.users_number_covered
        have_dealt_task_num = 0
        inter_episode_total_time = 0
        total_profits = -math.inf
        inter_layer_end_number = 0
        while True:
            # 获取内层动作
            inter_layer_actions = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, inter_layer_state)]
            # 移动长度
            move_distance = 10
            # 无人机根据动作去改变自己的位置
            for i in range(len(self.uavs)):
                temp_uav = self.uavs[i]
                move_direction = 180 * inter_layer_actions[i] + 180
    #--------------------------下面是改动的地方-----------------------------
                move_direction = math.radians(move_direction)
                temp_uav.uav_move(move_direction, move_distance)

            # 约束任意两个无人机之间的最小距离
            # 约束无人机位置时无人机的最小移动距离
            uav_min_move_distance = 5
            for _ in range(5):
                for uav in self.uavs:
                    uav.position[0] += random.randint(-uav_min_move_distance, uav_min_move_distance)
                    uav.position[1] += random.randint(-uav_min_move_distance, uav_min_move_distance)
                myUtils.get_uav_min_distance_restraint(self.uavs)



#------------------------------------------下面是改动的部分-----------------------------------
            # 无人机去为车辆提供服务
            for user in self.users:
                # 在当前user下看选择哪个uav更时延更小，成本更小
                for task in user.tasks:
                    # 判断当前任务是否已完成
                    if task[-1]:
                        continue

                    # 初始化最小处理时延和对应的 UAV 编号
                    task_min_time = math.inf
                    uav_cope_task_index = -1

                    # 遍历所有 UAV，找到处理当前任务耗时最短的那个
                    for idx, uav in enumerate(self.uavs):
                        d_u_v, d_u_cloud = uav.get_uv_distance(user.position, myConfig.cloud_position)
                        trans_rate_user, trans_rate_cloud = uav.get_uav_transmission_rate(d_u_v, d_u_cloud)
                        temp_time = uav.get_uav_transmission_computing_time(task, trans_rate_user, trans_rate_cloud)

                        if temp_time < task_min_time:
                            task_min_time = temp_time
                            uav_cope_task_index = idx

                    # 将任务交给最优 UAV 处理
                    if uav_cope_task_index != -1:
                        uav = self.uavs[uav_cope_task_index]

                        # 如果 UAV 命中缓存并且任务尚未完成
                        if (task[0] in uav.cached_service_type or task[0] in uav.cached_content_type) and not task[-1]:
                            # 扣除资源
                            uav.storage_space -= task[0]

                            if task[0] in uav.cached_service_type:
                                uav.F_cpu -= task[2]
                                temp_profit = 1.0 * (task[3] / task_min_time) * (
                                        task[1] / uav.storage_space +
                                        task[2] / uav.F_cpu +
                                        (task[2] * uav.alpha / uav.F_cpu) / uav.remain_power +
                                        task[-2] / 15
                                )
                                uav.F_cpu += task[2]  # 释放资源
                            else:
                                temp_profit = 1.0 * (task[3] / task_min_time) * (
                                        task[1] / uav.storage_space +
                                        task[-2] / 25
                                )

                            # 更新收益和任务信息
                            uav.profits += temp_profit
                            uav.tasked_number += 1
                            task[-1] = True
                            have_dealt_task_num += 1

                            # 释放存储资源
                            uav.storage_space += task[0]

                            # 累计任务处理时延
                            uav.uav_cope_tasks_time += task_min_time
                            inter_episode_total_time += task_min_time
# ------------------------------------------上面是改动的部分-----------------------------------

                    # 记录每个无人机处理任务的时延
                    self.uavs[uav_cope_task_index].uav_cope_tasks_time += task_min_time
                    inter_episode_total_time += task_min_time
#--------------------------
            # 计算每个无人机的奖励
            inter_layer_reward = []
            # 计算总的收益
            inter_layer_total_profits = 0
            for uav in self.uavs:
                uav.uav_reward = (uav.profits - uav.base_cost)
                inter_layer_reward.append(uav.tasked_number)
                inter_layer_total_profits += uav.uav_reward
                if uav.tasked_number > uav.task_num_sill:
                    uav.base_cost += (uav.tasked_number - uav.task_num_sill) * 3
                    uav.task_num_sill += uav.tasked_number - uav.task_num_sill

            total_profits = max(inter_layer_total_profits, total_profits)
            outer_layer_reward = inter_layer_total_profits
            # 让车辆移动
            for user in self.users:
                user.user_move(8)
            # 计算内层的下一个状态
            inter_layer_state_, self.users_number_covered = myUtils.get_inter_layer_state(self.uavs, self.users)

            if lock:
                # Store the transition
                self.replay_buffer.store_transition(inter_layer_state, inter_layer_actions, inter_layer_reward,
                                                    inter_layer_state_, [True for _ in range(self.args.N)])
                inter_layer_state = inter_layer_state_
                self.total_steps += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min
                if self.replay_buffer.current_size > self.args.batch_size:
                    # 策略冻结：提前计算所有 UAV 的 actor 网络输出（动作）
                    with torch.no_grad():
                        static_actions = [
                            agent.actor_target(batch_obs)
                            for agent, batch_obs in zip(self.agent_n, self.replay_buffer.sample()[0])
                        ]

                    # 并行训练每个 UAV，使用静态动作作为其他 UAV 的输入
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train_parallel(
                            self.replay_buffer,
                            static_actions,  # 所有 UAV 的静态策略动作
                            agent_id
                        )

            if have_dealt_task_num >= self.total_task_num * 0.9 or inter_layer_end_number >= 8:
                if have_dealt_task_num >= self.total_task_num * 0.9:
                    print("处理任务数大于百分之八十五：", have_dealt_task_num)
                outer_layer_reward += 20
                break
            elif users_number_covered_temp < self.users_number_covered:
                users_number_covered_temp = self.users_number_covered
                outer_layer_reward += 15
            else:
                inter_layer_end_number += 1
                outer_layer_reward -= 10
        if lock:
            return have_dealt_task_num, total_profits, inter_episode_total_time, self.users_number_covered, outer_layer_reward
        else:
            return inter_layer_total_profits


