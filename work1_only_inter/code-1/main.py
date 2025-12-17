import random
import numpy as np
import myUtils
import copy
import UAV
import DDQN
import PySimpleGUI as sg
import vehicle
import math
import inter_layer_make_env
import args
import myConfig
import DDQN
import maddpg

ddqn = DDQN.DDQN(alpha=0.0005, state_dim=6, action_dim=50,
                 fc1_dim=128, fc2_dim=128, fc3_dim=64, ckpt_dir=1)
# exp_pool_size = 5000
# 任意两个无人机之间最小距离的约束
# d_uav_min = 500

def run():
    # # 初始化车辆用户
    # users_num = 100
    # # 每个用户产生的任务数
    # user_tasks_num = 10
    users_copy, type_copy = myUtils.get_init_users(myConfig.users_num, myConfig.user_tasks_num)
    # 总的任务数
    total_task_num = myConfig.users_num * myConfig.user_tasks_num
    # 初始化算法的一些参数
    MEMORY_CAPACITY = 2560
    # 迭代次数
    episode_num = myConfig.episode_num
    # 远程云的位置
    # cloud_position = [20000, 20000]
    # 记录总的收益列表
    profits_lists = []
    # 记录总的奖励列表
    rewards_lists = []
    # 记录总的无人机数量列表
    uav_numbers_lists = []
    # 记录损失列表
    losses_lists = []
    # args对象初始化
    args_copy = args.args()

    for episode in range(episode_num):
        sg.one_line_progress_meter("MADDPG8 multi-option", episode + 1, episode_num, orientation='h')
        print("episode counter : ", episode)

        # 外层的状态
        # [用户的剩余任务数，无人机的数量，使用无人机的成本，总的收益，总时延、被覆盖的用户数]
        outer_layer_state = [1, 0, 1, 0, 1, 0]
        episode_total_time = 0
        episode_total_profits = 0
        # 记录外层奖励，也就是内存奖励的最大值
        outer_layer_reward = -math.inf
        # 记录总收益的值
        # total_profits = -math.inf
        # 外层结束的标志
        outer_over_symbol = -math.inf
        # 记录收益变化次数
        profits_change_num = 0
        # 记录每次迭代总的收益列表
        profits_list = []
        # 记录每次迭代总的奖励列表
        rewards_list = []
        # 记录每次迭代总的无人机数量列表
        uav_numbers_list = []
        # 记录每次迭代损失列表
        losses_list = []

        while True:
            # 外层动作的选取
            outer_layer_action = ddqn.choose_action(outer_layer_state)
            # TODO
            # if outer_layer_action < 2:
            #     continue
            uavs = myUtils.get_init_uav(outer_layer_action, type_copy)
            # print("uavs: ", uavs)
            # 无人机与车辆用户之间的距离矩阵
            # 记录使用无人机的总成本
            total_uav_base_cost = 0
            for uav in uavs:
                total_uav_base_cost += uav.base_cost
            # 这里还需要对车辆用户任务进行初始化
            users = copy.deepcopy(users_copy)
            # 内层的状态初始化
            inter_layer_state, users_number_covered = myUtils.get_inter_layer_state(uavs, users)
            print("users_number_covered: ", users_number_covered)
            users_number_covered_temp = users_number_covered
            # for uav in uavs:
            #     uav_state = []
            #     for index in range(5):  # uav_state的维度
            #         # 目前还没有对状态进行归一化，之后需要进行归一化
            #         uav_user_num = myUtils.get_uav_communication_range_user_number(uav, users)
            #         uav_state.append(uav_user_num)
            #         uav_state.append(uav.position[0])
            #         uav_state.append(uav.position[1])
            #         uav_state.append(uav.base_cost)
            #         uav_state.append(uav.profits)
            #         uav_state.append(uav.tasked_number)
            #
            #     inter_layer_state.append(uav_state)

            # # 记录内层的总收益
            # inter_episode_total_profits = 0
            # # 记录内存处理任务的总时延
            # inter_episode_total_time = 0
            # # 记录已经处理完的任务数
            # have_dealt_task_num = 0
            # # 记录当前内层的总收益
            # inter_total_profits = -math.inf
            # # 保存每个无人机的奖励
            # every_uav_rewards = [0 for i in range(outer_layer_action)]
            # # 保存每个无人机的收益
            # every_uav_profits = [0 for i in range(outer_layer_action)]cx
            # # 保存每个无人机的处理时间
            # every_uav_times = [0 for i in range(outer_layer_action)]
            # # 保存每个无人机的基本成本
            # every_uav_base_cost = [0 for i in range(outer_layer_action)]
            # # 内层循环结束次数
            # inter_layer_end_number = 0
            have_dealt_task_num = 0
            total_profits = -100
            inter_episode_total_time = 0
            outer_layer_reward = -100
            if outer_layer_action > 2:
                # 内层环境初始化
                inter_env = inter_layer_make_env.Runner(args_copy, 1, uavs, users)
                # 内层环境迭代过程
                have_dealt_task_num, total_profits, inter_episode_total_time, users_number_covered, outer_layer_reward = inter_env.run()

            # 计算外层的奖励函数，outer_layer_reward
            # outer_layer_reward = (1 * outer_layer_reward) / (total_uav_base_cost * 0.1) + users_number_covered
            # outer_layer_reward = (1 * outer_layer_reward) * users_number_covered / total_uav_base_cost
            # outer_layer_reward = (1 * outer_layer_reward + users_number_covered) / (total_uav_base_cost * 0.1)
            # outer_layer_reward = (1 * outer_layer_reward) + users_number_covered
            # outer_layer_reward += have_dealt_task_num
            # print("outer_layer_reward->>>: ", outer_layer_reward)
            # 计算外层的下一个状态
            outer_layer_state_ = [(total_task_num - have_dealt_task_num), outer_layer_action, total_uav_base_cost, total_profits, inter_episode_total_time, users_number_covered]
            print("outer_layer_state_: ", outer_layer_state_)
            # DDQN存储经验池
            ddqn.remember(outer_layer_state, outer_layer_action, outer_layer_reward, outer_layer_state_, True)
            # 如果经验池存满了就开始学习
            outer_layer_loss = ddqn.learn()
            # 外层状态的转变，outer_layer_state = outer_layer_state_
            outer_layer_state = outer_layer_state_
            # 外层结束的标志
            # outer_over_symbol =
            # 外层循环的结束条件
            # 当当前总成本（收益-无人机成本）在5次内都没有改变时就结束当前循环
            # if total_profits < inter_layer_total_profits:
            #     total_profits = inter_layer_total_profits
            #     profits_change_num = 0
            #     outer_layer_reward += 2
            if outer_over_symbol < total_profits:
                outer_over_symbol = total_profits
                profits_change_num = 0
                outer_layer_reward += 10
                # 记录当前迭代里面的数据
                # profits_list.append(total_profits)
                # rewards_list.append(outer_layer_reward)
                # uav_numbers_list.append(outer_layer_action)
                # losses_list.append(outer_layer_loss)
            else:
                profits_change_num += 1
                outer_layer_reward -= 5
            print("outer_layer_reward: ==> ", outer_layer_reward)
            # print("profits_change_num: ", profits_change_num)
            if profits_change_num >= 6:
                # 记录当前迭代里面的数据
                # profits_list.append(total_profits)
                # rewards_list.append(outer_layer_reward)
                # uav_numbers_list.append(outer_layer_action)
                # losses_list.append(outer_layer_loss)
                break

            # 记录当前迭代里面的数据
            profits_list.append(total_profits)
            rewards_list.append(outer_layer_reward)
            uav_numbers_list.append(outer_layer_action)
            losses_list.append(outer_layer_loss)

        # 无人机的数量
        # print("uav_numbers: ", math.ceil(sum(uav_numbers_list) / len(uav_numbers_list)))
        if len(profits_list) > 0:
            profits_lists.append(sum(profits_list) / len(profits_list))
            rewards_lists.append(sum(rewards_list) / len(rewards_list))
            # uav_numbers_lists.append(math.ceil(sum(uav_numbers_list) / len(uav_numbers_list)))
            uav_numbers_lists.append(sum(uav_numbers_list) / len(uav_numbers_list))
            losses_lists.append(sum(losses_list) / len(losses_list))

    # 每200次迭代求一次均值！！！！！！！改变奖励惩罚设置！！！
    profits_lists_mean = []
    rewards_lists_mean = []
    uav_numbers_lists_mean = []
    losses_mean = []
    nums = 10
    for i in range(episode_num - nums):
        profits_lists_mean.append(np.mean(profits_lists[i:nums + i]))
        rewards_lists_mean.append(np.mean(rewards_lists[i:nums + i]))
        uav_numbers_lists_mean.append(np.mean(uav_numbers_lists[i:nums + i]))

    # 保存每次迭代后所获取的数据
    myUtils.get_data_save(profits_lists, rewards_lists, uav_numbers_lists, losses_lists)
    myUtils.data_show(profits_lists_mean, rewards_lists_mean, uav_numbers_lists_mean, losses_lists)


if __name__ == "__main__":
    run()







