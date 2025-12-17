import numpy as np
import copy
import PySimpleGUI as sg
from torch.utils.tensorboard import SummaryWriter

import myUtils
import inter_layer_make_env
import args
import myConfig

def run():
    users_copy, task_types = myUtils.get_init_users(myConfig.users_num, myConfig.user_tasks_num)
    episode_num = myConfig.episode_num

    UAV_NUM = 10
    warmup_episodes = 10

    profits_lists = []
    rewards_lists = []
    uav_numbers_lists = []
    losses_lists = []

    args_copy = args.args()

    # TensorBoard writer
    writer = SummaryWriter(log_dir="./logs_maddpg_full")

    for episode in range(episode_num):
        sg.one_line_progress_meter("MADDPG Training", episode + 1, episode_num, orientation='h')
        print("Episode:", episode)

        uavs = myUtils.get_init_uav(UAV_NUM, task_types)
        users = copy.deepcopy(users_copy)

        inter_env = inter_layer_make_env.Runner(args_copy, 1, uavs, users, tb_logdir="./logs_step")

        if episode < warmup_episodes:
            have_dealt_task_num, total_profits, inter_episode_total_time, users_covered, outer_layer_reward = inter_env.run(lock=False)
        else:
            have_dealt_task_num, total_profits, inter_episode_total_time, users_covered, outer_layer_reward = inter_env.run(lock=True)

        profits_lists.append(total_profits)
        rewards_lists.append(outer_layer_reward)
        uav_numbers_lists.append(UAV_NUM)
        losses_lists.append(0.0)

        # 写入 TensorBoard（按 episode）
        writer.add_scalar("Episode/Profit", total_profits, episode)
        writer.add_scalar("Episode/Reward", outer_layer_reward, episode)
        writer.add_scalar("Episode/UAV_num", UAV_NUM, episode)

    writer.close()

    # 滑动均值
    window = 10
    profits_lists_mean = []
    rewards_lists_mean = []
    uav_numbers_lists_mean = []
    for i in range(episode_num - window):
        profits_lists_mean.append(np.mean(profits_lists[i:i+window]))
        rewards_lists_mean.append(np.mean(rewards_lists[i:i+window]))
        uav_numbers_lists_mean.append(np.mean(uav_numbers_lists[i:i+window]))

    myUtils.get_data_save(profits_lists, rewards_lists, uav_numbers_lists, losses_lists)
    myUtils.data_show(profits_lists_mean, rewards_lists_mean, uav_numbers_lists_mean, losses_lists)

if __name__ == "__main__":
    run()
