import numpy as np
import copy
import pandas as pd
import PySimpleGUI as sg
from torch.utils.tensorboard import SummaryWriter

import myUtils
import inter_layer_make_env
import args
import myConfig


def run_sweep():
    users_copy, task_types = myUtils.get_init_users(myConfig.users_num, myConfig.user_tasks_num)
    episode_num = 50         # 每个 UAV_NUM 的训练集数
    warmup_episodes = 5      # warmup 集数

    results = []

    for UAV_NUM in range(1, 51):  # 遍历 1~50 架无人机
        print(f"\n===== 开始训练 UAV_NUM = {UAV_NUM} =====")

        profits_lists = []
        rewards_norm_lists = []
        rewards_raw_lists = []

        args_copy = args.args()
        writer = SummaryWriter(log_dir=f"./logs_sweep_dual_reward/uav_{UAV_NUM}")

        for episode in range(episode_num):
            sg.one_line_progress_meter(
                f"MADDPG Training UAV_NUM={UAV_NUM}",
                episode + 1,
                episode_num,
                orientation="h"
            )

            uavs = myUtils.get_init_uav(UAV_NUM, task_types)
            users = copy.deepcopy(users_copy)
            inter_env = inter_layer_make_env.Runner(args_copy, 1, uavs, users)

            if episode < warmup_episodes:
                have_dealt_task_num, total_profits, inter_episode_total_time, users_covered, \
                outer_layer_reward_norm, outer_layer_reward_raw = inter_env.run(lock=False)
            else:
                have_dealt_task_num, total_profits, inter_episode_total_time, users_covered, \
                outer_layer_reward_norm, outer_layer_reward_raw = inter_env.run(lock=True)

            # 保存结果
            profits_lists.append(total_profits)
            rewards_norm_lists.append(outer_layer_reward_norm)
            rewards_raw_lists.append(outer_layer_reward_raw)

            # TensorBoard 每集日志
            writer.add_scalar("Episode/Profit", total_profits, episode)
            writer.add_scalar("Episode/Reward_Norm", outer_layer_reward_norm, episode)
            writer.add_scalar("Episode/Reward_Raw", outer_layer_reward_raw, episode)

        writer.close()

        # 取最后 10 集的平均值
        avg_profit = np.mean(profits_lists[-10:])
        avg_reward_norm = np.mean(rewards_norm_lists[-10:])
        avg_reward_raw = np.mean(rewards_raw_lists[-10:])

        results.append({
            "UAV_NUM": UAV_NUM,
            "Avg_Profit": avg_profit,
            "Avg_Reward_Norm": avg_reward_norm,
            "Avg_Reward_Raw": avg_reward_raw
        })

    # 保存总结果
    df = pd.DataFrame(results)
    df.to_csv("uav_num_sweep_dual_reward.csv", index=False)
    print("结果已保存到 uav_num_sweep_dual_reward.csv")


if __name__ == "__main__":
    run_sweep()
