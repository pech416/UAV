import numpy as np
import copy
import PySimpleGUI as sg
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import myUtils
import inter_layer_make_env
import args
import myConfig

def run_sweep():
    users_copy, task_types = myUtils.get_init_users(myConfig.users_num, myConfig.user_tasks_num)
    episode_num = 50       # 每个 UAV_NUM 训练的 episode 数
    warmup_episodes = 5    # warm-up

    results = []  # 保存每个 UAV_NUM 的结果

    for UAV_NUM in range(1, 51):  # 遍历 UAV 数量 1~50
        print(f"\n===== 开始训练 UAV_NUM = {UAV_NUM} =====")

        profits_lists = []
        rewards_lists = []

        args_copy = args.args()
        writer = SummaryWriter(log_dir=f"./logs_sweep/uav_{UAV_NUM}")

        for episode in range(episode_num):
            sg.one_line_progress_meter("MADDPG Training", episode + 1, episode_num, orientation='h')
            uavs = myUtils.get_init_uav(UAV_NUM, task_types)
            users = copy.deepcopy(users_copy)

            inter_env = inter_layer_make_env.Runner(args_copy, 1, uavs, users)

            if episode < warmup_episodes:
                have_dealt_task_num, total_profits, inter_episode_total_time, users_covered, outer_layer_reward = inter_env.run(lock=False)
            else:
                have_dealt_task_num, total_profits, inter_episode_total_time, users_covered, outer_layer_reward = inter_env.run(lock=True)

            profits_lists.append(total_profits)
            rewards_lists.append(outer_layer_reward)

            # TensorBoard 日志
            writer.add_scalar("Episode/Profit", total_profits, episode)
            writer.add_scalar("Episode/Reward", outer_layer_reward, episode)

        writer.close()

        # 计算均值（取最后 10 集的平均，避免前期波动）
        avg_profits = np.mean(profits_lists[-10:])
        avg_reward = np.mean(rewards_lists[-10:])

        results.append({"UAV_NUM": UAV_NUM, "Avg_Profit": avg_profits, "Avg_Reward": avg_reward})

    # 保存到 Excel/CSV
    df = pd.DataFrame(results)
    df.to_csv("uav_num_sweep.csv", index=False)
    print("结果已保存到 uav_num_sweep.csv")

if __name__ == "__main__":
    run_sweep()
