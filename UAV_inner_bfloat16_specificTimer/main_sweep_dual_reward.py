import time
import numpy as np
import copy
import pandas as pd
# import PySimpleGUI as sg
from torch.utils.tensorboard import SummaryWriter

import myUtils
import inter_layer_make_env
import args

def run_sweep():
    users_copy, task_types = myUtils.get_init_users(inter_layer_make_env.users_num, inter_layer_make_env.user_tasks_num)
    episode_num = 50         # 每个 UAV_NUM 的训练集数
    warmup_episodes = 5      # warmup 集数

    results = []

    for UAV_NUM in range(1, 51):  # 遍历 1~50 架无人机
        print(f"\n===== 开始训练 UAV_NUM = {UAV_NUM} =====")

        # 用于评价指标（与原来一致）
        profits_lists = []
        rewards_norm_lists = []
        rewards_raw_lists = []

        # 新增：用于累计六个关键阶段的时间（按 episode 累加，之后取平均）
        stage_sums = {
            "action": 0.0,
            "move": 0.0,
            "assign": 0.0,
            "reward": 0.0,
            "store": 0.0,
            "train": 0.0
        }

        args_copy = args.args()
        writer = SummaryWriter(log_dir=f"./logs_sweep_dual_reward/uav_{UAV_NUM}")

        # 初始化 Runner（经验池和智能体跨集共享）
        uavs = myUtils.get_init_uav(UAV_NUM, task_types)
        users = copy.deepcopy(users_copy)
        inter_env = inter_layer_make_env.Runner(args_copy, 1, uavs, users)

        # 统计当前 UAV_NUM 的总训练用时（墙钟时间）
        t_uav_begin = time.perf_counter()

        for episode in range(episode_num):
            # sg.one_line_progress_meter(
            #     f"MADDPG Training UAV_NUM={UAV_NUM}",
            #     episode + 1,
            #     episode_num,
            #     orientation="h"
            # )

            if episode > 0:
                # 重置环境状态
                uavs = myUtils.get_init_uav(UAV_NUM, task_types)
                users = copy.deepcopy(users_copy)
                inter_env.uavs = uavs
                inter_env.users = users
                inter_env.inter_layer_state_init, inter_env.users_number_covered = myUtils.get_inter_layer_state(inter_env.uavs, inter_env.users)
                inter_env.last_global_reward = 0.0
                inter_env._buf_reward_raw = []
                inter_env._buf_reward_norm = []
                inter_env._buf_profit = []

            if episode < warmup_episodes:
                (have_dealt_task_num, total_profits, inter_episode_total_time, users_covered,
                 outer_layer_reward_norm, outer_layer_reward_raw, timings) = inter_env.run(lock=False)
            else:
                (have_dealt_task_num, total_profits, inter_episode_total_time, users_covered,
                 outer_layer_reward_norm, outer_layer_reward_raw, timings) = inter_env.run(lock=True)

            # 保存结果（与原来一致）
            profits_lists.append(total_profits)
            rewards_norm_lists.append(outer_layer_reward_norm)
            rewards_raw_lists.append(outer_layer_reward_raw)

            # 累加本 episode 的六个阶段时间
            stage_sums["action"] += timings["action"]
            stage_sums["move"]   += timings["move"]
            stage_sums["assign"] += timings["assign"]
            stage_sums["reward"] += timings["reward"]
            stage_sums["store"]  += timings["store"]
            stage_sums["train"]  += timings["train"]

            # TensorBoard 每集日志
            writer.add_scalar("Episode/Profit", total_profits, episode)
            writer.add_scalar("Episode/Reward_Norm", outer_layer_reward_norm, episode)
            writer.add_scalar("Episode/Reward_Raw", outer_layer_reward_raw, episode)

            # Print rolling averages (uses up to the last 10 episodes)
            window_size = min(len(profits_lists), 10)
            avg_profit_window = np.mean(profits_lists[-window_size:])
            avg_reward_norm_window = np.mean(rewards_norm_lists[-window_size:])
            avg_reward_raw_window = np.mean(rewards_raw_lists[-window_size:])
            print(
                f"UAV_NUM={UAV_NUM} Episode {episode + 1}/{episode_num} | "
                f"Average Profit(last {window_size}): {avg_profit_window:.4f} | "
                f"Average Reward Raw(last {window_size}): {avg_reward_raw_window:.4f} | "
                f"Average Reward Norm(last {window_size}): {avg_reward_norm_window:.4f}"
            )

        # 当前 UAV_NUM 的总训练时长（秒）
        total_training_time_s = time.perf_counter() - t_uav_begin
        writer.close()

        # 取最后 10 集的平均值（与原来一致）
        avg_profit = np.mean(profits_lists[-10:]) if len(profits_lists) >= 10 else np.mean(profits_lists)
        avg_reward_norm = np.mean(rewards_norm_lists[-10:]) if len(rewards_norm_lists) >= 10 else np.mean(rewards_norm_lists)
        avg_reward_raw = np.mean(rewards_raw_lists[-10:]) if len(rewards_raw_lists) >= 10 else np.mean(rewards_raw_lists)

        # 计算六个阶段的“平均每集耗时”（秒）
        avg_action_s = stage_sums["action"] / episode_num
        avg_move_s   = stage_sums["move"]   / episode_num
        avg_assign_s = stage_sums["assign"] / episode_num
        avg_reward_s = stage_sums["reward"] / episode_num
        avg_store_s  = stage_sums["store"]  / episode_num
        avg_train_s  = stage_sums["train"]  / episode_num

        # 收集 CSV 行
        results.append({
            "UAV_NUM": UAV_NUM,
            "Avg_Profit": avg_profit,
            "Avg_Reward_Norm": avg_reward_norm,
            "Avg_Reward_Raw": avg_reward_raw,
            "Training_Total_Time_s": total_training_time_s,
            "Avg_Action_s": avg_action_s,
            "Avg_Move_s": avg_move_s,
            "Avg_Assign_s": avg_assign_s,
            "Avg_RewardCalc_s": avg_reward_s,
            "Avg_Store_s": avg_store_s,
            "Avg_Train_s": avg_train_s
        })

        print(
            f"===== UAV_NUM={UAV_NUM} "
            f"Avg Profit: {avg_profit:.4f} | "
            f"Avg Reward Raw: {avg_reward_raw:.4f} | "
            f"Avg Reward Norm: {avg_reward_norm:.4f} | "
            f"Total Train Time: {total_training_time_s:.2f}s | "
            f"Avg Stage (s) -> "
            f"Action {avg_action_s:.4f}, Move {avg_move_s:.4f}, Assign {avg_assign_s:.4f}, "
            f"Reward {avg_reward_s:.4f}, Store {avg_store_s:.4f}, Train {avg_train_s:.4f} ====="
        )

    # 保存总结果（加入了新的时间列）
    df = pd.DataFrame(results, columns=[
        "UAV_NUM",
        "Avg_Profit", "Avg_Reward_Norm", "Avg_Reward_Raw",
        "Training_Total_Time_s",
        "Avg_Action_s", "Avg_Move_s", "Avg_Assign_s", "Avg_RewardCalc_s", "Avg_Store_s", "Avg_Train_s"
    ])
    df.to_csv("uav_num_sweep_dual_reward.csv", index=False)
    print("结果已保存到 uav_num_sweep_dual_reward.csv")

if __name__ == "__main__":
    run_sweep()
