# main_sweep_dual_reward.py
import numpy as np
import copy
import pandas as pd
import time  # 计时总训练耗时
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

    for UAV_NUM in range(30, 31):  # 遍历 1~50 架无人机
        print(f"\n===== 开始训练 UAV_NUM = {UAV_NUM} =====")

        # --- 该 UAV_NUM 的滚动统计 ---
        profits_lists = []
        rewards_norm_lists = []
        rewards_raw_lists = []

        # --- 该 UAV_NUM 的总耗时（wall-clock） ---
        uav_wall_start = time.perf_counter()

        # --- 该 UAV_NUM 的分阶段累计（总秒数）与总步数 ---
        total_steps_accum = 0
        phase_sums = {
            "action_select_s": 0.0,
            "move_constraint_s": 0.0,
            "task_assign_s": 0.0,
            "reward_calc_s": 0.0,
            "buffer_store_s": 0.0,
            "train_step_s": 0.0,
        }

        args_copy = args.args()
        writer = SummaryWriter(log_dir=f"./logs_sweep_dual_reward/uav_{UAV_NUM}")

        # 初始化 Runner（经验池和智能体跨集共享）
        uavs = myUtils.get_init_uav(UAV_NUM, task_types)
        users = copy.deepcopy(users_copy)
        inter_env = inter_layer_make_env.Runner(args_copy, 1, uavs, users)

        for episode in range(episode_num):
            if episode > 0:
                # 重置环境状态
                uavs = myUtils.get_init_uav(UAV_NUM, task_types)
                users = copy.deepcopy(users_copy)
                inter_env.uavs = uavs
                inter_env.users = users
                inter_env.inter_layer_state_init, inter_env.users_number_covered = myUtils.get_inter_layer_state(
                    inter_env.uavs, inter_env.users
                )
                inter_env.last_global_reward = 0.0
                inter_env._buf_reward_raw = []
                inter_env._buf_reward_norm = []
                inter_env._buf_profit = []

            if episode < warmup_episodes:
                (have_dealt_task_num, total_profits, inter_episode_total_time, users_covered,
                 outer_layer_reward_norm, outer_layer_reward_raw, step_cnt, timing_sums) = inter_env.run(lock=False)
            else:
                (have_dealt_task_num, total_profits, inter_episode_total_time, users_covered,
                 outer_layer_reward_norm, outer_layer_reward_raw, step_cnt, timing_sums) = inter_env.run(lock=True)

            # 保存 reward/profit
            profits_lists.append(total_profits)
            rewards_norm_lists.append(outer_layer_reward_norm)
            rewards_raw_lists.append(outer_layer_reward_raw)

            # TensorBoard 每集日志
            writer.add_scalar("Episode/Profit", total_profits, episode)
            writer.add_scalar("Episode/Reward_Norm", outer_layer_reward_norm, episode)
            writer.add_scalar("Episode/Reward_Raw", outer_layer_reward_raw, episode)

            # 汇总该集的计时信息
            total_steps_accum += int(step_cnt)
            for k in phase_sums.keys():
                phase_sums[k] += float(timing_sums.get(k, 0.0))

            # 打印滚动均值（最近10集）
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

            # 【新增】打印当集六阶段“每步平均耗时”（ms）
            ep_steps = max(1, int(step_cnt))
            ep_action_ms = 1000.0 * timing_sums.get("action_select_s", 0.0) / ep_steps
            ep_move_ms   = 1000.0 * timing_sums.get("move_constraint_s", 0.0) / ep_steps
            ep_assign_ms = 1000.0 * timing_sums.get("task_assign_s", 0.0) / ep_steps
            ep_reward_ms = 1000.0 * timing_sums.get("reward_calc_s", 0.0) / ep_steps
            ep_store_ms  = 1000.0 * timing_sums.get("buffer_store_s", 0.0) / ep_steps
            ep_train_ms  = 1000.0 * timing_sums.get("train_step_s", 0.0) / ep_steps
            print(
                f"    PhaseAvg(ms, ep) -> act:{ep_action_ms:.2f} | move:{ep_move_ms:.2f} | "
                f"assign:{ep_assign_ms:.2f} | reward:{ep_reward_ms:.2f} | store:{ep_store_ms:.2f} | train:{ep_train_ms:.2f}"
            )

        writer.close()

        # 取最后 10 集的平均值（与原逻辑一致）
        avg_profit = np.mean(profits_lists[-10:]) if len(profits_lists) >= 10 else np.mean(profits_lists)
        avg_reward_norm = np.mean(rewards_norm_lists[-10:]) if len(rewards_norm_lists) >= 10 else np.mean(rewards_norm_lists)
        avg_reward_raw = np.mean(rewards_raw_lists[-10:]) if len(rewards_raw_lists) >= 10 else np.mean(rewards_raw_lists)

        # 该 UAV_NUM 的总训练耗时（wall-clock）
        uav_total_wall_s = time.perf_counter() - uav_wall_start

        # 分阶段“每步平均耗时”（毫秒）
        denom_steps = max(1, total_steps_accum)  # 防止除0
        avg_action_ms = 1000.0 * phase_sums["action_select_s"]   / denom_steps
        avg_move_ms   = 1000.0 * phase_sums["move_constraint_s"] / denom_steps
        avg_assign_ms = 1000.0 * phase_sums["task_assign_s"]     / denom_steps
        avg_reward_ms = 1000.0 * phase_sums["reward_calc_s"]     / denom_steps
        avg_store_ms  = 1000.0 * phase_sums["buffer_store_s"]    / denom_steps
        avg_train_ms  = 1000.0 * phase_sums["train_step_s"]      / denom_steps

        results.append({
            "UAV_NUM": UAV_NUM,
            "Avg_Profit": avg_profit,
            "Avg_Reward_Norm": avg_reward_norm,
            "Avg_Reward_Raw": avg_reward_raw,

            # 总训练耗时（秒）
            "Total_Train_Time_s": uav_total_wall_s,

            # 六阶段每步平均耗时（毫秒）
            "Avg_ActionSelect_ms":    avg_action_ms,
            "Avg_MoveConstraint_ms":  avg_move_ms,
            "Avg_TaskAssign_ms":      avg_assign_ms,
            "Avg_RewardCalc_ms":      avg_reward_ms,
            "Avg_BufferStore_ms":     avg_store_ms,
            "Avg_TrainStep_ms":       avg_train_ms,

            # 也可留一个总步数列以便复核
            "Total_Steps": total_steps_accum,
        })

        # 【新增】将六阶段平均耗时追加到汇总 debug 行
        print(
            f"===== UAV_NUM={UAV_NUM} Avg Profit: {avg_profit:.4f} | "
            f"Avg Reward Raw: {avg_reward_raw:.4f} | "
            f"Avg Reward Norm: {avg_reward_norm:.4f} | "
            f"Total Train Time: {uav_total_wall_s:.2f}s | "
            f"Avg(ms) [act:{avg_action_ms:.2f}, move:{avg_move_ms:.2f}, assign:{avg_assign_ms:.2f}, "
            f"reward:{avg_reward_ms:.2f}, store:{avg_store_ms:.2f}, train:{avg_train_ms:.2f}] ====="
        )

    # 保存总结果
    df = pd.DataFrame(results)
    df.to_csv("uav_num_sweep_dual_reward.csv", index=False)
    print("结果已保存到 uav_num_sweep_dual_reward.csv")

if __name__ == "__main__":
    run_sweep()
