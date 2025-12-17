import random
import numpy as np
import myUtils
import copy
import UAV
import inter_layer_make_env
import args
import myConfig


def run():
    # 固定无人机数量
    N_uav = 10  # 指定无人机数量
    users_copy, type_copy = myUtils.get_init_users(myConfig.users_num, myConfig.user_tasks_num)
    total_task_num = myConfig.users_num * myConfig.user_tasks_num

    # 初始化MADDPG参数
    args_copy = args.args()
    args_copy.N = N_uav  # 设置智能体数量
    episode_num = myConfig.episode_num

    # 记录性能指标
    profits_lists = []
    rewards_lists = []
    losses_lists = []

    for episode in range(episode_num):
        print("Episode:", episode)

        # 初始化无人机和用户（每个episode重新初始化）
        uavs = myUtils.get_init_uav(N_uav, type_copy)
        users = copy.deepcopy(users_copy)

        # 初始化内层Runner
        runner = inter_layer_make_env.Runner(args_copy, 1, uavs, users)

        # 运行内层环境
        have_dealt_task_num, total_profits, inter_episode_total_time, users_number_covered, outer_layer_reward = runner.run(
            lock=True)

        # 记录性能指标
        profits_lists.append(total_profits)
        rewards_lists.append(outer_layer_reward)
        # 如果需要记录损失，可以在runner.run()返回损失值

        print(
            f"Episode {episode}: Tasks done {have_dealt_task_num}/{total_task_num}, Profit: {total_profits}, Reward: {outer_layer_reward}")

    # 保存和显示结果
    myUtils.data_show(profits_lists, rewards_lists, [N_uav] * len(profits_lists), losses_lists)


if __name__ == "__main__":
    run()