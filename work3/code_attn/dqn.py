import copy
# import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import envs
import dqn_env
import offload
import dqn_agent


def train():
    n_episode = 2000

    # 初始化
    service_num = 60
    num_uav = 7
    fixed_wing = envs.FixedWing()
    env = dqn_env.DQNEnv(num_uav, service_num, fixed_wing)
    attention_mechanism = offload.AttentionMechanism()

    n_state = env.n_state
    n_action = env.n_action

    # IQL
    agents = [dqn_agent.Agent(idx=i, n_input=n_state, n_output=n_action, action_space=env.n_action, learning_rate=1e-4) for i in range(num_uav)]

    # # 加载模型参数
    # for idx, agent in enumerate(agents):
    #     agent.load_model(f"agent{idx}_dqn_model.pth")

    # 在训练循环开始前初始化空列表
    episode_rewards = []
    chains_delay = []
    losses = []
    episode_actions = []
    alphas = []
    betas = []
    alpha2_ps = []  # todo

    # 一条任务链
    for episode in range(n_episode):
        # sg.one_line_progress_meter("dqn multi-option", episode + 1, n_episode, orientation='h')
        # print(f"episode:{episode}")

        state = env.reset()
        done = False
        episode_reward = 0
        chain_delay = 0
        alpha2_count = 0  # todo

        # 一个任务
        for task_index in range(len(env.task_chain.tasks)):  # for循环中的task_index会自动更新
            task = env.task_chain.tasks[task_index]
            task_redo = False  # 任务重做标志

            # 存储所有uav的……
            states = []
            actions = []
            next_states = []

            # uav独立决策（根据自己的观察独立选择动作）
            for uav_index, current_uav in enumerate(env.uav_group):

                # ------------------1. uav缓存与电量------------------
                task.initial_battery[uav_index] = current_uav.battery
                task.initial_cache[uav_index] = copy.deepcopy(current_uav.uav_services)

                # ------------------2. 缓存决策------------------
                state = env.state_init()
                action = agents[uav_index].choose_action(state)

                # ------------------3.缓存决策后更新uav的state------------------
                env.cache_update(action, current_uav)
                env.current_uav_index = (env.current_uav_index + 1) % env.num_uav
                if env.task_chain.current_task_index < len(env.task_chain.tasks) - 1:
                    env.task_chain.current_task_index += 1
                    next_state = env.state_init()
                    env.task_chain.current_task_index -= 1
                else:  # 任务链已完成
                    next_state = env.reset()

                # ------------------4.存储经验------------------
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                episode_actions.append(actions)

            # 联合奖励
            # 1).卸载决策
            A_uavs, A_scores = attention_mechanism.process_single_task(task, env.uav_group)  # 选出注意力小组和注意力分数
            task.compare_time_delay(fixed_wing, task.user)  # 比较固定翼与用户的时延
            attention_mechanism.where_offload_single_task(task, A_uavs, A_scores)  # 得出卸载α，任务拆分β，任务拆分权重weight

            reward, delay = env.get_reward(task, A_uavs, fixed_wing)
            if task.alpha == 2:
                reward *= 5
                delay /= 5
                alpha2_count += 1  # todo

            # 2).卸载决策后更新状态
            if task.alpha == 2:
                weight_index = 0
                for uav in A_uavs:
                    average_power = 50
                    task_duration = task.compute / uav.compute
                    unload_battery = average_power * task_duration * task.weight[weight_index]
                    uav.battery -= unload_battery
                    weight_index += 1

                    # 更新service的访问次数
                    for service in uav.uav_services:
                        if service.service_id == task.service.service_id:
                            service.access += 1

                    # 电量耗尽
                    if uav.battery <= 0:
                        reward -= 5
                        task_redo = True

            # 3).标记该任务为已完成
            task.task_done = True
            episode_reward += reward
            chain_delay += delay

            if task.task_done:
                if env.task_chain.current_task_index < len(env.task_chain.tasks) - 1:  # 任务链未完成
                    env.task_chain.current_task_index += 1
                    env.state_init()
                else:
                    env.reset()

            # 独立学习（每个uav根据自身的经验更新自己的策略）
            done = task.task_done and env.task_chain.current_task_index == len(env.task_chain.tasks) - 1 # chain完成：当前task完成，且当前task为最后一个task。
            for i, agent in enumerate(agents):
                # 存储暂态
                agent.store_transition(states[i], actions[i], reward, next_states[i], done)
                # # 开始学习
                if agent.memory_count > agent.MEMORY_SIZE:
                    loss = agent.learn()
                    losses.append(loss)

                # 保存模型
                file_name = f"agent{i}_dqn_model.pth"
                agent.save_model(file_name)

            # 当任务需要重做
            if task_redo:

                task.task_done = False  # 重置task的状态

                for uav_index, uav in enumerate(env.uav_group):
                    uav.battery = task.initial_battery[uav_index]
                    uav.uav_services = copy.deepcopy(task.initial_cache[uav_index])

            if done:
                env.task_chain.current_task_index = 0
                for task in env.task_chain.tasks:
                    task.task_done = False

            alphas.append(task.alpha)
            betas.append(task.beta)

        # 存储迭代的结果
        episode_rewards.append(episode_reward)
        chains_delay.append(chain_delay)
        print(f"Episode {episode}: Total reward = {episode_reward}")

        alpha2_p = alpha2_count / len(env.task_chain.tasks)  # 计算概率
        alpha2_ps.append(alpha2_p)  # 添加到列表

    # 平滑处理
    window_size = 140
    moving_rewards = []
    for i in range(n_episode - window_size):
        moving_rewards.append((np.mean(episode_rewards[i:window_size + i])))

    # # 选择间隔的数据点进行绘制
    # interval = 20
    # selected_episodes = episode_rewards[window_size-1:][::interval]
    # selected_rewards = moving_rewards[::interval]

    return episode_rewards, moving_rewards, losses, alphas, betas, chains_delay, alpha2_ps


def plot_performance(ys, xs, title, xlabel, ylabel, filename, grid=True, colors=None):

    if len(xs) == 1:
        mycolor = colors[0] if colors else None
        plt.plot(ys, color=mycolor)
    else:
        for i, x in enumerate(xs):
            y = np.array(ys[i])
            mycolors = colors[i] if colors else None
            plt.plot(y, label=f'LR:{x}', color=mycolors)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.legend()  # 创建图例

    if grid:
        plt.grid(True)

    # 保存显示
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.show()


def save_data(data, file_name, column_name):
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        df = pd.DataFrame()

    df[column_name] = pd.Series(data)

    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    # 训练
    episode_rewards, moving_rewards, losses, alphas, betas, chains_delay, alpha2_ps = train()

    # 保存数据
    # 保存alpha=2的概率数据
    os.makedirs('./data', exist_ok=True)

    save_data(alpha2_ps, './data/AMADQN.csv', "Cache")
    save_data(chains_delay, './data/AMADQN.csv', "chains_delay")
    save_data(episode_rewards, './data/AMADQN.csv', "episode_rewards")
    save_data(moving_rewards, './data/AMADQN.csv', "moving_reward")
    save_data(losses, './data/AMADQN.csv', "loss")
    save_data(alphas, './data/AMADQN.csv', "alphas")
    save_data(betas, './data/AMADQN.csv', "betas")

    # 画图
    lrs = [1e-4]
    mycolors = [(200/255, 36/255, 35/255)]
    # 画出alpha=2的概率图
    os.makedirs('figure', exist_ok=True)
    plot_performance(ys=alpha2_ps, xs=lrs, title='Cache Hit Rate with Attention', xlabel='Episode',
                     ylabel='Cache Hit Rate', filename='figure/MADQN3-Cache Hit Rate.png', grid=True, colors=mycolors)
    plot_performance(ys=chains_delay, xs=lrs, title='Chain Delay with Attention', xlabel='Episode',
                     ylabel='Chain Delay', filename='figure/MADQN3-delay.png', grid=True, colors=mycolors)
    plot_performance(ys=episode_rewards, xs=lrs, title='Episode Reward with Attention', xlabel='Episode',
                     ylabel='Episode Reward', filename='figure/MADQN3-reward.png', grid=True, colors=mycolors)
    plot_performance(ys=moving_rewards, xs=lrs, title='Mean Episode Reward with Attention', xlabel='Episode',
                     ylabel='Mean Episode Reward', filename='figure/MADQN3-reward_mean.png', grid=True, colors=mycolors)
    plot_performance(ys=losses, xs=lrs, title='Loss with Attention', xlabel='Episode',
                     ylabel='Loss', filename='figure/MADQN3-losses.png', grid=True, colors=mycolors)
