import numpy as np
import torch
import torch.nn as nn
import random

import envs
import dqn_env


class DQN(nn.Module):
    def __init__(self, n_input, n_output):  # 构造函数
        super(DQN, self).__init__()  # 调用父类的构造函数

        self.fc1 = nn.Linear(n_input, 64)  # 第一个全连接层
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化权重

        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        # self.dropout2 = torch.nn.Dropout(0.5)  # 加入Dropout层防止过拟合

        self.fc3 = nn.Linear(64, 64)
        self.fc3.weight.data.normal_(0, 0.1)
        # self.dropout3 = torch.nn.Dropout(0.5)

        self.out = nn.Linear(64, n_output)  # 输出层
        self.out.weight.data.normal_(0, 0.1)

    # 定义前向传播过程
    def forward(self, x):  # x为输入
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x = self.dropout2(x)
        x = torch.tanh(self.fc3(x))
        # x = self.dropout3(x)
        actions_value = self.out(x)
        return actions_value


class Agent:
    def __init__(self, idx, n_input, n_output, action_space, learning_rate, GAMMA=0.9, MEMORY_SIZE=2300, BATCH_SIZE=64):
        self.idx = idx  # agent ID

        self.n_input = n_input  # 输入的维度
        self.n_output = n_output  # 输出的维度
        self.action_space = action_space

        self.GAMMA = GAMMA  # 奖励递减率/折扣因子。GAMMA值越高，表示我们希望agent更加关注未来，这比更加关注眼前更难，因此训练更加缓慢和困难。
        self.learning_rate = learning_rate  # 学习率

        self.MEMORY_SIZE = MEMORY_SIZE  # 经验池的大小
        self.BATCH_SIZE = BATCH_SIZE  # 每个批次的大小

        self.learn_step_counter = 0  # 学习步数计数，决定何时更新目标网络
        self.TARGET_REPLACE_ITER = 256  # 目标网络的更新频率

        # 经验池
        self.memory = np.zeros((self.MEMORY_SIZE, self.n_input * 2 + 2 + 20))
        self.memory_count = 0

        # 初始化在线网络和目标网络
        self.online_net = DQN(self.n_input, self.n_output)  # 用于选择动作和学习的网络
        self.target_net = DQN(self.n_input, self.n_output)  # 用于稳定学习过程的网络
        self.target_net.load_state_dict(self.online_net.state_dict())  # 将目标网络的参数设为在线网络当前的参数
        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        self.loss_func = nn.MSELoss()  # 回归任务，使用均方误差MSE

        # ε-greedy策略相关参数。
        self.epsilon_increment = 1.01  # ε值的增量。1.01表示每次ε值会增加1%
        self.epsilon_max = 1  # ε值的最大值。ε的值通常在0到1之间。
        self.epsilon = 0.1 if self.epsilon_increment is not None else self.epsilon_max  # ε值的初始值

        self.q_values_history = []  # Q值历史

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 增加维度以符合网络输入要求

        if np.random.uniform() < self.epsilon:  # 选取最优动作
            actions_value = self.online_net.forward(x)
            actions_value_ss = actions_value.detach().numpy()[0]
            # action = [x[0] for x in sorted(enumerate(actions_value_ss), key=lambda x: x[1])[-20:]]
            action = [x[0] for x in sorted(enumerate(actions_value_ss), key=lambda x: x[1])[-20:]]  # TODO
        else:  # 随机选取动作
            action = np.random.randint(0, self.action_space, size=(20,)).tolist()
        return action  # 返回选定的动作

    def store_transition(self, s, a, r, s_, done):
        # 存储经验回放数据
        r = np.array([r])
        done = np.array([done])
        transition = np.hstack((s, a, r, done, s_))
        index = self.memory_count % self.MEMORY_SIZE
        self.memory[index, :] = transition
        self.memory_count += 1

    def learn(self):
        # 更新目标网络参数
        # print(f"learn_step_counter:{self.learn_step_counter}, TARGET_REPLACE_ITER:{self.TARGET_REPLACE_ITER}")
        # print(f"learn_step_counter % TARGET_REPLACE_ITER:{self.learn_step_counter % self.TARGET_REPLACE_ITER}")
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # 取余数
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.learn_step_counter += 1

        # 从经验池中抽样
        indexes = np.random.choice(self.MEMORY_SIZE, self.BATCH_SIZE)
        batch_memory = self.memory[indexes, :]  # 根据上面计算出的 indexes 从 self.memory 数组中选择对应的样本。

        batch_s = torch.FloatTensor(batch_memory[:, :self.n_input])
        batch_a = torch.LongTensor(batch_memory[:, self.n_input:self.n_input + 20].astype(int))  # TODO
        batch_r = torch.FloatTensor(batch_memory[:, self.n_input + 20:self.n_input + 21])
        batch_s_ = torch.FloatTensor(batch_memory[:, -self.n_input:])
        batch_done = torch.FloatTensor(batch_memory[:, self.n_input + 20:self.n_input + 21])

        # 计算Q值和目标Q值
        # 当前状态Q值，下一状态Q值，目标Q值
        q_values = self.online_net(batch_s).gather(1, batch_a)
        # q_next = self.target_net(b_s_).detach()  # 使用目标网络计算下一状态的Q值。detach是将Q值从计算图中分离出来，使其不参与反向传播的计算
        # next_q_values = self.target_net(batch_s_).max(1, keepdim=True)[0]
        next_q_values = self.target_net(batch_s_).detach()
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_target = batch_r + self.GAMMA * next_q_values * (1 - batch_done)
        q_target = batch_r + self.GAMMA * next_q_values.max(1)[0].view(self.BATCH_SIZE, 1)
        # 进行梯度下降
        loss = self.loss_func(q_values, q_target)

        # 更新ε值
        self.epsilon = self.epsilon * self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 优化模型参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # 返回当前的损失值

    def save_model(self, file_name):
        torch.save(self.online_net.state_dict(), file_name)
        # torch.save(self.online_net, file_name)  # 保存整个模型

    def load_model(self, file_name):
        self.online_net.load_state_dict(torch.load(file_name))
        self.target_net.load_state_dict(torch.load(file_name))
        # self.target_net(torch.load(file_name))  # 加载整个模型




