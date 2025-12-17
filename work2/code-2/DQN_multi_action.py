import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from UAV_set import UAV
from UAV_copy import UAV

# DQN的参数设置
BATCH_SIZE = 64
LR = 0.005  # learning rate
# EPSILON = 0.8  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 256  # Q 现实网络的更新频率
MEMORY_CAPACITY = 2560  # 记忆库大小

env = UAV()

N_ACTIONS = env.n_actions
N_STATES = env.n_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        torch.nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 64)
        self.fc3.weight.data.normal_(0, 0.1)
        torch.nn.Dropout(0.5)

        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        # x = F.relu(x)
        x = torch.tanh(x)
        actions_value = self.out(x)
        # x = self.out(x)
        # actions_value = F.relu(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + 20))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)  # torch 的优化器
        # self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=LR, alpha=0.9)
        self.loss_func = nn.MSELoss()  # 误差公式

        self.epsilon_increment = 1.01
        self.epsilon_max = 1
        self.epsilon = 0.1 if self.epsilon_increment is not None else self.epsilon_max

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            # action = torch.max(actions_value, 1)[1].numpy()[0]  # return the argmax
            # print("action_value:", actions_value)
            actions_value_ss = actions_value.detach().numpy()[0]
            print("actions_value_ss:", actions_value_ss)
            action = [x[0] for x in sorted(enumerate(actions_value_ss), key=lambda x: x[1])[-20:]]

        else:  # 选随机动作
            # action = np.random.randint(0, N_ACTIONS)
            action = np.random.randint(0, N_ACTIONS, size=(20, )).tolist()

        print("action___:", action)
        return action

    def store_transition(self, s, a, r, s_):
        print("s:", s)
        print("a:", a)
        transition = np.hstack((s, a, [r], s_))
        # print("transition:", transition)
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        print("store_transition_memory_counter:", self.memory_counter)

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 20].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 20:N_STATES + 21])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        print("b_s:", b_s)
        print("b_a:", b_a)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        loss_ = loss.tolist()

        self.epsilon = self.epsilon * self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        print("loss_:", loss_)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_

    # def epsilon_incre(self):
    #     self.epsilon = self.epsilon * self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max