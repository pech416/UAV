import torch
import torch.nn.functional as F
import numpy as np
import copy
from networks import Actor, Critic_MADDPG


class MADDPG(object):
    def __init__(self, args, agent_id):
        self.N = args.N
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip

        # 创建本 agent 的 actor 和 critic 网络
        self.actor = Actor(args, agent_id)
        self.critic = Critic_MADDPG(args)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_n, cached_actions=None):
        # 从经验池采样
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        #从经验池中采样一个 batch 的数据，结构为每个 agent 一份对应的
        # 若未缓存动作，则计算所有 agent 的 target 动作（冻结所有 agent 策略输出）
        if cached_actions is None:
            #让每个 agent 都用自己当前的 actor_target 网络对 next_obs 做前向推理；
            #所有动作都提前计算好，相当于冻结当前策略输出。
            #后续每个 agent 都会用这个静态版本的动作训练自己的 critic，从而解耦依赖，支持并行
            cached_actions = [
                agent.actor_target(batch_obs_next_n[i])
                for i, agent in enumerate(agent_n)
            ]

        # critic 网络训练
        with torch.no_grad():
            Q_next = self.critic_target(batch_obs_next_n, cached_actions)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next

        current_Q = self.critic(batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(target_Q, current_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # actor 网络训练：替换当前 agent 的动作，其他 agent 使用缓存动作
        batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
        actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # 更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(),
                   f"./model/{env_name}/{algorithm}_actor_number_{number}_step_{int(total_steps / 1000)}k_agent_{agent_id}.pth")
