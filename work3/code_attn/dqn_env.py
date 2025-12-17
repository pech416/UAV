import numpy as np
import math

import envs


class DQNEnv:
    def __init__(self, num_uav, num_services, fixed_wing, n=3):
        self.done = False
        self.num_services = num_services
        self.num_uav = num_uav
        self.n = n  # 路径损耗指数

        self.fixed_wing = fixed_wing

        self.service_list = envs.Service.generate_services_list(self.num_services)
        self.task_chain = envs.TaskChain(self.service_list)
        self.uav_group = [envs.Uav(self.service_list) for _ in range(self.num_uav)]

        self.current_uav_index = 0  # 当前与任务交互的无人机的索引
        self.distances = []  # 任务到每个uav的距离

        self.state = self.state_init()
        self.n_state = len(self.state)
        # self.n_action = 2 ** num_services  # 服务集合数
        self.n_action = 70   # TODO

    def reset(self):
        self.task_chain = envs.TaskChain(self.service_list)
        self.uav_group = [envs.Uav(self.service_list) for _ in range(self.num_uav)]
        self.task_chain.current_task_index = 0

        # 重置环境到初始状态，返回初始观测。
        self.state = self.state_init()
        return self.state

    def state_init(self):
        task = self.task_chain.get_current_task()
        if task is None:
            return None

        # 当前任务的状态
        task_state = [
            task.data,
            task.compute,
            task.tau,
            task.service.service_id,
            task.service.service_size,
            task.service.cache_status,
            task.service.access  # 服务的访问次数
        ]

        uav = self.uav_group[self.current_uav_index]  # 根据索引取出当前的uav

        distance = self.euclid_distance(task.position, uav.position)

        # uav_services_list：0101列表，encode：01列表对应的一个整数数字
        encode = self.encode_uav_services(uav.uav_services_list)  # 将uav的缓存服务组合编码为一个整数动作

        # 当前uav的状态
        uav_state = [
            uav.compute,  # 计算资源
            uav.cache_space,  # 缓存空间
            encode,  # 缓存服务的id
            uav.battery  # 电量：缓存一个服务需要电量2-5，即服务大小。执行一个任务需要电量50，即task.compute/uav.compute * 50
        ]

        state_vector = task_state + uav_state + [distance]  # 状态变量

        return state_vector

    def encode_uav_services(self, uav_services_list):  # 将二进制列表编码为一个整数动作
        binary_services_list = ''.join([str(bit) for bit in uav_services_list])
        encoded = int(binary_services_list, 2)
        return encoded

    def action_space(self):
        # 返回所有可能的动作，即0~2^n-1的整数
        actions = [i for i in range(self.n_action)]
        return actions

    def action_decode(self, action):
        # 将整数动作解码为一个二进制列表，其中每个元素表示对应服务是否被缓存。
        binary_action = format(action, f'0{self.num_services}b')
        action_list = [int(bit) for bit in binary_action]
        return action_list

    def action_encode(self, action_list):
        # 将二进制列表编码为一个整数动作
        binary_action = ''.join([str(bit) for bit in action_list])
        action = int(binary_action, 2)
        return action

    def cache_update(self, action, current_uav):
        # print(f"缓存更新前的uav缓存服务uav_services：{current_uav.uav_services}")

        # # action解码，得到需要缓存的服务列表
        # action_list = self.action_decode(action)  # 解码为01
        # cache_list = [service for i, service in enumerate(current_uav.services) if action_list[i] == 1]  # action选择的服务
        # cache_list = [service for service in cache_list if service not in current_uav.uav_services]  # 移除已有的服务

        cache_list = [service for service in current_uav.services if service.service_id in action]
        cache_list = [service for service in cache_list if service not in current_uav.uav_services]

        # 进行缓存
        for service in cache_list:
            # 缓存service：1.有缓存空间，2.该服务未缓存，3.决定缓存该服务。4.有电量!!!
            if current_uav.battery >= service.service_size:

                # 移除service：1)uav的缓存空间不足， 2)uav_services不为空，即确实有服务可移除。
                while current_uav.cache_space < service.service_size and current_uav.uav_services:
                    # replaced_service = current_uav.uav_services.pop(0)
                    # replaced_service = random.choice(current_uav.uav_services)
                    # least_accessed_service = min(uav_services, key=lambda service: service.access)
                    replaced_service = min(current_uav.uav_services, key=lambda service: service.access)
                    replaced_service.cache_status = 0  # 标记为未缓存
                    current_uav.cache_space += replaced_service.service_size  # 释放缓存空间
                    current_uav.uav_services.remove(replaced_service)  # 从uav的缓存列表中移除服务
                    current_uav.uav_services_list[replaced_service.service_id] = 0  # 更新缓存二进制列表

                current_uav.uav_services.append(service)  # 添加服务到uav的缓存列表
                current_uav.uav_services_list[service.service_id] = 1  # 更新缓存二进制列表
                current_uav.cache_space -= service.service_size  # 减少缓存空间
                service.cache_status = 1  # 标记为已缓存
                current_uav.battery -= service.service_size  # 缓存服务需要电量

        return current_uav.uav_services

    def euclid_distance(self, p1, p2):
        d = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
        return d

    def shannon_formula(self, P, d, B, N):  # P发送功率，d距离，n路径损耗指数，B通信信道带宽，N总噪声功率（加入路径损耗模型）
        return B * np.log2(1 + (P * (d ** -self.n)) / N)

    def rate_transmit(self, task, A_uavs, fixed_wing):  # A_uavs 是一个二维列表，表示一组无人机任务。外部列表表示不同的任务，内部列表表示在每个任务中参与的无人机。
        # 任务到uav和固定翼的速率r.
        p1 = task.user.p0
        rates_uav = [float('inf')] if A_uavs is None else [self.shannon_formula(p1, self.euclid_distance(task.position, uav.position), uav.B, uav.N)
                     for uav in A_uavs]  # A_uavs是一个一维列表，表示task对应选择的uav组

        d2 = self.euclid_distance(task.position, fixed_wing.position)
        B2 = fixed_wing.B
        N2 = fixed_wing.N
        rate_fixed = self.shannon_formula(p1, d2, B2, N2)
        return rates_uav, rate_fixed

    def time_transmit(self, task, A_uavs, fixed_wing):
        # 计算固定翼和uav的传输速率
        rates_uav, rate_fixed = self.rate_transmit(task, A_uavs, fixed_wing)

        # 初始化时延
        t_fixed_transmit, t_user_transmit = 0, 0
        t_uav_transmit = [0 * len(rates_uav)]  # 列表

        if task.alpha == 0:  # 用户本地
            t_user_transmit = 0
        elif task.alpha == 1:  # 固定翼
            t_fixed_transmit = task.data / rate_fixed
        else:  # 任务卸载到uav
            if task.beta == 0:  # 任务不拆分
                t_uav_transmit = [task.data / rates_uav[0]]
            else:  # 任务拆分:1.计算每个uav传输时间。2.使用uav的注意力分数作为权重，乘传输时间。3.将结果存入列表
                # t_uav_transmit = [task.weight[i] * (task.data / rate) for i, rate in enumerate(rates_uav)]
                t_uav_transmit = [task.weight[i] * (task.data / rate) if rate is not None else 0 for i, rate in enumerate(rates_uav)]
        return t_fixed_transmit, t_uav_transmit, t_user_transmit

    def time_compute(self, task, A_uavs, fixed_wing):
        t_user_com, t_fixed_com = 0, 0
        t_uav_com = [0] if A_uavs is None else [0 * len(A_uavs)]

        if task.alpha == 0:  # 用户本地
            t_user_com = task.compute / task.user.compute
        elif task.alpha == 1:  # 固定翼
            t_fixed_com = task.compute / fixed_wing.compute
        else:  # uav
            if task.beta == 0:  # 任务不拆分
                t_uav_com = [task.compute / uav.compute for uav in A_uavs]
            else:  # 任务拆分
                t_uav_com = [task.weight[i] * (task.compute / uav.compute) if uav.compute is not None else 0 for i, uav in enumerate(A_uavs)]
        return t_user_com, t_uav_com, t_fixed_com

    def get_reward(self, task, A_uavs, fixed_wing):
        t_fixed_transmit, t_uav_transmit, t_user_transmit = self.time_transmit(task, A_uavs, fixed_wing)
        t_user_com, t_uav_com, t_fixed_com = self.time_compute(task, A_uavs, fixed_wing)

        # 根据alpha和beta值确定任务的总时延
        if task.alpha == 0:  # 本地
            task_time = t_user_com  # 本地只有计算时延
        elif task.alpha == 1:  # 固定翼
            task_time = t_fixed_transmit + t_fixed_com  # 固定翼的传输和计算时延之和
        else:  # 卸载到UAV
            if task.beta == 0:  # 不拆分
                task_time = min(t_uav_transmit) + min(t_uav_com)  # 选择时延最小的UAV
            else:  # 拆分
                # 拆分任务的时延为所有子任务时延的最大值
                task_time = max(t_uav_transmit[i] + t_uav_com[i] for i in range(len(A_uavs) - 1))
        time_reward = 1 / task_time if task_time > 0 else 0  # 避免除以0

        return time_reward, task_time


if __name__ == '__main__':
    service_num = 5
    services = envs.Service.generate_services_list(service_num)  # 生成服务列表

    task_chain = envs.TaskChain(services)  # 创建包含服务的任务链

    num_uav = 3
    uav_group = [envs.Uav(services) for _ in range(num_uav)]  # 这里应该为每个UAV创建一个新实例：Uav()创建实例

    fixed_wing = envs.FixedWing()

    env = DQNEnv(num_uav, service_num, fixed_wing)  # 实例化 DQNEnv 类

    # 输出初始化状态
    print("Initial State:", env.state)
    print(f"services:{services}")