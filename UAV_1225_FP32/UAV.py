import copy
import random

import numpy as np
import math

# 无人机的移动范围为以[0, 0]为原点，半径为3000的范围
class UAV:
    def __init__(self,
                 cached_service_type,
                 cached_content_type,
                 P_o=6,  # 无人机的发送功率增益
                 P_v_w=1,  # 车辆的传输功率(0.1W=20dbm)  0.1w~1w
                 N=0.001,  # 多传输之间的干扰
                 r=2.8,  # 传播参数
                 alpha=0.3,  # 缓存配比参数
                 c=3 * (10 ** 8),  # 无线电波的传输速度（光速）
                 eta_LoS=1.6,  # 附加视距路径损失
                 eta_NLoS=23,  # 附加非视距路径损失
                 sigma2=1.26 * 10 ** (-8),  # 白噪声
                 B=40,  # RSU和UAV分配给车辆的带宽，单位为MHz,为5G频段
                 B_cloud=120,  # 中心云分配给RSU和UAV的带宽，单位为MHz，为2.4G频段（其抗干扰的能力较强）
                 U_V_transmission_rate=400,  # UAV到车辆的传输速率为50M/s
                 ):
        # 无人机的覆盖范围
        self.service_distance = 600
        # 无人机的移动范围
        self.uav_movable_range = 20000
        # 无人机的位置
        self.position = [random.randint(-self.uav_movable_range, self.uav_movable_range), random.randint(-self.uav_movable_range, self.uav_movable_range), 50]  # [x, y, z] z基本不改变
        # 无人机的CPU频率
        self.F_cpu = 2 * 10 ** 4
        # 已经缓存的服务
        self.cached_service_type = cached_service_type
        # 已经缓存的内容
        self.cached_content_type = cached_content_type
        # 无人机接收用户任务的存储空间
        self.storage_space = random.randint(20000, 50000)  # MB
        # 无人机的剩余电量
        self.remain_power = random.randint(20000, 30000)  # 毫安
        # 当前的电费单价(0.8块钱一度电)
        self.base_cost_energy = 1000
        # 派遣该无人机机出去需要的成本
        self.base_cost = 0.1 * (self.remain_power / self.base_cost_energy + self.storage_space / 1000 + self.F_cpu / 1000) + random.randint(20, 50)
        # 当前为车辆提供服务所获得的收益
        self.profits = 0
        # 当前无人机已经处理的任务数
        self.tasked_number = 0

        # 无人机的动作空间维度和状态空间维度
        self.n_actions = 40
        self.n_features = 5 * 2 + 4 + 1 + 1

        # 设置无人机的相关服务参数
        # 无人机之间的距离最小值
        self.uu_min_dis = self.service_distance * 0.7  # 后续需要调整
        # 车辆的传输功率
        self.P_v_w = P_v_w * 120 * 2
        # UAV发射功率增益 (1.0-2.0)
        self.P_o = P_o
        # UAV 发射频率
        self.P_u = P_o * P_v_w
        # 基于环境的平均附加路径损耗值 (这里取 Dense Urban环境，单位dB)
        self.eta_LoS, self.eta_NLoS = eta_LoS, eta_NLoS
        # 传播参数
        self.r = r
        # 光速(无线电传播速度)
        self.c = c
        # 无人机的充电单位时间的能耗
        self.alpha = alpha
        # 多传播之间的干扰
        self.N = N
        # 白噪声
        self.sigma2 = sigma2
        # RSU或UAV分配给车辆的带宽,单位为MHz
        self.B = B
        # 中心云分配给UAV和RSU的带宽
        self.B_cloud = B_cloud
        # UAV到车辆用户的传输速率
        self.U_V_transmission_rate = U_V_transmission_rate
        # 当前无人机的服务范围内的车辆用户的数量
        self.communication_range_user_number = 0
        # 记录无人机处理任务的总时间
        self.uav_cope_tasks_time = 0
        # 记录无人机的奖励
        self.uav_reward = 0
        # 记录已经加入到任务增加导致成本增大的影响中
        self.task_num_sill = 20

    # 无人机的初始化
    def get_init_uav(self):
        self.service_distance = 600
        self.position = [random.randint(-self.uav_movable_range, self.uav_movable_range), random.randint(-self.uav_movable_range, self.uav_movable_range), 50]
        self.F_cpu = 2 * 10 ** 4
        self.storage_space = random.randint(20000, 30000)  # MB
        self.remain_power = random.randint(20000, 30000)  # 毫安
        self.base_cost = 0.1 * (self.remain_power / self.base_cost_energy + self.storage_space / 1000 + self.F_cpu / 1000) + random.randint(20, 50)

    # 控制无人机移动
    def uav_move(self, direction, distance):
        # # 先保存无人机当前位置
        # uav_position_temp = copy.deepcopy(self.position)
        # # direction就是无人机的移动方向，也就是action
        # if direction == 0:
        #     self.position[1] += distance
        # elif direction == 1:
        #     self.position[0] += distance / math.sqrt(2.0)
        #     self.position[1] += distance / math.sqrt(2.0)
        # elif direction == 2:
        #     self.position[0] += distance
        # elif direction == 3:
        #     self.position[0] += distance / math.sqrt(2.0)
        #     self.position[1] -= distance / math.sqrt(2.0)
        # elif direction == 4:
        #     self.position[1] -= distance
        # elif direction == 5:
        #     self.position[0] -= distance / math.sqrt(2.0)
        #     self.position[1] -= distance / math.sqrt(2.0)
        # elif direction == 6:
        #     self.position[0] -= distance
        # else:
        #     self.position[0] -= distance / math.sqrt(2.0)
        #     self.position[1] += distance / math.sqrt(2.0)

        # 无人机的连续的移动方向从0到360度
        self.position[0] = self.position[0] + distance * math.cos(direction)
        self.position[1] = self.position[1] + distance * math.sin(direction)

    # 计算第i个UAV与第j个用户之间的距离， 第i个UAV到远程云之间的距离
    def get_uv_distance(self, user_position, cloud_position):
        # print("self.position:", self.position)
        # print("user_position:", user_position)
        # print("(self.position[0] - user_position[0]) ** 2：", (self.position[0] - user_position[0]) ** 2)
        # print("(self.position[1] - user_position[1]) ** 2:", (self.position[1] - user_position[1]) ** 2)
        # 第i辆UAV跟第j个用户之间的距离
        temp1 = np.power((self.position[0] - user_position[0]), 2) + 0.0
        temp2 = np.power((self.position[1] - user_position[1]), 2) + 0.0
        d_level = math.sqrt(temp1 +temp2)  # 某一车辆到UAV的水平距离
        # print("d_level:", d_level)
        d_u_v = math.sqrt(d_level ** 2 + self.position[2] ** 2)  # 某一车辆到UAV的直线距离

        # 计算第i个UAV到远程云之间的距离
        d_u_cloud = math.sqrt((self.position[0] - cloud_position[0]) ** 2 + (self.position[1] - cloud_position[1]) ** 2)

        # 返回第i个UAV与第j个用户之间的距离， 第i个UAV到远程云之间的距离
        return d_u_v, d_u_cloud

    # 计算车辆到无人机之间的传输速率、无人机到中心云的传输速率
    def get_uav_transmission_rate(self, d_u_v, d_u_cloud):
        uav_cloud_transmission_rate = 10 * self.B_cloud * np.log2(1 + 10000 * self.P_u * np.power(d_u_cloud / 100, -self.r) / (self.sigma2 + self.N))  # 香农公式计算UAV到中心云之间的数据传输速率
        uav_user_transmission_rate = 100 * self.B * np.log2(1 + 5000 * (self.P_v_w * np.power(d_u_v / 10, -self.r) / (self.sigma2 + self.N)))  # 香农公式计算车到UAV之间的数据传输速率
        # print("uav_user_transmission_rate:", uav_user_transmission_rate)
        # print("uav_cloud_transmission_rate:", uav_cloud_transmission_rate)
        # 返回用户与无人机之间的传输速率，无人机与远程云之间的传输率
        return uav_user_transmission_rate, uav_cloud_transmission_rate

    # 计算任务在无人机上的计算时延、任务传输到无人机上的传输时延
    def get_uav_transmission_computing_time(self, task, uav_user_transmission_rate, uav_cloud_transmission_rate):
        # 总时延
        total_time = 0

        # 如果当前无人机上缓存了对应的服务或内容就可以在无人机上进行任务的处理
        # 如果当前无人机上缓存了对应的内容
        if task[0] in self.cached_content_type:
            total_time += 0
        # 如果当前无人机上缓存了对应的服务
        elif task[0] in self.cached_service_type:
            computing_capacity = random.randint(3, 8) * self.F_cpu / 30.0
            total_time += task[1] / computing_capacity
        # 如果当前无人机上没有缓存对应的服务和内容，就需要进一步传输到远程云中去处理
        else:
            uav_cloud_time = (task[1] * 1.0) / uav_cloud_transmission_rate
            total_time += 2 * uav_cloud_time
        #     在无人机上的处理时延          任务传输到无人机上的时延                     任务结果返回给车辆用户
        return total_time + (task[1] * 1.0) / uav_user_transmission_rate + (task[1] * 0.6) / self.U_V_transmission_rate
