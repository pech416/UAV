# coding=utf-8
import random

import numpy as np
from matplotlib import pyplot as plt
import math

# import UAV_set
import Users

# matplotlib 正常显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# UAV的服务半径为200米，RSU的服务半径为500米
# UAV的总的计算资源为2.5GHz

CACHE_SPACE = 20

class RSU(object):
    def __init__(self,
                 P_o=6,  # RSU的发射功率增益
                 P_v_w=1,  # 车辆的传输功率(0.1W=20dbm)  0.1w~1w
                 N=0.001,  # 多传输之间的干扰
                 r=2.8,  # 传播参数
                 alpha=0.5,  # 缓存配比参数
                 RSU_r = 500,  # RSU的信号覆盖范围，半径为500米
                 c=3 * (10 ** 8),  # 无线电波的传输速度（光速）
                 f_R=3 * 10 ** 3,  # RSU上CPU的计算频率
                 eta_LoS=1.6,  # 附加视距路径损失
                 eta_NLoS=23,  # 附加非视距路径损失
                 sigma2=1.26 * 10 ** (-8),  # 白噪声
                 RSU_coord=[0, 0],  # RSU的位置
                 users=5,  # 车辆用户的数量
                 B=80,  # RSU和UAV分配给车辆的带宽，单位为MHz,为5G频段
                 B_cloud=240,  # 中心云分配给RSU和UAV的带宽，单位为MHz，为2.4G频段（其抗干扰的能力较强）
                 R_V_transmission_rate=400,  # RSU到车辆的传输速率为50M/s
                 RSU_cache_memory = 2000  # 256 * 2 ** 10  # RSU总的缓存空间大小为256G
                 ):

        self.n_features = 1

        # 设置相关参数
        # 光速 300000 km/s
        self.c = c
        # RSU的服务信号覆盖范围
        self.RSU_r = RSU_r
        # 车辆的传输功率
        self.P_v_w = P_v_w * 120
        # RSU发射功率增益 (1.0-2.0)
        self.P_o = P_o
        # RSU发射频率
        self.P_r = P_o * P_v_w
        # 基于环境的平均附加路径损耗值 (这里取 Dense Urban环境，单位dB)
        self.eta_LoS, self.eta_NLoS = eta_LoS, eta_NLoS
        # 传播参数
        self.r = r
        # 缓存配比参数
        self.alpha = alpha
        # 多传播之间的干扰
        self.N = N
        # RSU上CPU的计算频率
        self.F_r = f_R
        # 白噪声
        self.sigma2 = sigma2
        # 用户的数量
        self.users = users
        # RSU的位置
        self.rsu_coord = RSU_coord
        # RSU或UAV分配给车辆的带宽,单位为MHz
        self.B = B
        # 中心云分配给UAV和RSU的带宽
        self.B_cloud = B_cloud
        # UAV到车辆用户的传输速率
        self.R_V_transmission_rate = R_V_transmission_rate
        # RSU总的缓存空间大小
        self.RSU_cache_memory = RSU_cache_memory
        self.cache_memory = np.zeros((CACHE_SPACE, 6))
        self.memory_counter = 0

    def get_distance(self, users_coord, cloud_coord):
        # 车辆到RSU的直线距离
        d_v_r = []
        for user_coord in users_coord:
            d_tasks_r = []
            for task_coord in user_coord:
                d_r = math.sqrt((task_coord[0] - self.rsu_coord[0])**2 + (task_coord[1] - self.rsu_coord[1])**2)  # 计算RSU到车辆用户之间的直线距离
                d_tasks_r.append(d_r)
            d_v_r.append(d_tasks_r)

        d_r_cloud = math.sqrt((self.rsu_coord[0] - cloud_coord[0])**2 + (self.rsu_coord[1] - cloud_coord[1])**2)

        return d_v_r, d_r_cloud

    def get_rsu_transmission_rate(self, d_v_r, d_r_cloud, req_users):
        # 计算车辆卸载任务到UAV之间的传输速率、UAV卸载任务到中心云的传输速率、中心云到UAV的传输速率（假设后两个速率一样）
        users_rsu_transmission_rates = []  # 车辆与RSU之间的数据传输速率
        rsu_cloud_transmission_rates = []  # RSU与中心云之间的数据传输速率
        user_rsu_transmission_rate, rsu_cloud_transmission_rate = 0, 0
        for i in range(req_users):
            user_rsu_transmission_rate = 1.5 * self.B * np.log2(1 + self.P_v_w * np.power(np.array(d_v_r[i])/30, -self.r) / (self.sigma2 + self.N))  # 计算车辆用户到RSU之间的数据传输速率
            users_rsu_transmission_rates.append(user_rsu_transmission_rate.tolist())
            rsu_cloud_transmission_rate = 0.2 * self.B_cloud * np.log2((1 + 1000 * self.P_r * np.power(d_r_cloud/100, -self.r) / (self.sigma2 + self.N)))  # 计算RSU到中心云之间的数据传输速率
            # rsu_cloud_transmission_rates.append(rsu_cloud_transmission_rate)

        print("rsu_cloud_transmission_rate--:", rsu_cloud_transmission_rate)
        print("user_rsu_transmission_rate--:", users_rsu_transmission_rates)

        return users_rsu_transmission_rates, rsu_cloud_transmission_rate

    def get_rsu_transmission_time(self, users, v_r_t_r, r_c_t_r, users_tasks, rsu_cached_type_list, rsu_cached_type, service_content_list):  # c_u_s1, c_u_s2是两个二维数组，一维长度与用户数相同，二维长度与用户的请求访问计算的任务数相同
        # v_r_t_r, r_c_t_r 分别表示车辆到RSU的传输速率、RSU到中心云的传输速率
        # 计算车辆卸载任务到UAV上去的传输时延、UAV卸载任务到中心云中去的传输时延、中心云到UAV的回程时延、UAV到车辆的传输时延
        # req_users表示请求访问服务的车辆用户数, c_u_s1表示UAV上是否缓存了相对应的服务,c_u_s2表示UAV上是否缓存了相对应的内容
        t_v_r_1, t_r_v_1 = [], []
        print("v_r_t_r:", v_r_t_r)
        print("r_c_t_r:", r_c_t_r)
        services, contents = service_content_list[0], service_content_list[1]
        for i in range(users):  # uav上缓存了相应的服务或内容
            t_vr, t_rv = [], []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 0 and users_tasks[i][j][-1] in rsu_cached_type_list:
                    t_vr.append(users_tasks[i][j][0] / v_r_t_r[i][j])
                    if users_tasks[i][j][-1] in rsu_cached_type[0]:
                        t_rv.append(services[users_tasks[i][j][-1]][0] * 0.5 / self.R_V_transmission_rate)
                    else:
                        t_rv.append(contents[users_tasks[i][j][-1]][0] / self.R_V_transmission_rate)
                else:
                    t_rv.append(0)
                    t_vr.append(0)
            t_v_r_1.append(t_vr)  # 车辆任务卸载到UAV的传输时延
            t_r_v_1.append(t_rv)  # UAV上的计算结果返回到车辆的传输时延

        t_v_r_0, t_r_c_0, t_c_r_0, t_r_v_0 = [], [], [], []
        for i in range(users):  # uav上没有缓存相应的服务和内容，需要进一步卸载到中心云中去
            t_vr0, t_rc0, t_cr0, t_rv0 = [], [], [], []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 0 and users_tasks[i][j][-1] not in rsu_cached_type_list:
                    t_vr0.append(users_tasks[i][j][0] / v_r_t_r[i][j])
                    t_rc0.append(users_tasks[i][j][0] / r_c_t_r)
                    t_cr0.append(users_tasks[i][j][0] * 0.6 / r_c_t_r)
                    t_rv0.append(users_tasks[i][j][0] * 0.6 / self.R_V_transmission_rate)
                else:
                    t_vr0.append(0)
                    t_rc0.append(0)
                    t_cr0.append(0)
                    t_rv0.append(0)
            t_v_r_0.append(t_vr0)
            t_r_c_0.append(t_rc0)
            t_c_r_0.append(t_cr0)
            t_r_v_0.append(t_rv0)

        t_v_r = (np.array(t_v_r_1) + np.array(t_v_r_0)).tolist()
        t_r_v = (np.array(t_r_v_1) + np.array(t_r_v_0)).tolist()
        t_r_c = t_r_c_0
        t_c_r = t_c_r_0

        print("t_v_r:", t_v_r)
        print("t_r_v:", t_r_v)
        print("t_c_r:", t_c_r)
        print("t_r_c:", t_r_c)

        # 计算任务从车辆卸载到UAV以及从UAV返回结果的总传输时延
        trans_t_vr = (np.array(t_v_r) + np.array(t_r_v)).tolist()
        # 计算任务从车到UAV到中心云再到UAV再到车辆的总传输时延
        trans_t_vrc = (np.array(t_v_r) + np.array(t_r_c) + np.array(t_c_r) + np.array(t_r_v)).tolist()

        print("trans_t_vr:", trans_t_vr)
        print("trans_t_vrc:", trans_t_vrc)

        return trans_t_vr, trans_t_vrc, t_v_r, t_r_v

    def get_rsu_computing_time(self, users_tasks, services_list):
        # 任务在RSU上的计算时延
        # computing_capacity = random.randint(3, 10)*self.F_r/30
        users_tasks_comp_s = []
        for i in range(len(users_tasks)):
            user_tasks_comp = []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 0 and users_tasks[i][j][-1] in services_list:
                    user_tasks_comp.append(users_tasks[i][j][1])
                else:
                    user_tasks_comp.append(0)
            users_tasks_comp_s.append(user_tasks_comp)

        rsu_comp_t = []
        for i in range(len(users_tasks)):
            computing_capacity = random.randint(3, 10) * self.F_r / 30
            rsu_comp_t.append((np.array(users_tasks_comp_s[i]) / computing_capacity).tolist())

        return rsu_comp_t

    def rsu_rest(self):
        # 初始化UAV的位置
        return [0, 0]

    # def get_uav_request_cache_action(self, action, services, contents, type_list, type_num):
    #     # 看UAV需要对哪些已经缓存的服务和内容进行替换
    #     action_uav_memory, services_list, contents_list = RSU().is_cached(action, services, contents, type_list,
    #                                                                       type_num)
    #
    #     return action_uav_memory, services_list, contents_list
    #
    # def is_cached(self, action, services, contents, type_list, type_num):
    #     cached_service_type, cached_content_type = type_list[0], type_list[1]
    #     for i in range(type_num):
    #         if action == i:  # 决定是否替换0类型服务或内容
    #             if i not in cached_service_type and i not in cached_content_type:
    #                 if np.random.uniform() < 0.5 and self.RSU_cache_memory >= services[i][0] and np.random.uniform() < services[i][2]:
    #                     services[i][3] = 1
    #                     self.RSU_cache_memory -= services[i][0]
    #                     cached_service_type.append(i)
    #                 elif np.random.uniform() >= 0.5 and self.RSU_cache_memory >= contents[i][0] and np.random.uniform() < contents[i][2]:
    #                     contents[i][3] = 1
    #                     self.RSU_cache_memory -= contents[i][0]
    #                     cached_content_type.append(i)
    #             elif i in cached_service_type and i not in cached_content_type:
    #                 services[i][3] = 2
    #                 self.RSU_cache_memory += services[i][0]
    #                 cached_service_type.remove(i)
    #             elif i not in cached_service_type and i in cached_content_type:
    #                 contents[i][3] = 2
    #                 self.RSU_cache_memory += contents[i][0]
    #                 cached_content_type.remove(i)
    #             else:
    #                 services[i][3] = 2
    #                 self.RSU_cache_memory += services[i][0]
    #
    #     return self.RSU_cache_memory, cached_service_type, cached_content_type

    def get_origin_cache(self, services, contents, req_num):
        # 初始化进行随机缓存
        rsu_services, rsu_contents = [[i for i in j]for j in services], [[i for i in j]for j in contents]
        rsu_cached_type_list = []
        rsu_services_list, rsu_contents_list = [], []
        rsu_service_cache_size = self.RSU_cache_memory * 0.5
        rsu_cached_service_type = []
        for i in range(len(rsu_services)):
            # rsu_services[i][3] = np.random.choice((0, 2), size=1, p=[rsu_services[i][2], 1-rsu_services[i][2]])[0]  # 给需要缓存的服务打上标签，1表示缓存在UAV上，2表示不进行缓存
            if req_num[i] > 0 and rsu_service_cache_size >= rsu_services[i][0]:
                rsu_services[i][3] = 0
                rsu_service_cache_size -= rsu_services[i][0]
                rsu_cached_service_type.append(rsu_services[i][-1])
                rsu_services_list.append(rsu_services[i])
                rsu_services_list[-1][3] = 0
                index = self.memory_counter % CACHE_SPACE
                self.cache_memory[index, :] = rsu_services[i]
                self.memory_counter += 1

        rsu_cached_content_type = []
        rsu_content_cache_size = self.RSU_cache_memory * 0.5
        # rsu_contents_sort = sorted(rsu_contents, key=rsu_contents[:][2], reverse=True)
        for i in range(len(rsu_contents)):
            # if rsu_contents[i][2] > rsu_contents[i+1][2]:
                # rsu_contents_sort = sorted()
            # rsu_contents[i][3] = np.random.choice((0, 2), size=1, p=[rsu_contents[i][2], 1-rsu_contents[i][2]])[0]  # 给需要缓存的内容打上标签，1表示缓存在UAV上，2表示不进行缓存
            if req_num[i] > 0 and rsu_content_cache_size >= rsu_contents[i][0]:
                rsu_contents[i][3] = 0
                rsu_content_cache_size -= rsu_contents[i][0]
                rsu_cached_content_type.append(rsu_contents[i][-1])
                rsu_contents_list.append(rsu_contents[i])
                rsu_contents_list[-1][3] = 0
                index = self.memory_counter % CACHE_SPACE
                self.cache_memory[index, :] = rsu_contents[i]
                self.memory_counter += 1

        self.RSU_cache_memory = (self.RSU_cache_memory - (self.RSU_cache_memory * 0.5 - rsu_service_cache_size) - (
                    self.RSU_cache_memory * 0.5 - rsu_content_cache_size))

        for i in rsu_cached_content_type:
            if i in rsu_cached_service_type:
                rsu_cached_service_type.remove(i)
                rsu_services_list.remove(rsu_services[i])
                self.RSU_cache_memory += rsu_services[i][0]

        cached_type = rsu_cached_service_type + rsu_cached_content_type
        for i in range(len(rsu_services)):
            # 给需要缓存的服务打上标签，1表示缓存在UAV上，2表示不进行缓存
            if req_num[i] > 0 and self.RSU_cache_memory >= rsu_services[i][0] and (rsu_services[i][-1] not in cached_type):
                rsu_services[i][3] = 0
                self.RSU_cache_memory -= rsu_services[i][0]
                rsu_cached_service_type.append(rsu_services[i][-1])
                rsu_services_list.append(rsu_services[i])
                rsu_services_list[-1][3] = 0
                index = self.memory_counter % CACHE_SPACE
                self.cache_memory[index, :] = rsu_services[i]
                self.memory_counter += 1

        rsu_cached_type_list.append(rsu_cached_service_type)
        rsu_cached_type_list.append(rsu_cached_content_type)

        return rsu_services_list, rsu_contents_list, rsu_cached_type_list, self.RSU_cache_memory

    def get_rsu_reward(self, rsu_total_t):  # rsu_total_t:[[0, 0.3],[0, 0],[1.2, 0.0],[1.1, 0],...]
        # 计算奖励，奖励与计算成本、计算时延、通信成本、通信时延有关
        rewards = []
        # for i in range(len(rsu_total_t)):
        #     reward = []
        #     for j in range(len(rsu_total_t[i])):
        #         if rsu_total_t[i][j] != 0:
        #             r = 1 / rsu_total_t[i][j]
        #             reward.append(r)
        #         else:
        #             reward.append(0)
        #     rewards.append(sum(reward))

        ############################################
        # rewards = (-np.array(rsu_total_t))
        for user_time in rsu_total_t:
            reward = []
            for j in user_time:
                reward.append(-j)
            rewards.append(sum(reward))

        return rewards

    def get_end(self, cars_coord, rewards, users_tasks, total_time):
        # 判断服务是否结束终止
        # is_achieve_ = [False, False, False, False, False]
        # is_achieve_ = []
        # for i in range(len(users_tasks)):
        #     is_achieve_.append(False)
        d_vr = []
        for i in range(len(cars_coord)):
            d_vr.append(np.sqrt((cars_coord[i]) ** 2))

        is_achieve = False
        uninstall_num = 0  # 卸载到UAV上的总任务数
        complete_num = 0  # 在UAV上已经处理的总任务数
        for i in range(len(users_tasks)):
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 0:
                    uninstall_num += 1
                if users_tasks[i][j][-3] == 1 and users_tasks[i][j][-2] == 0:  #  and total_time[i][j] <= users_tasks[i][j][2]
                    complete_num += 1
        #
        # for i in range(len(users_tasks)):
        #     if uninstall_num == complete_num and d_v_r[i] < self.RSU_r:
        #         is_achieve_[i] = True
        #         rewards[i] += 2
        #     elif d_v_r[i] >= self.RSU_r:
        #         is_achieve_[i] = True
        #         rewards[i] -= 2
        #     else:
        #         is_achieve_[i] = False
        #         rewards[i] -= 1

        # for i in range(len(users_tasks)):
        #     if d_v_r[i] >= self.RSU_r:
        #         is_achieve_[i] = True
        #         rewards[i] -= 2
        #
        #     elif d_v_r[i] < self.RSU_r and np.array(users_tasks[i][:][-3]).all() == 1:
        #         is_achieve_[i] = True
        #         rewards[i] += 2
        #
        #     else:
        #         rewards[i] += 0
        #
        # if False not in is_achieve_:
        #     is_achieve = True
        is_achieve_rsu = False
        # max_d = max(d_v_r[0])
        # for i in range(1, len(users_tasks)):
        #     if max_d < max(d_v_r[i]):
        #         max_d = max(d_v_r[i])

        # mean_d = []
        # for i in range(len(users_tasks)):
        #     mean_d.append(np.mean(d_v_r[i]))

        if uninstall_num == complete_num and uninstall_num != 0:
            is_achieve = True
            rewards = (np.array(rewards) + 1).tolist()

        elif min(d_vr) >= self.RSU_r and uninstall_num != complete_num:  # 所有车辆已经离开UAV的服务范围，并且未服务并未完成
            is_achieve_rsu = True
            rewards = (np.array(rewards) - 2).tolist()

        else:
            rewards = (np.array(rewards) - 0).tolist()

        return is_achieve, rewards, uninstall_num - complete_num, is_achieve_rsu

    def get_rsu_state(self, remain_tasks_num, cars_coord):
        # 获取UAV的状态, 剩余需要处理的任务数量、剩余缓存空间、车到UAV之间的距离
        state = [remain_tasks_num/60]
        # print("d_v_r:", d_v_r)
        for i in range(len(cars_coord)):
            d_v_r = np.sqrt((cars_coord[i])**2)
            if d_v_r >= self.RSU_r:
                state.append(1)
            else:
                state.append(d_v_r/self.RSU_r)

        return state

    def get_data_save(self, ):
        # 获取计算成本、计算时延、通信成本、通信时延、奖励
        pass

    def get_data_show(self):
        # 画出各数据与Episode和服务车辆数量之间的关系
        pass