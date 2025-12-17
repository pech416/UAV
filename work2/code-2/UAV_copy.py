# coding=utf-8
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from Users import User

# import RSU_set

# matplotlib 正常显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# UAV的服务半径为200米，RSU的服务半径为500米
# UAV的总的计算资源为2.5GHz

CACHE_SPACE = 30

class UAV(object):
    def __init__(self,
                 # state_space,
                 h=50,  # 无人机的飞行高度
                 P_o=6,  # 无人机的发送功率增益
                 P_v_w=1,  # 车辆的传输功率(0.1W=20dbm)  0.1w~1w
                 N=0.001,  # 多传输之间的干扰
                 r=2.8,  # 传播参数
                 alpha = 0.5,  # 缓存配比参数
                 c=3 * (10 ** 8),  # 无线电波的传输速度（光速）
                 fly_r=200,  # 以交叉路口为原点UAV的飞行半径
                 UAV_service_r = 300,  # UAV的服务信号覆盖半径为200米
                 f_U=2 * 10 ** 3,  # UAV上CPU的计算频率
                 eta_LoS=1.6,  # 附加视距路径损失
                 eta_NLoS=23,  # 附加非视距路径损失
                 sigma2=1.26 * 10 ** (-8),  # 白噪声
                 UAV_coord=[50, 5],  # UAV的初始位置
                 users = 6,  # 车辆用户的数量
                 B = 40,  # RSU和UAV分配给车辆的带宽，单位为MHz,为5G频段
                 B_cloud = 120,  # 中心云分配给RSU和UAV的带宽，单位为MHz，为2.4G频段（其抗干扰的能力较强）
                 U_V_transmission_rate = 400,  # UAV到车辆的传输速率为50M/s
                 UAV_cache_memory = 1500  # 64 * 2 ** 10,  # UAV总的缓存空间大小为64G
                 ):
        # self.state_space = state_space
        # self.state_num = len(state_space)
        # self.action_space = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # 10种缓存服务和内容的类型

        self.n_actions = 40
        self.n_features = 5 * 2 + 4 + 1 + 1

        # 设置相关参数
        # 光速 300000 km/s
        self.c = c
        # UAV以交叉路口为原点，以unit_r为半径进行移动(无人机在半径为fly_r的圆里飞行)
        self.fly_r = fly_r
        # UAV的服务信号覆盖范围
        self.UAV_service_r = UAV_service_r
        # UAV飞行高度
        self.h = h
        # 车辆的传输功率
        self.P_v_w = P_v_w * 120
        # UAV发射功率增益 (1.0-2.0)
        self.P_o = P_o
        # UAV 发射频率
        self.P_u = P_o * P_v_w
        # 基于环境的平均附加路径损耗值 (这里取 Dense Urban环境，单位dB)
        self.eta_LoS, self.eta_NLoS = eta_LoS, eta_NLoS
        # 传播参数
        self.r = r
        # 缓存配比参数
        self.alpha = alpha
        # 多传播之间的干扰
        self.N = N
        # UAV上CPU的计算频率
        self.F_u = f_U
        # 白噪声
        self.sigma2 = sigma2
        # UAV的位置
        self.UAV_coord = UAV_coord
        # 车辆用户的数量
        self.users = users
        # RSU或UAV分配给车辆的带宽,单位为MHz
        self.B = B
        # 中心云分配给UAV和RSU的带宽
        self.B_cloud = B_cloud
        # UAV到车辆用户的传输速率
        self.U_V_transmission_rate = U_V_transmission_rate
        # UAV总的缓存空间大小
        self.UAV_cache_memory = UAV_cache_memory
        # UAV的缓存空间
        self.cache_memory = np.zeros((CACHE_SPACE, 6))  # 初始化记忆库
        self.memory_counter = 0

    # def get_uav_coordinate_step(self, coord):  # coord表示UAV下一时刻的坐标位移，靠近原点的位移为负，远离原点的位移为正
    #     # 获取UAV的坐标，并对其坐标进行更新
    #     x0, y0 = self.UAV_coord[0], self.UAV_coord[1]
    #     u_r = ((x0 + coord[0]) ** 2 + (y0 + coord[1]) ** 2) ** 0.5
    #     if u_r < self.fly_r:
    #         self.UAV_coord[0] += coord[0]
    #         self.UAV_coord[1] += coord[1]

        # return self.UAV_coord
    def init_ratio_parameter(self, services, contents):
        services_size, contents_size = [], []
        for i in range(len(services)):
            if services[i][3] == 1:
                services_size.append(services[i][0])
            else:
                services_size.append(0)

        for i in range(len(contents)):
            if contents[i][3] == 1:
                contents_size.append(contents[i][0])
            else:
                contents_size.append(0)

        print("service_size:", services_size)
        print("content_size:", contents_size)

        print("services_size:", sum(services_size))
        print("contents_size:", sum(contents_size))
        alpha = (sum(services_size) + 5e-10) / (sum(services_size) + sum(contents_size) + 5e-10)

        return alpha

    def get_distance(self, UAV_coord, users_coord, cloud_coord):  # 传入vehicle的实时位置
        # 车辆到UAV的直线距离
        d_v_u = []  # 车到UAV的直线距离
        for user_coord in users_coord:
            d_tasks_u = []
            for task_coord in user_coord:
                d_level = math.sqrt((task_coord[0] - UAV_coord[0]) ** 2 + (task_coord[1] - UAV_coord[1]) ** 2)  # 某一车辆到UAV的水平距离
                d_u = math.sqrt(d_level ** 2 + self.h ** 2)  # 某一车辆到UAV的直线距离
                d_tasks_u.append(d_u)
            d_v_u.append(d_tasks_u)

        d_u_cloud = math.sqrt((UAV_coord[0] - cloud_coord[0]) ** 2 + (UAV_coord[1] - cloud_coord[1]) ** 2)

        return d_v_u, d_u_cloud

    def where_unload(self, users_tasks, uav_cached_type, rsu_cached_type, t_v_u, t_v_r, n):

        for ii in range(len(users_tasks)):
            for jj in range(len(users_tasks[ii])):
                if users_tasks[ii][jj][-1] in uav_cached_type and users_tasks[ii][jj][-1] not in rsu_cached_type:
                    users_tasks[ii][jj][-2] = 1
                    users_tasks[ii][jj][3] = 1
                elif users_tasks[ii][jj][-1] in rsu_cached_type and users_tasks[ii][jj][-1] not in uav_cached_type:
                    users_tasks[ii][jj][-2] = 0
                    users_tasks[ii][jj][3] = 0
                else:
                    # if t_v_u[ii][jj] < t_v_r[ii][jj]:
                    #     users_tasks[ii][jj][-2] = 1
                    #
                    # elif t_v_u[ii][jj] > t_v_r[ii][jj]:
                    #     users_tasks[ii][jj][-2] = 0

                    if t_v_u[ii][jj] < t_v_r[ii][jj] and n > 0:
                        users_tasks[ii][jj][-2] = 1
                    elif t_v_u[ii][jj] > t_v_r[ii][jj] and n > 0:
                        users_tasks[ii][jj][-2] = 0
                    n += 1
                    users_tasks[ii][jj][3] = 2

                    # if np.random.uniform() > 0.5:
                    #     users_tasks[ii][jj][-2] = 1
                    # else:
                    #     users_tasks[ii][jj][-2] = 0

        return users_tasks, n

    def finish_tasks_time(self, users, users_tasks, uav_cached_type_list, rsu_cached_type_list, trans_rsu, trans_uav):
        uav_cached_type = [k for type in uav_cached_type_list for k in type]
        rsu_cached_type = [k for type in rsu_cached_type_list for k in type]
        uav_total_t, rsu_total_t = [], []
        for i in range(users):
            u_t, r_t = [], []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 0:  # 卸载到RSU上
                    if users_tasks[i][j][-1] not in rsu_cached_type:  # 卸载到RSU上，但RSU上没有缓存相对应的服务和内容，只能进一步卸载到中心云去处理
                        r_t.append(trans_rsu[1][i][j])
                        u_t.append(0)
                    elif users_tasks[i][j][-1] in rsu_cached_type:
                        if users_tasks[i][j][-1] in rsu_cached_type_list[0]:
                            r_t.append(trans_rsu[0][i][j] + trans_rsu[3][i][j])
                        else:
                            r_t.append(trans_rsu[0][i][j])
                        u_t.append(0)
                elif users_tasks[i][j][-2] == 1:  # 卸载到UAV上
                    if users_tasks[i][j][-1] not in uav_cached_type:
                        u_t.append(trans_uav[1][i][j])
                        r_t.append(0)
                    elif users_tasks[i][j][-1] in uav_cached_type:
                        if users_tasks[i][j][-1] in uav_cached_type_list[0]:
                            u_t.append(trans_uav[0][i][j] + trans_uav[3][i][j])
                        else:
                            u_t.append(trans_uav[0][i][j])
                        r_t.append(0)
                else:
                    r_t.append(0)
                    u_t.append(0)

            uav_total_t.append(u_t)
            rsu_total_t.append(r_t)

        users_total_t = (np.array(uav_total_t) + np.array(rsu_total_t)).tolist()

        return users_total_t, uav_total_t, rsu_total_t

    def get_uav_transmission_rate(self, d_v_u, d_u_cloud, req_users):
        # 计算车辆卸载任务到UAV之间的传输速率、UAV卸载任务到中心云的传输速率、中心云到UAV的传输速率（假设后两个速率一样）
        # uav_cloud_transmission_rates = []  # UAV到中心云之间的数据传输速率
        users_uav_transmission_rates = []  # 车辆到UAV之间的数据传输速率
        user_uav_transmission_rate, uav_cloud_transmission_rate = 0, 0
        for i in range(req_users):
            # 修改了UAV到中心云之间传输速率的参数
            uav_cloud_transmission_rate = 0.4 * self.B_cloud * np.log2(1 + 1000 * self.P_u * np.power(d_u_cloud / 100, -self.r) / (self.sigma2 + self.N))  # 香农公式计算UAV到中心云之间的数据传输速率
            user_uav_transmission_rate = 3.6 * self.B * np.log2(1 + (self.P_v_w * np.power(np.array(d_v_u[i])/30, -self.r) / (self.sigma2 + self.N)))  # 香农公式计算车到UAV之间的数据传输速率
            users_uav_transmission_rates.append(user_uav_transmission_rate.tolist())
            # uav_cloud_transmission_rates.append(uav_cloud_transmission_rate)

            print("uav_cloud_transmission_rate--:", uav_cloud_transmission_rate)
            print("user_uav_transmission_rate--:", users_uav_transmission_rates)

        return users_uav_transmission_rates, uav_cloud_transmission_rate

    def get_uav_transmission_time(self, users, v_u_t_r, u_c_t_r, users_tasks, uav_cached_type_list, service_content_list, uav_cached_type):  # c_u_s1, c_u_s2是两个二维数组，一维长度与用户数相同，二维长度与用户的请求访问计算的任务数相同
        # 计算车辆卸载任务到UAV上去的传输时延、UAV卸载任务到中心云中去的传输时延、中心云到UAV的回程时延、UAV到车辆的传输时延
        # users_tasks_size = []
        # for i in range(len(users_tasks)):
        #     user_tasks_size = []
        #     for j in range(len(users_tasks[i])):
        #         if users_tasks[i][j][-2] == 1:
        #             user_tasks_size.append(users_tasks[i][j][0])
        #         else:
        #             user_tasks_size.append(0)
        #     users_tasks_size.append(user_tasks_size)

        t_v_u_1, t_u_v_1 = [], []
        print("v_u_t_r:", v_u_t_r)
        print("u_c_t_r:", u_c_t_r)
        services, contents = service_content_list[0], service_content_list[1]
        for i in range(users):  # uav上缓存了相应的服务或内容
            t_vu, t_uv = [], []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 1 and users_tasks[i][j][-1] in uav_cached_type_list:
                    t_vu.append(users_tasks[i][j][0] / v_u_t_r[i][j])
                    if users_tasks[i][j][-1] in uav_cached_type[0]:
                        t_uv.append(services[users_tasks[i][j][-1]][0] * 0.5 / self.U_V_transmission_rate)
                    else:
                        t_uv.append(contents[users_tasks[i][j][-1]][0] / self.U_V_transmission_rate)
                else:
                    t_uv.append(0)
                    t_vu.append(0)
            t_v_u_1.append(t_vu)  # 车辆任务卸载到UAV的传输时延
            t_u_v_1.append(t_uv)  # UAV上的计算结果返回到车辆的传输时延

        t_v_u_0, t_u_c_0, t_c_u_0, t_u_v_0 = [], [], [], []
        for i in range(users):  # uav上没有缓存相应的服务和内容，需要进一步卸载到中心云中去
            t_vu0, t_uc0, t_cu0, t_uv0 = [], [], [], []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 1 and users_tasks[i][j][-1] not in uav_cached_type_list:
                    t_vu0.append(users_tasks[i][j][0] / v_u_t_r[i][j])
                    t_uc0.append(users_tasks[i][j][0] / u_c_t_r)
                    t_cu0.append(users_tasks[i][j][0] * 0.6 / u_c_t_r)
                    t_uv0.append(users_tasks[i][j][0] * 0.6 / self.U_V_transmission_rate)
                else:
                    t_vu0.append(0)
                    t_uc0.append(0)
                    t_cu0.append(0)
                    t_uv0.append(0)
            t_v_u_0.append(t_vu0)
            t_u_c_0.append(t_uc0)
            t_c_u_0.append(t_cu0)
            t_u_v_0.append(t_uv0)

        t_v_u = (np.array(t_v_u_1) + np.array(t_v_u_0)).tolist()
        t_u_v = (np.array(t_u_v_1) + np.array(t_u_v_0)).tolist()
        t_u_c = t_u_c_0
        t_c_u = t_c_u_0

        print("t_v_u:", t_v_u)
        print("t_u_v:", t_u_v)
        print("t_c_u:", t_c_u)
        print("t_u_c:", t_u_c)

        # 计算任务从车辆卸载到UAV以及从UAV返回结果的总传输时延
        trans_t_vu = (np.array(t_v_u) + np.array(t_u_v)).tolist()
        # 计算任务从车到UAV到中心云再到UAV再到车辆的总传输时延
        trans_t_vuc = (np.array(t_v_u) + np.array(t_u_c) + np.array(t_c_u) + np.array(t_u_v)).tolist()

        print("trans_t_vu:", trans_t_vu)
        print("trans_t_vuc:", trans_t_vuc)

        return trans_t_vu, trans_t_vuc, t_v_u, t_u_v

    def get_uav_computing_time(self, users_tasks, services_list):
        # 任务在UAV上的计算时延
        # computing_capacity = random.randint(3, 10)*self.F_u/30
        users_tasks_comp_s = []
        for i in range(len(users_tasks)):
            user_tasks_comp = []
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 1 and users_tasks[i][j][-1] in services_list:
                    user_tasks_comp.append(users_tasks[i][j][1])
                else:
                    user_tasks_comp.append(0)
            users_tasks_comp_s.append(user_tasks_comp)

        uav_comp_t = []
        for i in range(len(users_tasks)):
            computing_capacity = random.randint(3, 10) * self.F_u / 30
            uav_comp_t.append((np.array(users_tasks_comp_s[i]) / computing_capacity).tolist())

        print("uav_computing_times:", uav_comp_t)

        return uav_comp_t

    def uav_rest(self):
        # 初始化UAV的位置
        return [50, 5]

    def request_frequency(self, users_tasks):
        request_num = [0 for i in range(40)]
        for user_tasks in users_tasks:
            for task in user_tasks:
                request_num[task[-1]] += 1
        print("request_num:", request_num)
        return request_num

    def get_uav_request_cache_action(self, action, services, contents, type_list, type_num, services_list, contents_list, rsu_cached_type_list, UAV_cache_memory, data_size, para, is_fill, request_num):
        # 看UAV需要对哪些已经缓存的服务和内容进行替换
        remain_action_uav_memory, services_type_list, contents_type_list, services_list, contents_list, data_size, para, is_fill = UAV().is_cached(action, services, contents, type_list, type_num, services_list, contents_list, rsu_cached_type_list, UAV_cache_memory, data_size, para, is_fill, request_num)
        uav_cached_type_list = [services_type_list, contents_type_list]

        return remain_action_uav_memory, uav_cached_type_list, services_list, contents_list, data_size, para, is_fill

    def is_cached(self, action, services, contents, type_list, type_num, services_list, contents_list, rsu_cached_type_list, UAV_cache_memory, data_size, para, is_fill, request_num):

        print("services:", services)
        print("contents:", contents)
        print("services_list:", services_list)
        print("contents_list:", contents_list)
        print("type_list:", type_list)

        # is_fill = False
        max_num = max(request_num)
        print("max_num:", max_num)
        cached_service_type, cached_content_type = type_list[0], type_list[1]
        rsu_cached_services_type, rsu_cached_contents_type = rsu_cached_type_list[0], rsu_cached_type_list[1]
        # data_size = []
        action.reverse()
        # action_ = action
        for action_ in action:
            p = np.random.uniform()
                # 选哪个服务或者内容进行缓存
            if request_num[action_] > 0 and (action_ not in cached_service_type) and (action_ not in cached_content_type):  # p <= request_num[action_]/max_num and
                if (UAV_cache_memory >= contents[action_][0]) and (action_ not in rsu_cached_services_type) and (action_ not in rsu_cached_contents_type) and services[action_][0] * 0.65 >= contents[action_][0] * 0.8:
                    contents[action_][3] = 1
                    UAV_cache_memory -= contents[action_][0]
                    # if UAV_cache_memory < contents[action_][0]:
                    #     is_fill = True
                    cached_content_type.append(action_)
                    contents_list.append(contents[action_])
                    # data_size += contents[action_][0]
                    data_size.append(contents[action_][0])
                # if (services[action_][0] < contents[action_][0]) and (UAV_cache_memory >= services[action_][0]):
                elif UAV_cache_memory >= services[action_][0] and action_ not in rsu_cached_contents_type:
                    services[action_][3] = 1
                    UAV_cache_memory -= services[action_][0]
                    # if UAV_cache_memory < services[action_][0]:
                    #     is_fill = True
                    cached_service_type.append(action_)
                    services_list.append(services[action_])
                    # data_size += services[action_][0]
                    data_size.append(services[action_][0])
                # elif (services[action_][0] >= contents[action_][0]) and (UAV_cache_memory >= contents[action_][0]) and (action_ not in rsu_contents_type):
                # elif p > 0.5 and (UAV_cache_memory >= contents[action_][0]) and (action_ not in rsu_cached_services_type) and (action_ not in rsu_cached_contents_type):
                #     contents[action_][3] = 1
                #     UAV_cache_memory -= contents[action_][0]
                #     # if UAV_cache_memory < contents[action_][0]:
                #     #     is_fill = True
                #     cached_content_type.append(action_)
                #     contents_list.append(contents[action_])
                #     # data_size += contents[action_][0]
                #     data_size.append(contents[action_][0])
                else:
                    data_size.append(0)
            else:
                data_size.append(0)

        print("UAV_cache_memory:", UAV_cache_memory)
        print("is_fill:", is_fill)
        if UAV_cache_memory < 100:
            is_fill = True

        # if UAV_cache_memory < services[action_][0] or UAV_cache_memory < contents[action_][0]:
        #     is_fill = True

        #######################################################################
        # for i in range(type_num):
        #     if action == i:  # 决定是否替换0类型服务或内容
        #         if i not in cached_service_type and i not in cached_content_type:
        #             if services[i][0] < contents[i][0] and self.UAV_cache_memory >= services[i][0] and np.random.uniform() < services[i][2]:
        #                 services[i][3] = 1
        #                 self.UAV_cache_memory -= services[i][0]
        #                 cached_service_type.append(i)
        #                 services_list.append(services[i])
        #             elif services[i][0] >= contents[i][0] and self.UAV_cache_memory >= contents[i][0] and np.random.uniform() < contents[i][2]:
        #                 contents[i][3] = 1
        #                 self.UAV_cache_memory -= contents[i][0]
        #                 cached_content_type.append(i)
        #                 contents_list.append(contents[i])
        #         elif i in cached_service_type and i not in cached_content_type:
        #             services[i][3] = 1
        #             self.UAV_cache_memory += services[i][0]
        #             cached_service_type.remove(i)
        #             services_list.remove(services[i])
        #             services[i][3] = 2
        #         elif i not in cached_service_type and i in cached_content_type:
        #             contents[i][3] = 1
        #             self.UAV_cache_memory += contents[i][0]
        #             cached_content_type.remove(i)
        #             contents_list.remove(contents[i])
        #             contents[i][3] = 2
        #         else:
        #             if services[i][0] < contents[i][0]:
        #                 services[i][3] = 2
        #                 self.UAV_cache_memory += services[i][0]
        #                 services_list.remove(services[i])
        #                 cached_service_type.remove(i)
        #             else:
        #                 contents[i][3] = 2
        #                 self.UAV_cache_memory += contents[i][0]
        #                 contents_list.remove(contents[i])
        #                 cached_content_type.remove(i)

        return UAV_cache_memory, cached_service_type, cached_content_type, services_list, contents_list, data_size, para, is_fill

    def get_origin_cache(self, services, contents):
        # 初始化进行随机缓存(不对UAV进行事先的缓存)
        uav_services, uav_contents = [[i for i in j]for j in services], [[i for i in j]for j in contents]
        uav_cached_type_list = []
        uav_services_list, uav_contents_list = [], []
        uav_service_cache_size = self.UAV_cache_memory * 0.5
        uav_cached_service_type = []
        for i in range(len(services)):
            uav_services[i][3] = np.random.choice((1, 2), size=1, p=[uav_services[i][2], 1-uav_services[i][2]])[0]  # 给需要缓存的服务打上标签，1表示缓存在UAV上，2表示不进行缓存
            if uav_services[i][3] == 1 and uav_service_cache_size >= uav_services[i][0]:
                uav_service_cache_size -= uav_services[i][0]
                uav_cached_service_type.append(uav_services[i][-1])
                uav_services_list.append(uav_services[i])
                uav_services_list[-1][3] = 1
                # uav_services[i][3] = 2
                index = self.memory_counter % CACHE_SPACE
                self.cache_memory[index, :] = uav_services[i]
                self.memory_counter += 1

        uav_cached_content_type = []
        uav_content_cache_size = self.UAV_cache_memory * 0.5
        for i in range(len(contents)):
            uav_contents[i][3] = np.random.choice((1, 2), size=1, p=[uav_contents[i][2], 1-uav_contents[i][2]])[0]  # 给需要缓存的内容打上标签，1表示缓存在UAV上，2表示不进行缓存
            if uav_contents[i][3] == 1 and uav_content_cache_size >= uav_contents[i][0]:
                uav_content_cache_size -= uav_contents[i][0]
                uav_cached_content_type.append(uav_contents[i][-1])
                uav_contents_list.append(uav_contents[i])
                uav_contents_list[-1][3] = 1
                # uav_contents[i][3] = 2
                index = self.memory_counter % CACHE_SPACE
                self.cache_memory[index, :] = uav_contents[i]
                self.memory_counter += 1

        print("cached_service_type:", uav_cached_service_type)
        print("cached_content_type:", uav_cached_content_type)

        self.UAV_cache_memory = (self.UAV_cache_memory - (self.UAV_cache_memory * 0.5 - uav_service_cache_size) - (self.UAV_cache_memory * 0.5 - uav_content_cache_size))

        for i in uav_cached_content_type:
            if i in uav_cached_service_type:
                uav_cached_service_type.remove(i)
                uav_services_list.remove(uav_services[i])
                self.UAV_cache_memory += uav_services[i][0]

        cached_type = uav_cached_service_type + uav_cached_content_type

        for i in range(len(services)):
            # 给需要缓存的服务打上标签，1表示缓存在UAV上，2表示不进行缓存
            if uav_services[i][2] >= 0.2 and self.UAV_cache_memory >= uav_services[i][0] and (uav_services[i][-1] not in cached_type):
                uav_services[i][3] = 1
                self.UAV_cache_memory -= uav_services[i][0]
                uav_cached_service_type.append(uav_services[i][-1])
                uav_services_list.append(uav_services[i])
                # uav_services[i][3] = 2
                index = self.memory_counter % CACHE_SPACE
                self.cache_memory[index, :] = uav_services[i]
                self.memory_counter += 1

        uav_cached_type_list.append(uav_cached_service_type)
        uav_cached_type_list.append(uav_cached_content_type)

        return uav_services_list, uav_contents_list, uav_cached_type_list

    def get_uav_reward(self, uav_total_t):  # uav_total_t:[[0, 0.3],[0, 0],[1.2, 0.0],[1.1, 0],...]
        # 计算奖励，奖励与计算成本、计算时延、通信成本、通信时延有关
        rewards = []
        # for i in range(len(uav_total_t)):
        #     reward = []
        #     for j in range(len(uav_total_t[i])):
        #         if uav_total_t[i][j] != 0:
        #             r = 1 / uav_total_t[i][j]
        #             reward.append(r)
        #         else:
        #             reward.append(0)
        #     rewards.append(sum(reward))

        #################################################
        for user_time in uav_total_t:
            reward = []
            for j in user_time:
                reward.append(-j)
            rewards.append(sum(reward))

        return rewards

    def get_end(self, cars_coord, rewards, users_tasks, total_time):  # 加上每个任务的服务约束
        # 判断服务是否结束终止
        # is_achieve_ = [[False, False], [False, False], [False, False], [False, False], [False, False]]
        # is_achieve_ = []
        # for i in range(len(users_tasks)):
        #     is_achieve_.append(False)
        d_vu = []
        for i in range(len(cars_coord)):
            d_vu.append(np.sqrt((cars_coord[i] - 5)**2))

        # is_achieve_ = [False, False, False, False, False]
        is_achieve = False
        uninstall_num = 0  # 卸载到UAV上的总任务数
        complete_num = 0  # 在UAV上被处理的总任务数
        for i in range(len(users_tasks)):
            for j in range(len(users_tasks[i])):
                if users_tasks[i][j][-2] == 1:
                    uninstall_num += 1
                if users_tasks[i][j][-2] == 1 and users_tasks[i][j][-3] == 1:  # and total_time[i][j] <= users_tasks[i][j][2] 缺少了时延约束条件
                    complete_num += 1

        # for i in range(len(users_tasks)):
        #     if complete_num == 0 and d_v_u[i] < self.UAV_service_r and n > 1:
        #         is_achieve_[i] = True
        #         rewards[i] += 2
        #
        #     elif d_v_u[i] >= self.UAV_service_r:
        #         is_achieve_[i] = True
        #         rewards[i] -= 2
        #
        #     else:
        #         rewards[i] -= 1
        #
        # if False not in np.array(is_achieve_):
        #     is_achieve = True
        is_achieve_uav = False
        # max_d = min(d_v_u[0])
        # for i in range(1, len(users_tasks)):
        #     if max_d < min(d_v_u[i]):
        #         max_d = min(d_v_u[i])

        # mean_d = []
        # for i in range(len(users_tasks)):
        #     mean_d.append(np.mean(d_v_u[i]))

        # for i in range(len(users_tasks)):
        if uninstall_num == complete_num and uninstall_num != 0:
            is_achieve = True
            # rewards = (np.array(rewards) + 1).tolist()  # 改变奖励设置
            rewards += 1

        # elif d_v_u[i] < self.UAV_service_r and np.array(users_tasks[i][:][-3]).all() == 1:

        elif min(d_vu) >= self.UAV_service_r and uninstall_num != complete_num:  # 所有车辆已经离开UAV的服务范围，并且未服务并未完成
            is_achieve_uav = True
            # rewards = (np.array(rewards) - 2).tolist()
            rewards -= 2

        else:
            rewards = (np.array(rewards) - 0).tolist()

        # if False not in np.array(is_achieve_):
        #     is_achieve = True

        return is_achieve, rewards, uninstall_num - complete_num, is_achieve_uav

    def get_uav_state(self, remain_tasks_num, remain_cache_memory, cars_coord):
        # 获取UAV的状态, 剩余需要处理的任务数量、剩余缓存空间、车到UAV之间的距离
        state = [remain_tasks_num/60]
        for i in range(len(cars_coord)):
            d_v_u = np.sqrt((cars_coord[i] - 5)**2)
            if d_v_u >= self.UAV_service_r:
                state.append(1)
            else:
                state.append(d_v_u/self.UAV_service_r)

        return state

    def get_data_save(self, t_total, reward, alpha, losses):  # 获取所有用户的计算时延、通信时延、总服务时延、奖励、缓存配比参数、用户编号
        # 获取计算成本、计算时延、通信成本、通信时延、总的服务时延、缓存参数配比、奖励、用户编号
        # my_data = pd.concat([pd.DataFrame(t_comp, columns=['t_comp']),
        #                      pd.DataFrame(t_trans, columns=['t_trans']),
        #                      pd.DataFrame(t_total, columns=['t_total']),
        #                      pd.DataFrame(reward, columns=['reward']),
        #                      pd.DataFrame(alpha, columns=['alpha'])],
        #                     axis=1)
        my_data = pd.concat([pd.DataFrame(t_total, columns=['times']),
                             pd.DataFrame(reward, columns=['reward']),
                             pd.DataFrame(alpha, columns=['alpha']),
                             pd.DataFrame(losses, columns=['losses'])],
                            axis=1)
        print(my_data)
        my_data.to_excel("UAV.xlsx", index=False)
        return

    def data_show(self, total_t, rewards, alphas, losses):
        mean_value = [total_t, rewards, alphas, losses]
        x_label = 'Episode'
        y_labels = ['time', 'reward', 'alpha', 'loss']
        fig, axs = plt.subplots(nrows=2, ncols=2, dpi=120)
        for i in range(4):
            axs[int(i / 2), i % 2].plot(range(len(mean_value[i])), mean_value[i], color="r")
            axs[int(i / 2), i % 2].set_xlabel(x_label)
            axs[int(i / 2), i % 2].set_ylabel(y_labels[i])
            axs[int(i / 2), i % 2].grid(linewidth=0.5, zorder=-10)
        plt.tight_layout()
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.show()
        return