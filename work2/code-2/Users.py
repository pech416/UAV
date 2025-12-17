# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
# from UAV_copy import UAV
from RSU_copy import RSU
import math

# matplotlib 正常显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

tasks = 10

class User(object):
    def __init__(self,
                 # state_space,
                 P_v_w=1,  # 车辆的传输功率(0.1W=20dbm)  0.1w~1w
                 N=0.01,  # 多传输之间的干扰
                 r=3,  # 传播参数
                 c=3 * (10 ** 8),  # 无线电波的传输速度（光速）
                 eta_LoS=1.6,  # 附加视距路径损失
                 eta_NLoS=23,  # 附加非视距路径损失
                 sigma2=1.26 * 10 ** (-8),  # 白噪声
                 users=5,  # 车辆用户的数量
                 user_speed=2  # 车辆的移动速度为5米每秒，为18千米每小时
                 ):
        # self.state_space = state_space
        # self.state_num = len(state_space)

        self.n_features = 1
        # 设置相关参数
        # 光速 300000 km/s
        self.c = c
        # 车辆的传输功率
        self.P_v_w = P_v_w
        # 基于环境的平均附加路径损耗值 (这里取 Dense Urban环境，单位dB)
        self.eta_LoS, self.eta_NLoS = eta_LoS, eta_NLoS
        # 传播参数
        self.r = r
        # 多传播之间的干扰
        self.N = N
        # 白噪声
        self.sigma2 = sigma2
        # 车辆用户的数量
        self.users = users
        # 车辆的移动速度
        self.user_speed = user_speed

    def get_generate_user(self, users, user_tasks_num):
        # 随机产生车辆用户，并得到其位置
        users_coord = []
        for i in range(users):
            user_tasks_coord = []
            for j in range(user_tasks_num):
                user_coord = [0, 0]  # 保存某个车辆用户的位置
                user_coord[0] = np.random.randint(-400, 400, 1)[0]  # 车辆用户位置的X
                user_coord[1] = np.random.randint(-5, 5, 1)[0]  # 车辆用户位置的Y

                user_tasks_coord.append(user_coord)
            users_coord.append(user_tasks_coord)

        return users_coord

    def get_user_coordinate_step(self, move_t, users_coord):  # 该move_t表示某一车辆用户总的传输时延和计算时延之和
        # 获取车辆用户的实时位置，并对其位置进行更新（只考虑车辆的横坐标移动，不考虑纵坐标移动）不考虑转弯
        user_move_t = []
        for move_t1 in move_t:
            user_move_t.append(sum(move_t1))

        users_coord_ = []
        for i in range(len(users_coord)):
            users_coord_.append(users_coord[i] + user_move_t[i] * self.user_speed * 2)
        # i = 0
        # users_coord_ = []
        # for user_coord in users_coord:
        #     if d_v_r[i] <= RSU().RSU_r:
        #         users_coord_.append([user_coord[0] + self.user_speed * user_move_t[i], user_coord[1]])
        #     else:
        #         users_coord_.append([user_coord[0], user_coord[1]])
        #     i += 1
        return users_coord_

    def get_users_tasks_coord(self, users_time, users_tasks_coord):

        users_tasks_coord_ = []
        for i in range(len(users_time)):
            user_tasks_coord_ = []
            for j in range(len(users_time[i])):
                d_v_r = np.sqrt((users_tasks_coord[i][j][0])**2 + (users_tasks_coord[i][j][1])**2)
                if d_v_r <= RSU().RSU_r:
                    user_tasks_coord_.append([users_tasks_coord[i][j][0] + self.user_speed * users_time[i][j], users_tasks_coord[i][j][1]])
                else:
                    user_tasks_coord_.append([users_tasks_coord[i][j][0], users_tasks_coord[i][j][1]])
            users_tasks_coord_.append(user_tasks_coord_)

        return users_tasks_coord_

        # users_coord_s = []
        # for i in range(len(move_t)):
        #     user_coord = []
        #     for j in range(len(move_t[i])):
        #         task_coord = [users_coord[i][j][0] + self.user_speed * move_t[i][j], users_coord[i][j][1]]
        #         user_coord.append(task_coord)
        #     users_coord_s.append(user_coord)

        # return users_coord

    def get_generate_request_task(self, users):
        # 每个车辆用户随机产生计算任务，计算任务包括任务的大小、对计算资源的需求、时延约束、编号(remove)、流行度、服务缓存参数、内容缓存参数
        # 一种服务类型的应用可以处理相关的大部分任务，而一种内容类型的应用只能处理单一类型的任务！！！！！！
        users_tasks = []  # 所有车辆用户的计算任务
        computing_resource_coefficient = 0.8  # 计算系数
        for i in range(users):
            user_tasks = []  # 某个车辆用户的全部任务数
            # tasks = np.random.randint(1, 2, 1)  # 某个车辆随机产生任务数

            for j in range(tasks):
                user_task = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 某个车辆用户的计算任务中的一个
                task_size = np.random.randint(20, 200, 1)[0]  # 计算任务的大小，单位为M
                task_computing_resource = task_size * computing_resource_coefficient  # 任务对计算资源的需求
                t_constraint = (0.4 * task_size * 0.6 * task_computing_resource) / (50 - 5)
                user_task[0], user_task[1], user_task[2] = task_size, task_computing_resource, t_constraint
                # user_task_no = 10 * i + j
                # user_task[3] = user_task_no
                user_task[3] = np.random.randint(0, 1000, 1)[0]  # 用任务的请求访问次数来表示任务的流行度
                user_task[4] = user_task[3] / 1000  # 表示该任务相关服务和内容的缓存概率
                user_task[5] = 1  # 表示车辆是否对某一任务的服务进行访问（是否访问跟是否卸载不是同一件事）
                user_task[6] = np.random.randint(0, 2, 1)[0]  # 来决定任务是卸载到RSU还是卸载到UAV（0卸载到RSU， 1卸载到UAV）
                user_task[7] = 0  # 表示以流行度概率来选取该任务相对应的服务缓存参数
                user_task[8] = 0  # 表示以流行度概率来选取该任务相对应的内容缓存参数

                user_tasks.append(user_task)

            users_tasks.append(user_tasks)

        return users_tasks

    def get_generate_request_task_copy(self, users):
        # 每个车辆用户随机产生计算任务，计算任务包括任务的大小、所需计算资源、时延约束、访问次数、访问概率、卸载位置、任务类型
        users_tasks = []
        computing_resource_coefficient = 0.5  # 计算资源系数
        req_num = 1000  # 最大请求数
        service_content_type = 40
        for i in range(users):
            user_tasks = []

            for j in range(tasks):
                user_task = [0, 0, 0, 0, 0, 0, 0]
                task_size = np.random.randint(80, 160, 1)[0]  # 计算任务的大小
                task_computing_resource = task_size * computing_resource_coefficient  # 任务对于计算资源的需求
                task_t_constraint = (0.15 * task_size * task_computing_resource) / 1000  # 任务的时延约束
                # task_t_constraint = 0.8
                user_task[0], user_task[1], user_task[2] = task_size, task_computing_resource, task_t_constraint
                user_task[3] = np.random.randint(10, req_num, 1)[0]  # 任务访问次数（应该不需要）
                user_task[3] = 3  # 0RSU 1AUV 2 cloud
                # user_task[4] = user_task / req_num  # 任务的访问概率（应该不需要）
                user_task[4] = 0  # 表示任务是否处理完成，0表示没有被处理完，1表示已经被处理完
                user_task[5] = 2  # 卸载位置，0为卸载到RSU上去，1为卸载到UAV上去
                user_task[6] = np.random.randint(0, service_content_type, 1)[0]  # 任务类型

                user_tasks.append(user_task)

            users_tasks.append(user_tasks)

        return users_tasks

    def get_tasks_step(self, users_tasks):
        for i in range(len(users_tasks)):
            for j in range(len(users_tasks[i])):
                users_tasks[i][j][3] = 3
                users_tasks[i][j][4] = 0
                users_tasks[i][j][5] = 2

        return users_tasks

    def get_trans_time(self, users_tasks, v_u_r, v_r_r):
        t_v_u, t_v_r = [], []
        for i in range(len(users_tasks)):
            t_vu, t_vr = [], []
            for j in range(len(users_tasks[i])):
                t_vu.append(users_tasks[i][j][0] / v_u_r[i][j])
                t_vr.append(users_tasks[i][j][0] / v_r_r[i][j])
            t_v_u.append(t_vu)
            t_v_r.append(t_vr)

        return t_v_u, t_v_r

    def get_car_coord(self, users_coord):
        cars_coord = []
        for i in range(len(users_coord)):
            car_coord = []
            for j in range(len(users_coord[i])):
                car_coord.append(users_coord[i][j][0])
            cars_coord.append(min(car_coord))

        return cars_coord
