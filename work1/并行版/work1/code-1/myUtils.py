import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import vehicle
import UAV
import myConfig
import math


def get_data_save(profits, reward, uavs_num, losses):  # 获取总的收益、奖励、无人机数量、损失
    my_data = pd.concat([pd.DataFrame(profits, columns=['profits']),
                         pd.DataFrame(reward, columns=['reward']),
                         pd.DataFrame(uavs_num, columns=['number']),
                         pd.DataFrame(losses, columns=['losses'])],
                        axis=1)
    print(my_data)
    my_data.to_excel \
        ("C:/Users/peich/Desktop/UAV/work1/并行版/work1/code-1/" + "data.xlsx",
                     index=False)
    return


def data_show(profits, rewards, uavs_num, losses):
    mean_value = [profits, rewards, uavs_num, losses]
    x_label = 'Episode'
    y_labels = ['profits', 'reward', 'uavs_num', 'loss']
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


# 初始化车辆用户群
def get_init_users(users_num, user_tasks_num):
    users = []
    users_tasks_types = []
    # user_tasks_num = 15
    for i in range(users_num):
        user = vehicle.User()
        user.get_user_init_position()
        user_types = user.make_task(user_tasks_num)
        for type in user_types:
            if type not in users_tasks_types:
                users_tasks_types.append(type)
        users.append(user)

    return users, users_tasks_types


# 初始化无人机群
def get_init_uav(uav_num, users_tasks_types):
    uavs = []
    for i in range(uav_num):
        cached_service_type = []
        cached_content_type = []
        # 总共100种服务和内容的类型
        # cached_service_type_index = 0
        # while cached_service_type_index < 30:
        #     temp_service_type = random.randint(1, 100)
        #     if temp_service_type not in cached_service_type:
        #         cached_service_type.append(temp_service_type)
        #         cached_service_type_index += 1
        #
        # cached_content_type_index = 0
        # while cached_content_type_index < 20:
        #     temp_content_type = random.randint(1, 100)
        #     if temp_content_type not in cached_content_type and temp_content_type not in cached_service_type:
        #         cached_content_type.append(temp_content_type)
        #         cached_content_type_index += 1

        service_number = (len(users_tasks_types) / (4 * uav_num)) + random.randint(0, 3)
        content_number = (len(users_tasks_types) / (4 * uav_num))
        # print("service_number", service_number)
        # print("content_number", content_number)
        # 在无人机上缓存service类型
        cached_service_type_index = 0
        while cached_service_type_index <= service_number:
            temp_service_index = random.randint(0, len(users_tasks_types) - 1)
            if users_tasks_types[temp_service_index] not in cached_service_type:
                cached_service_type.append(users_tasks_types[temp_service_index])
                cached_service_type_index += 1
        while cached_service_type_index > 0:
            temp_service_type = random.randint(0, vehicle.User().task_type_number)
            if temp_service_type not in cached_service_type:
                cached_service_type.append(temp_service_type)
                cached_service_type_index -= 1

        # 在无人机上缓存content类型
        cached_content_type_index = 0
        while cached_content_type_index < content_number:
            temp_content_index = random.randint(0, len(users_tasks_types) - 1)
            if users_tasks_types[temp_content_index] not in cached_content_type and users_tasks_types[temp_content_index] not in cached_service_type:
                cached_content_type.append(users_tasks_types[temp_content_index])
                cached_content_type_index += 1
        while cached_content_type_index > 0:
            temp_content_type = random.randint(0, vehicle.User().task_type_number)
            if temp_content_type not in cached_content_type and temp_content_type not in cached_service_type:
                cached_content_type.append(temp_content_type)
                cached_content_type_index -= 1

        uav = UAV.UAV(cached_service_type, cached_content_type)
        # 任意两个无人机之间的最小距离约束

        uav.get_init_uav()  # 对无人机进行初始化
        uavs.append(uav)

    return uavs

# 实现任意两个无人机之间的最小距离约束
def get_uav_min_distance_restraint(uavs, min_distance=400):
    # 无人机的最大飞行范围
    uav_max_fly_radius = uavs[0].uav_movable_range
    # 检查每个无人机之间的距离并进行调整
    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            distance, cloud_distance = uavs[i].get_uv_distance(uavs[j].position, myConfig.cloud_position)
            if distance < min_distance:
                overlap = min_distance - distance
                dx = overlap * (uavs[i].position[0] - uavs[j].position[0])
                dy = overlap * (uavs[i].position[1] - uavs[j].position[1])
                uavs[i].position[0] += dx / 2
                uavs[i].position[1] += dy / 2
                uavs[j].position[0] -= dx / 2
                uavs[j].position[1] -= dy / 2

        if abs(uavs[i].position[0]) > uav_max_fly_radius:
            uavs[i].position[0] = uav_max_fly_radius * (uavs[i].position[0] / abs(uavs[i].position[0]))
        if abs(uavs[i].position[1]) > uav_max_fly_radius:
            uavs[i].position[1] = uav_max_fly_radius * (uavs[i].position[1] / abs(uavs[i].position[1]))

# 计算无人机群的总成本
def get_uav_costs(uavs):
    total_costs = 0
    for uav in uavs:
        total_costs += uav.base_cost

    return total_costs


# 任意两个无人机之间的最小距离的约束
# 符合最小的距离的要求就返回true，不合符就返回false
def uav_min_distance_restraint(uav1, uav2):
    uav1_coord = uav1.position
    uav2_coord = uav2.position

    d_level_uav1_uav2 = math.sqrt((uav1_coord[0] - uav2_coord[0]) ** 2 + (uav1_coord[1] - uav2_coord[1]) ** 2)
    d_uav1_uav2 = d_level_uav1_uav2

    return d_uav1_uav2 >= myConfig.d_uav_min


# 计算每个无人机的服务范围内的用户数量
def get_uav_communication_range_user_number(uav, users):
    uav_user_number = 0
    for user in users:
        d_u_v, d_u_cloud = uav.get_uv_distance(user.position, myConfig.cloud_position)
        if uav.service_distance > d_u_v:
            uav_user_number += 1

    return uav_user_number


# 获取内层状态
def get_inter_layer_state(uavs, users):
    # 内层的状态初始化
    inter_layer_state = []
    users_number_covered = 0
    for uav in uavs:
        uav_state = []
        # uav_state的维度
        # 目前还没有对状态进行归一化，之后需要进行归一化
        uav_user_num = get_uav_communication_range_user_number(uav, users)
        users_number_covered += uav_user_num
        # uav_state.append(uav_user_num / (len(users) * 3 / len(uavs)))
        # uav_state.append(uav.position[0] / 5000)
        # uav_state.append(uav.position[1] / 5000)
        # uav_state.append(uav.base_cost / 30)  # 假设每个无人机的最大成本是500
        # uav_state.append(uav.profits / 80)  # 假设每个无人机的纯收益是10
        # uav_state.append(uav.tasked_number / (len(users) * 10))

        uav_state.append(uav_user_num)
        uav_state.append(uav.position[0])
        uav_state.append(uav.position[1])
        uav_state.append(uav.base_cost)
        uav_state.append(uav.profits)
        uav_state.append(uav.tasked_number)

        inter_layer_state.append(uav_state)
    ###############################################################################
    # inter_layer_state_temp = []
    # # 覆盖的总用户数
    # users_number_covered = 0
    # for uav in uavs:
    #     uav_state = []
    #     # uav_state的维度
    #     # 目前还没有对状态进行归一化，之后需要进行归一化
    #     uav_user_num = get_uav_communication_range_user_number(uav, users)
    #     uav.communication_range_user_number = uav_user_num
    #     users_number_covered += uav_user_num
    #     uav_state.append(1 * math.ceil(uav_user_num / len(users)))
    #     uav_state.append(uav.position[0] / 500)
    #     uav_state.append(uav.position[1] / 500)
    #     uav_state.append(uav.base_cost / 220)  # 假设每个无人机的最大成本是500
    #     uav_state.append(uav.profits / 100)  # 假设每个无人机的纯收益是10
    #     uav_state.append(math.ceil(uav.tasked_number / (len(users) * 1)))
    #
    #     inter_layer_state_temp.append(uav_state)
    #
    # inter_layer_state = [0 for i in range(6)]
    # for temp_state in inter_layer_state_temp:
    #     inter_layer_state[0] += temp_state[0]
    #     inter_layer_state[1] += temp_state[1]
    #     inter_layer_state[2] += temp_state[2]
    #     inter_layer_state[3] += temp_state[3]
    #     inter_layer_state[4] += temp_state[4]
    #     inter_layer_state[5] += temp_state[5]
    # for i in range(len(inter_layer_state)):
    #     inter_layer_state[i] /= len(inter_layer_state_temp)

    return inter_layer_state, users_number_covered

