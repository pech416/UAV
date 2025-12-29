# myUtils.py
# 优化说明: 本工具模块中函数均在合理范围内使用循环，无需额外优化
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import vehicle
import UAV
import math
import inter_layer_make_env

def get_data_save(profits, reward, uavs_num, losses):  # 获取总的收益、奖励、无人机数量、损失
    my_data = pd.concat([pd.DataFrame(profits, columns=['profits']),
                         pd.DataFrame(reward, columns=['reward']),
                         pd.DataFrame(uavs_num, columns=['number']),
                         pd.DataFrame(losses, columns=['losses'])],
                        axis=1)
    print(my_data)
    my_data.to_excel("data.xlsx", index=False)
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
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.show()
    return

# 初始化车辆用户群
def get_init_users(users_num, user_tasks_num):
    users = []
    users_tasks_types = []
    for i in range(users_num):
        user = vehicle.User()
        user.get_user_init_position()
        tasks_type = user.make_task(user_tasks_num)
        for t in tasks_type:
            if t not in users_tasks_types:
                users_tasks_types.append(t)
        users.append(user)
    return users, users_tasks_types

# 初始化无人机群
def get_init_uav(uav_num, users_tasks_types):
    uavs = []
    for i in range(uav_num):
        cached_service_type = []
        cached_content_type = []
        service_number = (len(users_tasks_types) / (4 * uav_num)) + random.randint(0, 3)
        content_number = (len(users_tasks_types) / (4 * uav_num))
        # 缓存 service 类型
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
        # 缓存 content 类型
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
        uav.get_init_uav()
        uavs.append(uav)
    return uavs

# Version 1
# 实现任意两个无人机之间的最小距离约束
# def get_uav_min_distance_restraint(uavs, min_distance=400):
#     uav_max_fly_radius = uavs[0].uav_movable_range
#     for i in range(len(uavs)):
#         for j in range(i + 1, len(uavs)):
#             distance, cloud_distance = uavs[i].get_uv_distance(uavs[j].position, inter_layer_make_env.cloud_position)
#             if distance < min_distance:
#                 overlap = min_distance - distance
#                 dx = overlap * (uavs[i].position[0] - uavs[j].position[0])
#                 dy = overlap * (uavs[i].position[1] - uavs[j].position[1])
#                 uavs[i].position[0] += dx / 2
#                 uavs[i].position[1] += dy / 2
#                 uavs[j].position[0] -= dx / 2
#                 uavs[j].position[1] -= dy / 2
#         if abs(uavs[i].position[0]) > uav_max_fly_radius:
#             uavs[i].position[0] = uav_max_fly_radius * (uavs[i].position[0] / abs(uavs[i].position[0]))
#         if abs(uavs[i].position[1]) > uav_max_fly_radius:
#             uavs[i].position[1] = uav_max_fly_radius * (uavs[i].position[1] / abs(uavs[i].position[1]))

#Version 2
def _enforce_min_distance_j_broadcast_block_gs(uav_positions, min_distance=400.0, max_range=20000.0,
                                              max_iter=3, alpha=1.0, eps=1e-6):
    """
    j 循环，i 向量化（i = 0..j-1），Block-GS：每个 j 直接就地更新 pos
    """
    pos = uav_positions.copy()
    n = pos.shape[0]

    for _ in range(max_iter):
        violated = 0

        for j in range(1, n):
            dx = pos[:j, 0] - pos[j, 0]
            dy = pos[:j, 1] - pos[j, 1]
            dist = np.hypot(dx, dy)

            mask = dist < min_distance
            if not np.any(mask):
                continue

            violated += int(mask.sum())

            dist_safe = np.where(dist > eps, dist, 1.0)
            ux = dx / dist_safe
            uy = dy / dist_safe
            ux = np.where(dist > eps, ux, 1.0)
            uy = np.where(dist > eps, uy, 0.0)

            overlap = (min_distance - dist)
            step = 0.5 * overlap
            delta_x = np.where(mask, step * ux, 0.0)
            delta_y = np.where(mask, step * uy, 0.0)

            # 直接更新
            pos[:j, 0] += alpha * delta_x
            pos[:j, 1] += alpha * delta_y
            pos[j, 0]  -= alpha * delta_x.sum()
            pos[j, 1]  -= alpha * delta_y.sum()

        pos[:, 0] = np.clip(pos[:, 0], -max_range, max_range)
        pos[:, 1] = np.clip(pos[:, 1], -max_range, max_range)

        if violated == 0:
            break

    return pos


def get_uav_min_distance_restraint(uavs, min_distance=400):
    if uavs is None or len(uavs) <= 1:
        return

        # 与工程变量对齐：飞行范围
    uav_max_fly_radius = float(uavs[0].uav_movable_range)

    # 组装位置矩阵 (N, 2)
    pos = np.empty((len(uavs), 2), dtype=np.float64)
    for idx, u in enumerate(uavs):
        pos[idx, 0] = float(u.position[0])
        pos[idx, 1] = float(u.position[1])

    # 执行约束（返回新位置矩阵）
    pos_new = _enforce_min_distance_j_broadcast_block_gs(
        pos,
        min_distance=float(min_distance),
        max_range=uav_max_fly_radius,
        max_iter=3,
        alpha=1.0,
        eps=1e-6
    )

    # 回写到 UAV 对象（就地更新）
    for idx, u in enumerate(uavs):
        u.position[0] = float(pos_new[idx, 0])
        u.position[1] = float(pos_new[idx, 1])


#Version 3
# def _enforce_min_distance_j_broadcast_jacobi(uav_positions, min_distance=400.0, max_range=20000.0,
#                                             max_iter=5, alpha=0.5, eps=1e-6):
#     """
#     j 循环，i 向量化（i = 0..j-1），Jacobi：先累加位移，再统一更新
#
#     与用户提供的 enforce_min_distance_j_broadcast_jacobi 版本一致（仅函数名调整为内部函数）。
#     """
#     pos = uav_positions.copy()
#     n = pos.shape[0]
#
#     for _ in range(max_iter):
#         disp = np.zeros((n, 2), dtype=pos.dtype)
#         violated = 0
#
#         for j in range(1, n):
#             # i 向量：0..j-1
#             dx = pos[:j, 0] - pos[j, 0]          # (j,)
#             dy = pos[:j, 1] - pos[j, 1]          # (j,)
#             dist = np.hypot(dx, dy)              # (j,)
#
#             mask = dist < min_distance
#             if not np.any(mask):
#                 continue
#
#             violated += int(mask.sum())
#
#             # 避免除 0；对重合点给确定性方向 (1,0)
#             dist_safe = np.where(dist > eps, dist, 1.0)
#             ux = dx / dist_safe
#             uy = dy / dist_safe
#             ux = np.where(dist > eps, ux, 1.0)
#             uy = np.where(dist > eps, uy, 0.0)
#
#             overlap = (min_distance - dist)      # (j,)
#             step = 0.5 * overlap
#             delta_x = step * ux
#             delta_y = step * uy
#
#             delta_x = np.where(mask, delta_x, 0.0)
#             delta_y = np.where(mask, delta_y, 0.0)
#
#             # i 批量加
#             disp[:j, 0] += delta_x
#             disp[:j, 1] += delta_y
#
#             # j 批量减（把所有 i 对 j 的作用累加起来）
#             disp[j, 0] -= delta_x.sum()
#             disp[j, 1] -= delta_y.sum()
#
#         if violated == 0:
#             break
#
#         pos[:, 0] += alpha * disp[:, 0]
#         pos[:, 1] += alpha * disp[:, 1]
#
#         pos[:, 0] = np.clip(pos[:, 0], -max_range, max_range)
#         pos[:, 1] = np.clip(pos[:, 1], -max_range, max_range)
#
#     return pos
#
# def get_uav_min_distance_restraint(uavs, min_distance=400):
#     """
#     最小距离约束（并行化/向量化版本）
#
#     - 保持函数名与工程一致：get_uav_min_distance_restraint(uavs, min_distance=400)
#     - 输入：uavs 为 UAV 对象列表，要求每个 uav.position 为 [x, y]
#     - 行为：就地更新 uavs[i].position，使任意两机之间尽量满足 distance >= min_distance
#     - 约束：位置裁剪到 [-uav_movable_range, uav_movable_range]（与工程原逻辑一致）
#
#     实现来自用户提供的 Jacobi（j 循环，i 向量化广播，先累加位移再统一更新）方法，默认最多迭代 5 次，alpha=0.5。
#     """
#     if uavs is None or len(uavs) <= 1:
#         return
#
#     # 与工程变量对齐：飞行范围
#     uav_max_fly_radius = float(uavs[0].uav_movable_range)
#
#     # 组装位置矩阵 (N, 2)
#     pos = np.empty((len(uavs), 2), dtype=np.float64)
#     for idx, u in enumerate(uavs):
#         pos[idx, 0] = float(u.position[0])
#         pos[idx, 1] = float(u.position[1])
#
#     # 执行约束（返回新位置矩阵）
#     pos_new = _enforce_min_distance_j_broadcast_jacobi(
#         pos,
#         min_distance=float(min_distance),
#         max_range=uav_max_fly_radius,
#         max_iter=5,
#         alpha=0.5,
#         eps=1e-6
#     )
#
#     # 回写到 UAV 对象（就地更新）
#     for idx, u in enumerate(uavs):
#         u.position[0] = float(pos_new[idx, 0])
#         u.position[1] = float(pos_new[idx, 1])


# 计算无人机群的总成本
def get_uav_costs(uavs):
    total_costs = 0
    for uav in uavs:
        total_costs += uav.base_cost
    return total_costs

# 任意两个无人机之间的最小距离的约束
def uav_min_distance_restraint(uav1, uav2):
    uav1_coord = uav1.position
    uav2_coord = uav2.position
    d_level_uav1_uav2 = math.sqrt((uav1_coord[0] - uav2_coord[0]) ** 2 + (uav1_coord[1] - uav2_coord[1]) ** 2)
    return d_level_uav1_uav2 >= inter_layer_make_env.d_uav_min

# 计算每个无人机的服务范围内的用户数量
def get_uav_communication_range_user_number(uav, users):
    uav_user_number = 0
    for user in users:
        d_u_v, d_u_cloud = uav.get_uv_distance(user.position, inter_layer_make_env.cloud_position)
        if uav.service_distance > d_u_v:
            uav_user_number += 1
    return uav_user_number

# 获取内层状态 (带归一化)
def get_inter_layer_state(uavs, users):
    """获取内层状态 (带归一化)

    优化点（CPU 加速，结果保持一致）：
    - 原实现：对每个 UAV 遍历全部 users，并在循环内调用 uav.get_uv_distance（包含多次 sqrt / power）
      时间复杂度 O(N_uav * N_user) 且 Python 循环开销大。
    - 新实现：使用 NumPy 广播一次性计算所有 UAV-User 的 3D 距离矩阵，再按 service_distance 计数覆盖用户数。
      数值逻辑与原来一致（使用同样的严格 '>' 比较），输出 state 与 users_number_covered 保持一致。
    """
    if len(uavs) == 0:
        return [], 0

    n_uav = len(uavs)
    n_user = len(users)

    uav_pos = np.asarray([u.position for u in uavs], dtype=np.float64)  # (n_uav, 3)
    if n_user > 0:
        user_pos = np.asarray([u.position for u in users], dtype=np.float64)  # (n_user, 2)
        dx = uav_pos[:, 0:1] - user_pos[:, 0].reshape(1, -1)
        dy = uav_pos[:, 1:2] - user_pos[:, 1].reshape(1, -1)
    else:
        user_pos = np.zeros((0, 2), dtype=np.float64)
        dx = np.zeros((n_uav, 0), dtype=np.float64)
        dy = np.zeros((n_uav, 0), dtype=np.float64)

    # 与 UAV.get_uv_distance 内部计算顺序保持一致
    temp1 = np.power(dx, 2) + 0.0
    temp2 = np.power(dy, 2) + 0.0
    d_level = np.sqrt(temp1 + temp2)  # (n_uav, n_user)
    z2 = np.power(uav_pos[:, 2], 2).reshape(-1, 1)
    d_u_v = np.sqrt(np.power(d_level, 2) + z2)  # (n_uav, n_user)

    service_dist = np.asarray([u.service_distance for u in uavs], dtype=np.float64).reshape(-1, 1)
    if n_user > 0:
        covered_counts = (service_dist > d_u_v).sum(axis=1).astype(np.int64)
    else:
        covered_counts = np.zeros((n_uav,), dtype=np.int64)

    users_number_covered = int(covered_counts.sum())

    denom_users = (n_user + 1e-5)
    uav_movable = np.asarray([u.uav_movable_range for u in uavs], dtype=np.float64)
    base_cost = np.asarray([u.base_cost for u in uavs], dtype=np.float64)
    profits = np.asarray([u.profits for u in uavs], dtype=np.float64)
    tasked = np.asarray([u.tasked_number for u in uavs], dtype=np.float64)

    state = np.empty((n_uav, 6), dtype=np.float64)
    state[:, 0] = covered_counts / denom_users
    state[:, 1] = uav_pos[:, 0] / uav_movable
    state[:, 2] = uav_pos[:, 1] / uav_movable
    state[:, 3] = base_cost / 100.0
    state[:, 4] = profits / 100.0
    state[:, 5] = tasked / (n_user * 1.0 if n_user > 0 else 1.0)

    return state.tolist(), users_number_covered
