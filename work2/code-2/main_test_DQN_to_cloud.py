import time

from UAV_copy import UAV
from RSU_copy import RSU
from Users import User
from DQN_multi_action import DQN
from Provider import Provider
import numpy as np
import PySimpleGUI as sg


dqn = DQN()

MEMORY_CAPACITY = 2560

episode_num = 1000  # 迭代次数
users = 6  # 车辆用户的数量
user_tasks_num = 10  # 每个车辆用户产生的任务数量

cloud_coord = [5000, 5000]  # 中心云的位置

startTime = time.time()
def run():
    alphas_ = []
    rewards_ = []
    total_t_ = []
    losses_ = []

    services = Provider().get_service()
    contents = Provider().get_content(services)

    users_tasks = User().get_generate_request_task_copy(users)

    request_num = UAV().request_frequency(users_tasks)
    print("request_num:", request_num)

    rsu_services, rsu_contents, rsu_cached_type_list, rsu_cached_memory = RSU().get_origin_cache(services, contents, request_num)

    print("总的service:", services)
    print("总的content:", contents)

    for i in range(episode_num):
        sg.one_line_progress_meter("dqn multi-option", i + 1, episode_num, orientation='h')
        print("episode counter : ", i)

        users_tasks = User().get_tasks_step(users_tasks)

        uav_coord = UAV().uav_rest()
        rsu_coord = RSU().rsu_coord
        users_coord = User().get_generate_user(users, user_tasks_num)
        cars_coord = User().get_car_coord(users_coord)
        # users_tasks = User().get_generate_request_task_copy(users)
        UAV_cache_memory = 1500  # UAV的缓存容量

        type_num = Provider.TYPE_NUM

        # rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
        rewards = 0
        alphas = []
        rewards_s = []
        total_ts = []
        losses = []

        # is_achieve_ = [False, False, False, False, False]
        n = 1

        # uav_services, uav_contents, uav_cached_type_list = UAV().get_origin_cache(services, contents)
        uav_services, uav_contents, uav_cached_type_list = [], [], [[], []]
        services_contents = [services, contents]
        # rsu_services, rsu_contents, rsu_cached_type_list = RSU().get_origin_cache(services, contents)

        # trans_vu_time, trans_vr_time = [], []

        trans_vu_time, trans_vr_time = [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], [
                                           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                           [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]
        data_size_s = []
        # data_size = 0
        # para = 0.5
        alpha = 0.5
        is_fill = False

        state = [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, alpha, 0]  # [UAV剩余需要处理的任务数， 剩余缓存空间，  车到UAV之间的距离， alpha， RSU剩余需要处理的任务数， 车到RSU之间的距离]
        state_ = state
        m = 0

        while True:
            uav_cached_service_type, uav_cached_content_type = uav_cached_type_list[0], uav_cached_type_list[1]  # 无人机缓存的服务和内容的类型
            rsu_cached_service_type, rsu_cached_content_type = rsu_cached_type_list[0], rsu_cached_type_list[1]  # RSU缓存的服务和内容的类型
            # cached_service_type = [uav_cached_service_type, rsu_cached_service_type]
            # cached_content_type = [uav_cached_content_type, rsu_cached_content_type]
            print("uav_cached_type_list:", uav_cached_type_list)

            uav_cached_type = [k for type in uav_cached_type_list for k in type]
            rsu_cached_type = [k for type in rsu_cached_type_list for k in type]

            actions = dqn.choose_action(state)
            UAV_cache_memory, uav_cached_type_list, uav_services, uav_contents, data_size_s, alpha, is_fill = UAV().get_uav_request_cache_action(actions, services, contents, uav_cached_type_list, type_num, uav_services, uav_contents, rsu_cached_type_list, UAV_cache_memory, data_size_s, alpha, is_fill, request_num)

            d_v_u, d_u_cloud = UAV().get_distance(uav_coord, users_coord, cloud_coord)  # 计算得到车辆到UAV之间的距离、UAV到中心云之间的距离
            d_v_r, d_r_cloud = RSU().get_distance(users_coord, cloud_coord)  # 计算车辆到RSU之间的距离、RSU到中心云之间的距离

            users_uav_trans_rate, uav_cloud_trans_rate = UAV().get_uav_transmission_rate(d_v_u, d_u_cloud, users)
            users_rsu_trans_rate, rsu_cloud_trans_rate = RSU().get_rsu_transmission_rate(d_v_r, d_r_cloud, users)

            trans_vu_time, trans_vr_time = User().get_trans_time(users_tasks, users_uav_trans_rate, users_rsu_trans_rate)

            users_tasks, n = UAV().where_unload(users_tasks, uav_cached_type, rsu_cached_type, trans_vu_time, trans_vr_time, n)

            trans_t_vu, trans_t_vuc, t_v_u, t_u_v = UAV().get_uav_transmission_time(users, users_uav_trans_rate, uav_cloud_trans_rate, users_tasks, uav_cached_type, services_contents, uav_cached_type_list)
            trans_t_vr, trans_t_vrc, t_v_r, t_r_v = RSU().get_rsu_transmission_time(users, users_rsu_trans_rate, rsu_cloud_trans_rate, users_tasks, rsu_cached_type, rsu_cached_type_list, services_contents)

            uav_t_computing = UAV().get_uav_computing_time(users_tasks, uav_cached_service_type)
            rsu_t_computing = RSU().get_rsu_computing_time(users_tasks, rsu_cached_service_type)

            trans_rsu = [trans_t_vr, trans_t_vrc, t_v_r, rsu_t_computing]
            trans_uav = [trans_t_vu, trans_t_vuc, t_v_u, uav_t_computing]

            users_total_t, uav_total_t, rsu_total_t = UAV().finish_tasks_time(users, users_tasks, uav_cached_type_list, rsu_cached_type_list, trans_rsu, trans_uav)

            total_time = (np.array(uav_total_t) + np.array(rsu_total_t)).tolist()

            t_back_vehicle = (np.array(t_u_v) + np.array(t_r_v)).tolist()  # 任务从RSU或UAV上返回给车辆所需要的时延
            t_back_result_vehicle = (np.array(total_time) - np.array(t_back_vehicle)).tolist()  # 在任务从车辆开始卸载到计算完成在RSU或者UAV上准备返回给车辆时所花费的时延

            users_coord = User().get_users_tasks_coord(t_back_result_vehicle, users_coord)
            cars_coord = User().get_user_coordinate_step(total_time, cars_coord)

            d_v_u, d_u_cloud = UAV().get_distance(uav_coord, users_coord, cloud_coord)  # 计算得到车辆到UAV之间的距离、UAV到中心云之间的距离
            d_v_r, d_r_cloud = RSU().get_distance(users_coord, cloud_coord)  # 计算车辆到RSU之间的距离、RSU到中心云之间的距离
            print("users_tasks__:", users_tasks)
            for ii in range(users):
                for jj in range(len(users_tasks[ii])):
                    if users_tasks[ii][jj][2] >= uav_total_t[ii][jj] > 0 and users_tasks[ii][jj][-2] == 1 and d_v_u[ii][jj] < UAV().UAV_service_r and users_tasks[ii][jj][3] == 1:  # 此时任务车辆必须还在UAV的服务范围内才能将计算结果返回给车辆
                        users_tasks[ii][jj][-3] = 1
                    elif users_tasks[ii][jj][2] >= rsu_total_t[ii][jj] > 0 and users_tasks[ii][jj][-2] == 0 and d_v_r[ii][jj] < RSU().RSU_r and users_tasks[ii][jj][3] == 0:  # 此时任务车辆必须还在RS的服务范围内才能将计算结果返回给车辆
                        users_tasks[ii][jj][-3] = 1

            to_cloud_users_tasks_nums = 0
            for ii in range(users):
                for jj in range(len(users_tasks[ii])):
                    if users_tasks[ii][jj][3] == 2:
                        to_cloud_users_tasks_nums += 1

            # uav_rewards = UAV().get_uav_reward(uav_total_t)
            # rsu_rewards = RSU().get_rsu_reward(rsu_total_t)
            uav_rewards = -to_cloud_users_tasks_nums
            rsu_rewards = 0

            uav_is_achieve, uav_rewards, uav_remain_tasks_num, is_achieve_uav = UAV().get_end(cars_coord, uav_rewards, users_tasks, total_time)
            rsu_is_achieve, rsu_rewards, rsu_remain_tasks_num, is_achieve_rsu = RSU().get_end(cars_coord, rsu_rewards, users_tasks, total_time)

            uav_state = UAV().get_uav_state(uav_remain_tasks_num, UAV_cache_memory, cars_coord)
            rsu_state = RSU().get_rsu_state(rsu_remain_tasks_num, cars_coord)

            alpha = UAV().init_ratio_parameter(uav_services, uav_contents)
            state_ = np.hstack((uav_state, rsu_state, [alpha], [-uav_rewards/20])).astype('float32').reshape([16, ])
            # r = (np.array(uav_rewards) + np.array(rsu_rewards))
            rewards = uav_rewards

            dqn.store_transition(state, actions, rewards, state_)

            users_coord = User().get_users_tasks_coord(t_back_vehicle, users_coord)

            if dqn.memory_counter > MEMORY_CAPACITY:
                loss = dqn.learn()
                print("loss:", loss)
                losses.append(loss)
            state = state_
            alphas.append(alpha)
            # alpha = UAV().init_ratio_parameter(uav_services, uav_contents)

            # rewards_s.append(rewards)
            # user_t = []
            # for user_time in total_time:
            #     user_t.append(sum(user_time))
            # total_ts.append(sum(user_t))

            if uav_is_achieve or is_achieve_uav:  # 如果回合结束, 进入下回合
                # if (uav_is_achieve and rsu_is_achieve) or is_achieve_uav or is_achieve_rsu or m > 3:
                rewards_s.append(rewards)
                user_t = []
                for user_time in total_time:
                    user_t.append(sum(user_time))
                total_ts.append(sum(user_t))
                break
            # users_tasks = UAV().where_unload(users_tasks, uav_cached_type, rsu_cached_type, t_v_u, t_v_r)
            # if UAV_cache_memory < 60:
            #     break

        print("alpha:", alpha)

        alphas_.append(np.mean(alphas))
        rewards_.append(rewards_s)
        total_t_.append(total_ts)
        losses_.append(np.mean(losses))

    # 每200次迭代求一次均值！！！！！！！改变奖励惩罚设置！！！
    alphas_mean = []
    rewards_mean = []
    total_t_mean = []
    losses_mean = []
    nums = 200
    for i in range(episode_num - nums):
        alphas_mean.append(np.mean(alphas_[i:nums + i]))
        rewards_mean.append(np.mean(rewards_[i:nums + i]))
        total_t_mean.append(np.mean(total_t_[i:nums + i]))

    UAV().get_data_save(total_t_, rewards_, alphas_, losses_)
    UAV().data_show(total_t_mean, rewards_mean, alphas_mean, losses_)


if __name__ == "__main__":
    run()

