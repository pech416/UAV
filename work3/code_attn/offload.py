import numpy as np
import math

import envs


class AttentionMechanism:
    ATTENTION_THRESHOLD = 0  # 注意力分数的阈值

    def __init__(self):
        self.W_qc1 = np.random.rand(2)  # c:通信，m：计算
        self.W_fc1 = np.random.rand(2)
        self.W_qc2 = np.random.rand(2)
        self.W_fc2 = np.random.rand(2)
        self.W_qc3 = np.random.rand(2)
        self.W_fc3 = np.random.rand(2)
        self.W_qm = np.random.rand(2)
        self.W_fm = np.random.rand(2)
        self.W_qs = np.random.rand(2)
        self.W_fs = np.random.rand(2)
        # 查询向量
        self.query = [0.5, 0.5]

    def query_update(self, task):
        data_min = 5e6
        data_max = 1e7
        compute_min = 5e8
        compute_max = 1e9
        data_normalized = (task.data - data_min) / (data_max - data_min)
        compute_normalized = (task.compute - compute_min) / (compute_max - compute_min)
        self.query = [data_normalized / (data_normalized + compute_normalized),
                      compute_normalized / (data_normalized + compute_normalized)]
        # self.query = [task.data, task.compute]

    def feature_mapping(self, value):
        return np.sqrt(value)

    def euclid_distance(self, p1, p2):
        d = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
        return d

    def additive_attention(self, query, feature, W_q, W_f):
        score = np.dot(query, W_q) + np.dot(feature, W_f)  # 查询向量与特征值的加权和
        score = np.sum(score)  # 确保score是一个标量
        return score

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def process_single_task(self, task, uavs):
        self.query_update(task)  # 针对当前任务更新query

        A_uavs = [uav for uav in uavs if task.service.service_id in [service.service_id for service in uav.uav_services]]
        A_scores = []

        if A_uavs:
            for uav in A_uavs:
                if uav.battery < 20:
                    A_uavs.remove(uav)

        # 计算A组无人机的注意力分数
        for uav in A_uavs:
            F_C1 = self.feature_mapping(uav.B)
            F_C2 = self.feature_mapping(uav.battery)  # 加入电量
            d = self.euclid_distance(task.position, uav.position)
            F_C3 = self.feature_mapping(d)
            F_M = self.feature_mapping(uav.compute)
            F_S = self.feature_mapping(uav.cache_space)  # 加入缓存空间大小

            scores_C1 = self.additive_attention(self.query, F_C1, self.W_qc1, self.W_fc1)
            scores_C2 = self.additive_attention(self.query, F_C2, self.W_qc2, self.W_fc2)
            scores_C3 = self.additive_attention(self.query, F_C3, self.W_qc3, self.W_fc3)
            scores_M = self.additive_attention(self.query, F_M, self.W_qm, self.W_fm)
            scores_S = self.additive_attention(self.query, F_S, self.W_qs, self.W_fs)

            scores = np.array([scores_C1, scores_C2, scores_C3, scores_M, scores_S])
            weights = self.softmax(scores)

            weighted_F_C1 = weights[0] * F_C1
            weighted_F_C2 = weights[1] * F_C2
            weighted_F_C3 = weights[2] * F_C3
            weighted_F_M = weights[3] * F_M
            weighted_F_S = weights[4] * F_S

            A_k = np.sum([weighted_F_C1, weighted_F_C2, weighted_F_C3, weighted_F_M, weighted_F_S])
            A_scores.append(A_k)

        # 如果A组为空，则标记该任务没有UAV有相应的服务缓存
        A_uavs = A_uavs if A_uavs else None  # uav对象

        return A_uavs, A_scores

    def where_offload_single_task(self, task, A_uavs, A_scores):
        if A_uavs:
            task.alpha = 2
        elif task.t_fixed < task.t_user:
            task.alpha = 1
        else:
            task.alpha = 0

        task.beta = 0 if A_uavs and len(A_uavs) == 1 else (1 if A_uavs else None)

        # 计算拆分权重
        if A_uavs:
            total_score = sum(A_scores)
            task.weight = [score / total_score for score in A_scores]
        else:
            task.weight = None


if __name__ == "__main__":
    service_num = 10
    services = envs.Service.generate_services_list(service_num)

    task_chain = envs.TaskChain(services)

    num_uav = 4
    uav_group = [envs.Uav(services) for _ in range(num_uav)]
    # print(f"uav_group:{uav_group}")

    fixed_wing = envs.FixedWing()  # 固定翼

    # 比较固定翼和用户的时延
    print("\n--------------比较时延--------------")
    for i, task in enumerate(task_chain.get_all_tasks()):
        task.compare_time_delay(fixed_wing, task.user)
        # print(f"task:{task}")
        print(f"fixed:{task.t_fixed},user:{task.t_user}")

    # 注意力机制
    print("\n--------------Attention mechanism:--------------")
    attention_mechanism = AttentionMechanism()
    for task in task_chain.tasks:
        A_uavs, A_scores = attention_mechanism.process_single_task(task, uav_group)  # 选出注意力小组和注意力分数
        task.compare_time_delay(fixed_wing, task.user)  # 比较固定翼与用户的时延
        attention_mechanism.where_offload_single_task(task, A_uavs, A_scores) 
        print(f"task id:{task.task_id}")
        print(f"A_uavs:{A_uavs}, \nA_scores:{A_scores}")
        print(f"α：{task.alpha}, beta:{task.beta}, weight:{task.weight}")