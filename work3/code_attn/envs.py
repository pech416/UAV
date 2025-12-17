import numpy as np
import random
from random import randint
import copy
import math


class Task:
    def __init__(self, task_id, service, data, compute, tau, user):
        self.task_id = task_id
        self.service = service
        self.data = data
        self.compute = compute
        self.tau = tau
        self.user = user  # 任务关联的用户
        self.position = user.position  # 任务的位置与用户的位置相同
        self.task_done = False  # 任务是否已执行

        self.alpha = None  # 卸载因子α
        self.beta = None  # 拆分因子β
        self.weight = None  # 任务拆分权重weigh

        # 用于卸载决策前比较时延
        self.t_fixed = 0
        self.t_user = 0

        # 用于保存任务开始前uav的状态
        self.initial_battery = {}
        self.initial_cache = {}

    def compare_time_delay(self, fixed_wing, user):
        """比较任务在固定翼和本地的完成时延。"""
        d = self.euclid_distance(self.position, fixed_wing.position)  # 距离
        n = 3  # 路径损耗指数
        r = fixed_wing.B * np.log2(1 + (self.user.p0 * (d ** -n)) / fixed_wing.N)
        t_transmit = self.data / r  # 传输时延
        t_compute_fixed = self.compute / fixed_wing.compute  # 计算时延
        self.t_fixed = t_transmit + t_compute_fixed
        self.t_user = self.compute / user.compute

    def euclid_distance(self, p1, p2):
        d = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
        return d

    def __repr__(self):
        return f"Task ID: {self.task_id}, Service: {self.service}, Data: {self.data}, Compute: {self.compute}, Tau: {self.tau}, Position: {self.position}"


class Service:
    service_id = 0

    @classmethod
    def reset_id(cls):
        cls.service_id = 0  # 重置service_id为0

    @classmethod
    def get_next_id(cls):
        current_id = cls.service_id
        cls.service_id += 1
        return current_id

    def __init__(self):
        # 0:服务id、1:服务大小、2:是否进行缓存、(3:服务访问次数、4:访问频率)
        self.service_id = Service.get_next_id()
        self.service_size = np.random.randint(2, 5)
        self.cache_status = 0  # 1缓存，0不缓存
        self.access = 0  # 初始化访问次数为0。当卸载到uav的任务需要该服务时，access + 1. todo

    def __repr__(self):
        return f"Service ID: {self.service_id}, Size: {self.service_size}, Cache: {self.cache_status}, Access:{self.access}"

    @staticmethod
    def generate_services_list(service_num):  # 输入：服务总数量，返回：服务列表
        Service.reset_id()  # 每次生成services_list之前重置service_id
        services = []
        for i in range(service_num):
            service = Service()  # 创建新服务，其属性将自动随机生成
            services.append(service)
        return services


class TaskChain:
    def __init__(self, services):  # 接受服务列表作为参数
        self.task_num = 20  # 任务链的任务个数

        self.services = random.sample(services, self.task_num)  # 从服务列表中随机选择，不可重复选择
        # self.services = random.choices(services, k=self.task_num)  # 可重复选择

        # 创建用户列表
        self.users = [User() for _ in range(self.task_num)]
        # 创建任务链中的任务，每个任务分配一个用户和一个服务
        self.tasks = [self._create_task(i, user, service) for i, (user, service) in enumerate(zip(self.users, self.services))]
        self.current_task_index = 0  # 当前任务的索引

    def _create_task(self, i, user, service):
        tau = np.random.uniform(2.5, 3)  # 最大处理时间:1s~10s
        data = np.random.randint(5e6, 1e7)  # 1Mbit~10Mbits, (bit)
        compute = np.random.randint(5e8, 1e9)  # 计算能力要求:CPU的计算频率。任务复杂度：1e8:100M周期，1e9:1G周期.100MHz-1GHz,(Hz)
        return Task(user.user_id, service, data, compute, tau, user)

    def get_current_task(self):
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        else:
            return None

    def get_all_tasks(self):
        return self.tasks

    def __repr__(self):
        representation = f"TaskChain with {self.task_num} tasks:"
        for i, task in enumerate(self.tasks):
            representation += f"\nTask ID: {task.task_id}, "
            representation += f"User ID: {task.user.user_id}, "
            representation += f"Service: {task.service}, "
            representation += f"Data: {task.data}, "
            representation += f"Compute: {task.compute}, "
            representation += f"Tau: {task.tau}, "
            representation += f"Position: {task.position}"
        return representation


class User:
    user_id = 0

    @classmethod
    def get_next_id(cls):
        current_id = cls.user_id
        cls.user_id += 1
        return current_id

    def __init__(self):
        self.user_id = User.get_next_id()
        # self.p0 = np.random.randint(5, 20)  # 用户的发射功率：5~20dBm
        self.p0 = np.random.uniform(0.5, 1.2)  # 设备的发射功率 1.2瓦特
        # self.p0 = 1.0
        self.position = (np.random.uniform(0, 1000), np.random.uniform(0, 1000), 0)  # 用户始终在地面上，所以z = 0
        self.compute = 1e8  # UE的计算能力：3GHz。  任务复杂度：1e8:100M周期，1e9:1G周期 todo

    def __repr__(self):
        return f"user_id:{self.user_id}, p0: {self.p0}, position: {self.position}, compute: {self.compute}"


class Uav:
    uav_id = 0

    @classmethod
    def get_next_uavid(cls):
        result = cls.uav_id
        cls.uav_id += 1
        return result

    def __init__(self, services):  # services:可用的服务列表
        self.uav_id = Uav.get_next_uavid()
        self.B = np.random.uniform(0.8e5, 1.2e5)  # 通信信道的带宽 1.2MHz
        # self.B = 1.0e5
        N0 = -174  # 噪声功率密度，单位 dBm/Hz
        N0_linear = 10 ** ((N0 - 30) / 10)  # 转换为线性单位 W/Hz
        self.N = N0_linear * self.B  # 总噪声功率，单位瓦特

        x, y = np.random.uniform(0, 1000, 2)  # 假设无人机可以在一个1000x1000的区域内移动
        z = np.random.uniform(50, 100)  # 高度范围
        self.position = (x, y, z)  # 位置坐标，为一个三元组 (x, y, z)
        self.battery = np.random.randint(200, 400)  # 电量为100kj~200kj
        # self.battery = 300

        # 确保传入的services列表被赋值给实例变量
        self.services = services

        # caching storage capacities
        # self.cache_space = np.random.uniform(50, 100)  # 缓存空间大小。每个任务(2, 5)，40个（80，200）
        self.cache_space = 70  # todo

        # computing capacities
        # self.compute = np.random.randint(1e9, 2e9)  # UAV的计算频率2GHz
        self.compute = 1.8e9  # todo

        # 初始化uav的缓存服务
        self.uav_services = []  # 初始化uav的缓存服务的具体信息 [Service ID: 0, Size: 8, Cache Status: 1, Access: 0]
        self.uav_services_list = [0] * len(self.services)  # uav缓存的服务的二进制列表 [1, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        # self.uav_services, self.uav_services_list = self.cache_init()

    def cache_init(self):
        self.services = copy.deepcopy(self.services)  # 创建服务列表的深度副本，以便每个UAV都有自己的副本
        for service in self.services:
            # cache_decision = np.random.choice((1, 0), size=1, p=[service.access_frequency, 1-service.access_frequency])  # 访问频率
            cache_decision = np.random.choice((1, 0), size=1, p=[0.5, 0.5])
            # 如果决定缓存该服务、缓存空间足够（缓存空间大于该服务所需空间）、该服务未缓存，则进行缓存
            if cache_decision == 1 and self.cache_space > service.service_size and service.cache_status == 0:
                self.uav_services.append(service)  # 将服务添加到当前uav的缓存列表中
                service.cache_status = 1  # 标记为已缓存
                self.cache_space -= service.service_size  # 减少可用的缓存空间
                self.uav_services_list[service.service_id] = 1  # 更新缓存二进制列表
        return self.uav_services, self.uav_services_list

    def euclid_distance(self, p1, p2):
        d = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
        return d

    # 任务到uav群的距离：卸载，时延
    def d_task_to_uav(self, task_position, uavs_position):  # uav坐标，任务坐标
        d_task_u = []  # 初始化一个列表存储从当前任务到每个UAV的直线距离的列表
        # 遍历每个uav的坐标
        for uav_position in uavs_position:
            d = self.euclid_distance(task_position, uav_position)
            d_task_u.append(d)
            # 将结果添加到uav集合中
        return d_task_u

    def __repr__(self):
        return f"\nUAV ID:{self.uav_id}, UAV Services:{self.uav_services}, UAV Position:{self.position}, UAV battery:{self.battery}"


class FixedWing:
    def __init__(self):
        self.B = 2.0e5
        N0 = -174  # 噪声功率密度，单位 dBm/Hz
        N0_linear = 10 ** ((N0 - 30) / 10)  # 转换为线性单位 W/Hz
        self.N = N0_linear * self.B  # 总噪声功率，单位瓦特
        x, y = np.random.uniform(0, 1000, 2)
        # z = np.random.uniform(200, 500)  # 更高的高度范围
        z = 1000
        self.position = (x, y, z)
        self.compute = 8e9  # UAV的计算频率9.0GHz


if __name__ == "__main__":
    service_num = 10
    services = Service.generate_services_list(service_num)  # 生成服务列表

    task_chain = TaskChain(services)  # 创建包含服务的任务链

    num_uav = 3
    uavs = [Uav(services) for _ in range(num_uav)]  # 这里应该为每个UAV创建一个新实例：Uav()创建实例

    # 测试uav缓存
    for uav_index, uav in enumerate(uavs):
        print(f"\nuav index:{uav_index}, services:{uav.uav_services}, services_list:{uav.uav_services_list}")
