import random
import numpy as np

class User:
    def __init__(self,):
        # 车辆的开始位置
        self.position_init = 20000
        # 车辆用户的位置 [x, y]
        self.position = [np.random.randint(-self.position_init, self.position_init, 1)[0], np.random.randint(-self.position_init, self.position_init, 1)[0]]
        # 车辆用户卸载的任务 [type, size, CR, TR, EC]
        self.tasks = []
        # 车辆移动的速度 1.5m/s
        self.move_speed = 4
        # 车辆用户产生的任务类型数
        self.task_type_number = 400
        # 用户车辆的移动方向
        self.direction = random.randint(0, 3)

    # 产生车辆用户
    def get_user_init_position(self):
        user_coord = [0, 0]
        user_coord[0] = np.random.randint(-self.position_init, self.position_init, 1)[0]  # 车辆用户的初始位置x
        user_coord[1] = np.random.randint(-self.position_init, self.position_init, 1)[0]  # 车辆用户的初始位置y
        self.position = user_coord
        # print("user_position: ", self.position)

    def user_move(self, time):
        if self.direction == 0:
            self.position[1] += self.move_speed * time
        elif self.direction == 1:
            self.position[0] += self.move_speed * time
        elif self.direction == 2:
            self.position[1] -= self.move_speed * time
        else:
            self.position[0] -= self.move_speed * time

        # print("user_position: ", self.position)

    def make_task(self, task_num):
        tasks = []
        tasks_type = []
        for i in range(task_num):
            task_computing_resource = 0.8  # 计算资源系数

            type = random.Random().randint(0, self.task_type_number)
            tasks_type.append(type)
            size = random.Random().randint(100, 500)
            CR = size * task_computing_resource
            TR = (0.15 * size * task_computing_resource) / 100  # 任务的时延约束
            # 在计算之前还不知道处理当前任务的能耗是多少
            EC = size * 0.1
            # 任务是否处理完
            isValid = False

            task = [type, size, CR, TR, EC, isValid]
            # print("task:", task)
            tasks.append(task)

        self.tasks = tasks

        return tasks_type
