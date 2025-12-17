# coding=utf-8
import numpy as np


class Provider(object):
    TYPE_NUM = 40

    def get_service(self):
        # 产生并提供服务，信息包括服务的大小、访问次数、访问概率、是否进行缓存参数、是否已经缓存了相同类型的内容、类型
        service_nums = 40
        type_the_number = 100
        req_num = 1000  # 最大请求数
        services = []
        for i in range(service_nums):
            service = [0, 0, 0, 0, 0, 0]
            service[0] = np.random.randint(80, 200, 1)[0]  # 服务的大小
            service[1] = np.random.randint(100, req_num, 1)[0]  # 服务的访问次数
            service[2] = service[1] / req_num
            service[3] = 2  # 是否进行缓存，0表示缓存在RSU，1表示缓存在UAV，2表示都没缓存
            service[4] = 1  # 表示服务
            service[5] = i  # 服务类型

            services.append(service)

        return services

    def get_content(self, services):
        # 产生并提供内容，信息包括内容的大小、访问次数、访问概率、是否进行缓存参数、是否已经缓存了相同类型的服务、类型
        content_nums = 40
        type_the_number = 100
        req_num = 1000  # 最大请求数
        contents = []
        for i in range(content_nums):
            content = [0, 0, 0, 0, 0, 0]
            content[0] = np.random.randint(60, services[i][0] - 19, 1)[0]  # 服务的大小
            content[1] = np.random.randint(100, req_num, 1)[0]  # 服务的访问次数
            content[2] = content[1] / req_num
            content[3] = 2  # 是否进行缓存，0表示缓存在RSU，1表示缓存在UAV，2表示都没缓存
            content[4] = 0  # 表示内容
            content[5] = i  # 服务类型

            contents.append(content)

        return contents

