from env.environment_old import Environment
from time import time
from env.user_node import UserNode

class BaseMinMaxAlgorithm:
    def __init__(self, env: Environment, *args, **kwargs):
        self.algorithm_name = None

        self.env: Environment = env

        self.start_time = None
        self.end_time = None
        self.running_time = None    # 算法运行时间

        self.max_delay = None          # 最大交互时延
        self.final_avg_cost = None     # 系统平均开销
        self.target_value = None

        self.local_count = 0
        self.common_count = 0

        self.results = dict()       # 结果记录（运行时间，cost等）

        self.debug = True

    def get_running_time(self):
        if self.running_time is None:
            self.end_time = time()
            self.running_time = (self.end_time - self.start_time) * 1000    # ms
        return self.running_time

    def get_max_delay(self):
        self.max_delay = self.env.compute_max_interactive_delay(self.env.users)

    def get_final_avg_cost(self):
        self.final_avg_cost = self.env.compute_cost()

    def get_target_value(self):
        self.target_value, self.max_delay, self.final_avg_cost = self.env.compute_target_function_value("min-max")
        self.get_service_offloading_distribution()

    """
        获取服务分布
    """
    def get_service_offloading_distribution(self):
        on_local = 0
        on_common = 0
        for user in self.env.users:  # type: UserNode
            if self.env.sites[user.service_a.node_id].area_id == 0:
                on_common += 1
            else:
                on_local += 1

            if self.env.sites[user.service_r.node_id].area_id == 0:
                on_common += 1
            else:
                on_local += 1

        self.local_count = on_local
        self.common_count = on_common

        assert self.local_count + self.common_count == self.env.user_num * 2
        return self.local_count, self.common_count

    """
        查看各个用户的卸载和分配情况
    """
    def result_info(self):
        def get_site_area_id(site_id):
            site = self.env.sites[site_id]
            return site.area_id

        for user in self.env.users:     # type: UserNode
            sa_site_area = get_site_area_id(user.service_a.node_id)
            sr_site_area = get_site_area_id(user.service_r.node_id)
            print("[user{} - area{}] serv_A --> site #{}(area {}), num_server = {}, Tq = {}\n"
                  "                 serv_R --> site #{}(area {}), num_server = {}, Tq = {}".format(user.user_id, user.area_id,
                  user.service_a.node_id, sa_site_area, user.service_a.num_server, user.service_a.queuing_delay,
                  user.service_r.node_id, sr_site_area, user.service_r.num_server, user.service_r.queuing_delay))

    """
        开始运行，具体的逻辑由各算法类实现
    """
    def run(self):
        pass

    """
        获取最终的统计信息
    """
    def get_results(self):
        self.results["max_delay"] = self.max_delay
        self.results["cost"] = self.final_avg_cost
        self.results["target_value"] = self.target_value
        self.results["running_time"] = self.running_time
        self.results["local_count"] = self.local_count
        self.results["common_count"] = self.common_count
        return self.results
