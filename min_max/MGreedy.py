import copy
import math

from min_max.base import BaseMinMaxAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
from env.service import Service

class MGreedyAlgorithm(BaseMinMaxAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinMaxAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "MGreedy" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only  # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.debug_flag = False

        self.selected_sites = []
        self.candidate_sites = set([i for i in range(self.env.site_num)])
        self.assignment = dict()
        for u in range(self.env.user_num):
            self.assignment[u] = None

        self.M = self.env.site_num      # fixme

    def run(self):
        self.start_time = time()

        self.solve()
        # self.check_result()

        self.get_running_time()
        self.get_target_value()
        # self.result_info()

        self.DEBUG("max_delay = {}".format(self.max_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))

    def solve(self):
        max_delay = math.inf

        while len(self.selected_sites) < self.M:
            self.DEBUG("max_delay = {}".format(max_delay))
            best_assignment = None          # 本轮的最佳assignment
            s_star = None
            for s in self.candidate_sites:
                # 添加site之后，用户离它更近则关联这个site
                temp_assignment = copy.deepcopy(self.assignment)
                for u, assign in temp_assignment.items():
                    if temp_assignment[u] is None or self.env.tx_user_node[u][assign] > self.env.tx_user_node[u][s]:
                        temp_assignment[u] = s

                # 计算 max-T
                cur_max_delay, _ = self.compute_max_delay(temp_assignment)

                if cur_max_delay < max_delay:
                    max_delay = cur_max_delay
                    best_assignment = copy.deepcopy(temp_assignment)
                    s_star = s

            if s_star is not None:
                # print("s_star = {}".format(s_star))
                self.selected_sites.append(s_star)
                self.candidate_sites.remove(s_star)
                self.assignment = copy.deepcopy(best_assignment)
            else:
                break

        for u, s in self.assignment.items():
            user = self.env.users[u]
            site = self.env.sites[s]

            user.service_a.node_id = site.global_id
            user.service_a.service_rate = site.service_rate_a
            user.service_a.price = site.price_a

            user.service_r.node_id = site.global_id
            user.service_r.service_rate = site.service_rate_r
            user.service_r.price = site.price_r

        # 资源分配
        if not self.consider_cost_tq:
            for user in self.env.users:   # type: UserNode
                user.service_a.num_server = 0
                user.service_a.queuing_delay = 0
                user.service_r.num_server = 0
                user.service_r.queuing_delay = 0
        else:
            if not self.stable_only:
                for user in self.env.users:     # type: UserNode
                    x_a = self.resource_allocation(user.service_a.arrival_rate, user.service_a.service_rate, self.env.eta, user.service_a.price)
                    user.service_a.update_num_server(x_a)

                    x_r = self.resource_allocation(user.service_r.arrival_rate, user.service_r.service_rate, self.env.eta, user.service_r.price)
                    user.service_r.update_num_server(x_r)
            else:
                for user in self.env.users:  # type: UserNode
                    num_a = user.service_a.get_num_server_for_stability(service_rate=user.service_a.service_rate)
                    user.service_a.update_num_server(num_a)

                    num_r = user.service_r.get_num_server_for_stability(service_rate=user.service_r.service_rate)
                    user.service_r.update_num_server(num_r)

    """
       计算一个某个服务的最佳数量
       输入：lambda_ = 到达率，miu_ = 服务率，eta，f = 服务的单价
       输出：服务的最佳个数
   """
    def resource_allocation(self, lambda_, miu_, eta, f):
        a = lambda_ / miu_
        eta_f = eta * f
        x = 0
        B = 1

        while x <= a:
            x += 1
            B = a * B / (x + a * B)
        C_old = x * B / (x - a + a * B)  # 排队时延

        while True:
            x = x + 1
            B = a * B / (x + a * B)
            C_new = x * B / (x - a + a * B)
            delta_t = C_old / ((x - 1 - a) * miu_) - C_new / ((x - a) * miu_)
            delta_t *= 1000
            if delta_t <= eta_f or (C_new * 1000 <= 1.0):
                break

            C_old = C_new

        x_star = x - 1
        return int(x_star)

    def compute_max_delay(self, assignment: dict):
        max_delay = -math.inf
        u_pair = None
        for ui in range(self.env.user_num):
            for uj in range(ui + 1, self.env.user_num):
                d1 = self.env.tx_user_node[ui][assignment[ui]] * self.env.data_size[0]
                d2 = self.env.tx_node_node[assignment[ui]][assignment[uj]] * self.env.data_size[1]
                d3 = self.env.tx_node_user[assignment[uj]][uj] * self.env.data_size[2]
                delay = d1 + d2 + d3
                if delay > max_delay:
                    max_delay = delay
                    u_pair = (ui, uj)
        return max_delay, u_pair


    def check_result(self):
        for u, s in self.assignment.items():
            print("client {}(area {})---> site {}(area {})".format(u, self.env.users[u].area_id, s, self.env.sites[s].area_id))

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)

if __name__ == "__main__":
    from env.environment import Environment
    from configuration.config import config as conf

    import random

    u_seed = random.randint(0, 10000000000)
    # u_seed = 4932794917
    print("user_seed = {}".format(u_seed))

    env_seed = 99497
    print("env_seed: ", env_seed)

    num_user = 50

    print("----------------------------------------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    mg = MGreedyAlgorithm(env)
    mg.debug_flag = True
    mg.M = 4
    mg.run()
    print(mg.get_results())

    print("----------------------------------------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    mg = MGreedyAlgorithm(env)
    mg.debug_flag = True
    mg.M = 8
    mg.run()
    print(mg.get_results())

    print("----------------------------------------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    mg = MGreedyAlgorithm(env)
    mg.debug_flag = True
    mg.M = mg.env.site_num
    mg.run()
    print(mg.get_results())