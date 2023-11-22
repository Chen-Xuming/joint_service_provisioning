"""
论文：Analysis of Server Provisioning for Distributed Interactive Applications

相当于 M-Greedy 的 min-avg 版本
"""

import copy
import math

from min_avg.base import BaseMinAvgAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
from env.service import Service
import numpy as np

class GreedyServerProvisioningAlgorithm(BaseMinAvgAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, avg_t_compositions=0b100, *args, **kwargs):
        BaseMinAvgAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "GSP" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

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

        self.n_2 = self.env.user_num ** 2

        # 计算 max-T 过程中考虑哪些成分（Tx, Tp, Tq）
        # 100: Tx = 1, Tp = 0, Tq = 0   # 只考虑Tx
        self.T_flags = [False for i in range(3)]
        self.T_flags[0] = bool(avg_t_compositions & 4)
        self.T_flags[1] = bool(avg_t_compositions & 2)
        self.T_flags[2] = bool(avg_t_compositions & 1)
        self.DEBUG("Tx = {}, Tp = {}, Tq = {}".format(self.T_flags[0], self.T_flags[1], self.T_flags[2]))

        if self.T_flags[1]:
            self.tp_serviceA = np.zeros(self.env.site_num)      # ms
            self.tp_serviceR = np.zeros(self.env.site_num)      # ms

        if self.T_flags[2]:
            self.best_x_serviceA = np.zeros((self.env.user_num, self.env.site_num))
            self.best_x_serviceR = np.zeros((self.env.user_num, self.env.site_num))
            self.tq_serviceA = np.zeros((self.env.user_num, self.env.site_num))         # ms
            self.tq_serviceR = np.zeros((self.env.user_num, self.env.site_num))         # ms

    def run(self):
        self.start_time = time()

        self.solve()
        # self.check_result()

        self.get_running_time()
        self.get_target_value()
        # self.result_info()

        self.DEBUG("avg_delay = {}".format(self.avg_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))

    def solve(self):
        if self.T_flags[1]:
            for sid in range(self.env.site_num):
                site = self.env.sites[sid]          # type: SiteNode
                self.tp_serviceA[sid] = 1 / site.service_rate_a * 1000
                self.tp_serviceR[sid] = 1 / site.service_rate_r * 1000

        if self.T_flags[2]:
            for uid in range(self.env.user_num):
                user = self.env.users[uid]          # type: UserNode
                for sid in range(self.env.site_num):
                    site = self.env.sites[sid]      # type: SiteNode

                    self.best_x_serviceA[uid][sid], self.tq_serviceA[uid][sid] = self.get_best_x_and_tq(user.service_a.arrival_rate,
                                                                                                        site.service_rate_a,
                                                                                                        sigma=self.env.eta * site.price_a)
                    self.best_x_serviceR[uid][sid], self.tq_serviceR[uid][sid] = self.get_best_x_and_tq(user.service_r.arrival_rate,
                                                                                                        site.service_rate_r,
                                                                                                        sigma=self.env.eta * site.price_r)

        avg_delay = math.inf
        while len(self.selected_sites) < self.M:
            self.DEBUG("avg_delay = {}".format(avg_delay))
            best_assignment = None          # 本轮的最佳assignment
            s_star = None
            for s in self.candidate_sites:
                # 添加site之后，用户离它更近则关联这个site
                temp_assignment = copy.deepcopy(self.assignment)
                for u, assign in temp_assignment.items():
                    if temp_assignment[u] is None or self.env.tx_user_node[u][assign] > self.env.tx_user_node[u][s]:
                        temp_assignment[u] = s

                # 计算 avg-T
                cur_avg_delay = self.compute_avg_delay(temp_assignment)

                if cur_avg_delay < avg_delay:
                    avg_delay = cur_avg_delay
                    best_assignment = copy.deepcopy(temp_assignment)
                    s_star = s

            if s_star is not None:
                self.DEBUG("s_star = {}".format(s_star))
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
                    if self.T_flags[2]:
                        user.service_a.update_num_server(self.best_x_serviceA[user.user_id][user.service_a.node_id])
                        user.service_r.update_num_server(self.best_x_serviceR[user.user_id][user.service_r.node_id])

                    else:
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
        计算最佳服务器及其排队时延
        sigma = eta * price
    """
    @staticmethod
    def get_best_x_and_tq(lambda_, miu_, sigma):
        a = lambda_ / miu_
        x = 0
        B = 1

        while x <= a:
            x += 1
            B = a * B / (x + a * B)
        C_old = x * B / (x - a + a * B)

        while True:
            x = x + 1
            B = a * B / (x + a * B)
            C_new = x * B / (x - a + a * B)

            t_now = C_new / ((x - a) * miu_) * 1000
            t_old = C_old / ((x - 1 - a) * miu_) * 1000
            delta_t = t_old - t_now
            if delta_t <= sigma:
                break

            C_old = C_new

        x_star = x - 1
        t_x_star = t_old
        # phi_star = t_x_star + sigma * x_star

        return int(x_star), t_x_star

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
            if delta_t <= eta_f:
                break

            C_old = C_new

        x_star = x - 1
        return int(x_star)

    def compute_avg_delay(self, assignment: dict):
        total_delay = 0
        for ui in range(self.env.user_num):
            for uj in range(self.env.user_num):
                delay = 0

                if self.T_flags[0]:
                    tx1 = self.env.tx_user_node[ui][assignment[ui]] * self.env.data_size[0]
                    tx2 = self.env.tx_node_node[assignment[ui]][assignment[uj]] * self.env.data_size[1]
                    tx3 = self.env.tx_node_user[assignment[uj]][uj] * self.env.data_size[2]
                    delay += tx1 + tx2 + tx3

                if self.T_flags[1]:
                    tp1 = self.tp_serviceA[assignment[ui]]
                    tp2 = self.tp_serviceR[assignment[uj]]
                    delay += tp1 + tp2

                if self.T_flags[2]:
                    tq1 = self.tq_serviceA[ui][assignment[ui]]
                    tq2 = self.tq_serviceR[uj][assignment[uj]]
                    delay += tq1 + tq2

                total_delay += delay

        avg_delay = total_delay / self.n_2
        return avg_delay

    def check_result(self):
        for u, s in self.assignment.items():
            print("client {}(area {})---> site {}(area {})".format(u, self.env.users[u].area_id, s, self.env.sites[s].area_id))

    """
        获取最终的统计信息
    """
    def get_results(self):
        self.results["avg_delay"] = self.avg_delay
        self.results["cost"] = self.final_avg_cost
        self.results["target_value"] = self.target_value
        self.results["running_time"] = self.running_time
        self.results["n_selected_site"] = len(self.selected_sites)
        self.results["local_count"] = self.local_count
        self.results["common_count"] = self.common_count
        return self.results

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)