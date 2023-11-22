"""
   Modify-Assignment-V2 衍生版本：每个用户的服务A和R卸载到相同位置
"""

import copy
import math
import numpy as np

from min_avg.base import BaseMinAvgAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
from env.service import Service

class ModifyAssignmentAlgorithm_AR_same(BaseMinAvgAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, t_compositions=0b100, *args, **kwargs):
        BaseMinAvgAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "modify_assignment_v2_ar_same" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only              # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.debug_flag = False

        assert self.have_initial_solution(), "Input environment is illegal."

        self.L = dict()

        # 计算 max-T 过程中考虑哪些成分（Tx, Tp, Tq）
        # 100: Tx = 1, Tp = 0, Tq = 0   # 只考虑Tx
        self.T_flags = [False for i in range(3)]
        self.T_flags[0] = bool(t_compositions & 4)
        self.T_flags[1] = bool(t_compositions & 2)
        self.T_flags[2] = bool(t_compositions & 1)
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

        # self.server_allocation()

        self.get_running_time()
        self.get_target_value()
        # self.result_info()

        self.DEBUG("avg_delay = {}".format(self.avg_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))

    """
        尝试修改用户服务的卸载位置，使得交互时延之和减小，重复操作知道不再减小为止
    """
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

        # ---------------------------------- Modify Assignment -------------------------------------
        iteration = 1
        while True:
            self.DEBUG("---------- iteration {} -----------".format(iteration))
            delta_star = -math.inf
            c_star = None
            s_star = None
            for site in self.env.sites:  # type: SiteNode
                for user in self.env.users:  # type: UserNode
                    assert user.service_a.node_id == user.service_r.node_id
                    if user.service_a.node_id == site.global_id:
                        continue
                    delta = self.cal_reduction(user, site)
                    if delta > 0 and delta > delta_star:
                        c_star = user
                        s_star = site
                        delta_star = delta
            if delta_star > 0:
                self.DEBUG("[modify] user{}: site{} --> site{}".format(c_star.user_id, c_star.service_a.node_id, s_star.global_id))
                c_star.service_a.node_id = s_star.global_id
                c_star.service_r.node_id = s_star.global_id
            else:
                break
            iteration += 1

        # -----------------------------------------------------------------------------------------

        """ 重新初始化各个服务的服务器个数 """
        for user in self.env.users:  # type: UserNode
            user.service_a.service_rate = self.env.sites[user.service_a.node_id].service_rate_a
            user.service_a.price = self.env.sites[user.service_a.node_id].price_a
            if self.consider_cost_tq:
                if not self.stable_only:
                    x_a = self.resource_allocation(user.service_a.arrival_rate, user.service_a.service_rate, self.env.eta, user.service_a.price)
                else:
                    x_a = user.service_a.get_num_server_for_stability(user.service_a.service_rate)
                user.service_a.update_num_server(x_a)

            user.service_r.service_rate = self.env.sites[user.service_r.node_id].service_rate_r
            user.service_r.price = self.env.sites[user.service_r.node_id].price_r
            if self.consider_cost_tq:
                if not self.stable_only:
                    x_r = self.resource_allocation(user.service_r.arrival_rate, user.service_r.service_rate, self.env.eta, user.service_r.price)
                else:
                    x_r = user.service_r.get_num_server_for_stability(user.service_r.service_rate)
                user.service_r.update_num_server(x_r)

    def cal_reduction(self, cur_user: UserNode, target_site: SiteNode):
        # ------------------------ 计算原来的时延和 --------------------------
        original_sum = 0
        for user_to in self.env.users:  # type: UserNode
            if self.T_flags[0]:
                original_sum += self.env.tx_user_node[cur_user.user_id][cur_user.service_a.node_id] * \
                                self.env.data_size[0] + \
                                self.env.tx_node_node[cur_user.service_a.node_id][user_to.service_r.node_id] * \
                                self.env.data_size[1] + \
                                self.env.tx_node_user[user_to.service_r.node_id][user_to.user_id] * \
                                self.env.data_size[2]
            if self.T_flags[1]:
                original_sum += self.tp_serviceA[cur_user.service_a.node_id]
            if self.T_flags[2]:
                original_sum += self.tq_serviceA[cur_user.user_id][cur_user.service_a.node_id]
        for user_from in self.env.users:  # type: UserNode
            if self.T_flags[0]:
                original_sum = self.env.tx_user_node[user_from.user_id][user_from.service_a.node_id] * \
                               self.env.data_size[0] + \
                               self.env.tx_node_node[user_from.service_a.node_id][cur_user.service_r.node_id] * \
                               self.env.data_size[1] + \
                               self.env.tx_node_user[cur_user.service_r.node_id][cur_user.user_id] * \
                               self.env.data_size[2]
            if self.T_flags[1]:
                original_sum += self.tp_serviceR[cur_user.service_r.node_id]
            if self.T_flags[2]:
                original_sum += self.tq_serviceR[cur_user.user_id][cur_user.service_r.node_id]

        # ------------------------ 计算修改后的时延和 --------------------------
        new_sum = 0
        for user_to in self.env.users:  # type: UserNode
            if self.T_flags[0]:
                new_sum += self.env.tx_user_node[cur_user.user_id][target_site.global_id] * self.env.data_size[0] + \
                           self.env.tx_node_node[target_site.global_id][user_to.service_r.node_id] * \
                           self.env.data_size[1] + \
                           self.env.tx_node_user[user_to.service_r.node_id][user_to.user_id] * self.env.data_size[2]
            if self.T_flags[1]:
                new_sum += self.tp_serviceA[target_site.global_id]
            if self.T_flags[2]:
                new_sum += self.tq_serviceA[cur_user.user_id][target_site.global_id]
        for user_from in self.env.users:  # type: UserNode
            if self.T_flags[0]:
                new_sum += self.env.tx_user_node[user_from.user_id][user_from.service_a.node_id] * \
                           self.env.data_size[0] + \
                           self.env.tx_node_node[user_from.service_a.node_id][target_site.global_id] * \
                           self.env.data_size[1] + \
                           self.env.tx_node_user[target_site.global_id][cur_user.user_id] * self.env.data_size[2]
            if self.T_flags[1]:
                new_sum += self.tp_serviceR[target_site.global_id]
            if self.T_flags[2]:
                new_sum += self.tq_serviceR[cur_user.user_id][target_site.global_id]

        return original_sum - new_sum

    """
        计算交互时延总和（考虑数据传输量的传输时延），单位ms
    """
    def get_total_interaction_delay(self):
        total = 0
        for user_from in self.env.users:  # type: UserNode
            for user_to in self.env.users:  # type: UserNode
                transmission_delay = self.env.tx_user_node[user_from.user_id][user_from.service_a.node_id] * self.env.data_size[0] + \
                                     self.env.tx_node_node[user_from.service_a.node_id][user_to.service_r.node_id] * self.env.data_size[1] + \
                                     self.env.tx_node_user[user_to.service_r.node_id][user_to.user_id] * self.env.data_size[2]
                total += transmission_delay
        return total

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

            # delta_target = (C_new - C_old) * 1000 + eta_f
            # if delta_target >= 0:
            #     break
            C_old = C_new

        x_star = x - 1
        return int(x_star)

    """ 本算法是建立在其他算法的解之上！ """
    def have_initial_solution(self):
        for u in self.env.users:            # type: UserNode
            if u.service_a.node_id is None or u.service_r.node_id is None:
                return False

        return True

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)


if __name__ == "__main__":
    from env.environment_old import Environment
    from configuration.config import config as conf
    from min_avg.nearest import NearestAlgorithm
    from min_avg.min_avg_ours import MinAvgOurs
    import random

    u_seed = random.randint(0, 10000000000)
    # u_seed = 4932794917
    print("user_seed = {}".format(u_seed))

    env_seed = 99497
    print("env_seed: ", env_seed)

    num_user = 40

    print("=========================== MA-alg VERSION 2 ===============================")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    nearest_alg = NearestAlgorithm(env)
    nearest_alg.run()
    print(nearest_alg.avg_delay, nearest_alg.final_avg_cost, nearest_alg.target_value)

    ma2_ar_same = ModifyAssignmentAlgorithm_AR_same(env, t_compositions=0b111)
    ma2_ar_same.debug_flag = True
    ma2_ar_same.run()
    ma2_ar_same.result_info()
    print(ma2_ar_same.avg_delay, ma2_ar_same.final_avg_cost, ma2_ar_same.target_value)




