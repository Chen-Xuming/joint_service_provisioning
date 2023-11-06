"""
   Modify-Assignment 算法的原论文实现方法（时间复杂度低）
"""

import copy
import math

from min_avg.base import BaseMinAvgAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
from env.service import Service

class ModifyAssignmentAlgorithm(BaseMinAvgAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinAvgAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "modify_assignment" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only  # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.debug_flag = False

        self.assignments = dict()
        assert self.have_initial_solution(), "Input environment is illegal."

        self.L = dict()

    def run(self):
        self.start_time = time()

        self.solve()

        # self.server_allocation()

        self.get_running_time()
        self.get_target_value()
        self.result_info()

        self.DEBUG("avg_delay = {}".format(self.avg_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))

    def solve(self):
        # line 2-3
        self.calculate_Ls()

        while True:
            delta_star = -math.inf
            c_star = None
            s_star = None

            for site in self.env.sites:         # type: SiteNode
                for client in self.env.users:   # type: UserNode
                    s_a = self.assignments[client.user_id]   # 用户 c 关联的节点（服务A和R关联相同的节点）
                    delta = 0
                    delta += (self.env.user_num + 1) * self.env.tx_user_node[client.user_id][s_a]
                    delta += self.L[s_a]
                    delta -= (self.env.user_num + 1) * self.env.tx_user_node[client.user_id][site.global_id]
                    delta -= self.L[site.global_id]
                    delta += self.env.tx_node_node[site.global_id][s_a]

                    if delta > 0 and delta > delta_star:
                        delta_star = delta
                        c_star = client.user_id
                        s_star = site.global_id

            if delta_star > 0:
                sa_cstar = self.assignments[c_star]
                for site in self.env.sites:         # type: SiteNode
                    self.L[site.global_id] = self.L[site.global_id] - self.env.tx_node_node[site.global_id][sa_cstar] + \
                                             self.env.tx_node_node[site.global_id][s_star]

                # 重新 assign
                self.assignments[c_star] = s_star

            else:
                break

        """ 重新初始化各个服务的服务器个数 """
        for user in self.env.users:  # type: UserNode
            user.service_a.node_id = self.assignments[user.user_id]
            user.service_a.service_rate = self.env.sites[user.service_a.node_id].service_rate_a
            user.service_a.price = self.env.sites[user.service_a.node_id].price_a
            if self.consider_cost_tq:
                if not self.stable_only:
                    x_a = self.resource_allocation(user.service_a.arrival_rate, user.service_a.service_rate, self.env.eta, user.service_a.price)
                else:
                    x_a = user.service_a.get_num_server_for_stability(user.service_a.service_rate)
                user.service_a.update_num_server(x_a)

            user.service_r.node_id = self.assignments[user.user_id]
            user.service_r.service_rate = self.env.sites[user.service_r.node_id].service_rate_r
            user.service_r.price = self.env.sites[user.service_r.node_id].price_r

            if self.consider_cost_tq:
                if not self.stable_only:
                    x_r = self.resource_allocation(user.service_r.arrival_rate, user.service_r.service_rate, self.env.eta, user.service_r.price)
                else:
                    x_r = user.service_r.get_num_server_for_stability(user.service_r.service_rate)
                user.service_r.update_num_server(x_r)

    def calculate_Ls(self):
        a = [0] * self.env.site_num

        for u in self.env.users:
            a[u.service_a.node_id] += 1

        for s in range(self.env.site_num):
            self.L[s] = 0
            for sj in range(self.env.site_num):
                self.L[s] += a[sj] * self.env.tx_node_node[s][sj]

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

            # delta_target = (C_new - C_old) * 1000 + eta_f
            # if delta_target >= 0:
            #     break
            C_old = C_new

        x_star = x - 1
        return int(x_star)


    """ 
        在现有的 assignment 基础上进行资源分配，试图进一步是目标函数值变小
    """
    def server_allocation(self):
        # 计算服务s的目标函数值：Tp + eta * w * x
        def f(s: Service):
            value = s.queuing_delay * 1000 + self.env.eta * s.price * s.num_server  # 注意：这里的排队时延是以ms为单位的
            return value

        # 对于一个服务，尝试增加服务器使 f 值最小
        def min_f(s: Service):
            changed = False
            cur_val = f(s)
            while True:
                # fixme: 当排队时延小于 1ms 时，不要再增加了
                if s.queuing_delay * 1000 <= 1.0:
                    break

                s.update_num_server(s.num_server + 1)
                new_val = f(s)
                if new_val >= cur_val:  # 当f值无法再减小时停止
                    s.update_num_server(s.num_server - 1)
                    break
                self.DEBUG("user {}, service {} add a server, f: {} --> {}".format(s.user_id, s.service_type,
                                                                                   cur_val, new_val))
                cur_val = new_val
                changed = True
            return changed

        while True:
            # 找出交互时延最大的用户对
            user_i, user_j, max_delay = self.env.compute_max_interactive_delay(self.env.users)

            changed_sa = min_f(user_i.service_a)
            changed_sr = min_f(user_j.service_r)

            if (not changed_sa) and (not changed_sr):
                break


    """ 本算法是建立在其他算法的解之上！ """
    def have_initial_solution(self):
        for u in self.env.users:            # type: UserNode
            if u.service_a.node_id is None or u.service_r.node_id is None:
                return False
            else:
                self.assignments[u.user_id] = u.service_a.node_id
        return True

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)


if __name__ == "__main__":
    from env.environment import Environment
    from configuration.config import config as conf
    from min_avg.nearest import NearestAlgorithm
    import random

    u_seed = random.randint(0, 10000000000)
    # u_seed = 4932794917
    print("user_seed = {}".format(u_seed))

    env_seed = 58972
    print("env_seed: ", env_seed)

    num_user = 40

    print("=========================== VERSION 1 ===============================")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    nearest_alg = NearestAlgorithm(env, consider_cost_tq=False, stable_only=False)
    nearest_alg.run()
    ma1 = ModifyAssignmentAlgorithm(env, consider_cost_tq=False, stable_only=False)
    ma1.run()
    print(ma1.avg_delay, ma1.final_avg_cost, ma1.target_value)
