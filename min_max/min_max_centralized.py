"""
    Fix some bugs of min_max_ours_v2.py
"""

import copy
import math

from min_max.base import BaseMinMaxAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
import networkx as nx
import numpy as np

from heapdict import heapdict

class MinMaxOursCentralized(BaseMinMaxAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinMaxAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_max_centralized" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only  # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.n = self.env.user_num
        self.n_2 = self.env.user_num ** 2

        self.alpha = 0.0005      # fixme: adjust this value
        self.epsilon = 5

        self.backup_empty_graph = None
        self.graph = nx.DiGraph()  # 有向图
        self.reversed_graph = None  # graph的反向图

        self.shortest_paths = dict()
        self.shortest_path_lengths = heapdict()

        self.T_matrix = np.zeros((self.env.user_num, self.env.user_num))  # 次梯度（时延矩阵）

        # resource allocation 问题的最优解（虽然使用二维矩阵存储，但是每个矩阵中有一些值是无效值）
        self.phi_cs_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.phi_cpi_spi_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.o_cs_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.o_cpi_spi_arr = np.zeros((self.env.user_num, self.env.site_num))

        # 权重矩阵
        self.psi_c_arr = np.zeros(self.env.user_num)
        self.psi_c_pi_arr = np.zeros(self.env.user_num)
        self.psi_arr = np.zeros((self.env.user_num, self.env.user_num))
        for i in range(self.env.user_num):
            for j in range(self.env.user_num):
                self.psi_arr[i][j] = 1 / self.n_2

        # 存储每个服务卸载到各个站点，最佳的服务器个数
        self.best_x_serviceA = np.zeros((self.env.user_num, self.env.site_num))
        self.best_x_serviceR = np.zeros((self.env.user_num, self.env.site_num))

        # 刚好满足稳态条件时的服务器个数
        self.queue_stable_x_serviceA = np.zeros((self.env.user_num, self.env.site_num))
        self.queue_stable_x_serviceR = np.zeros((self.env.user_num, self.env.site_num))
        for user in self.env.users:         # type: UserNode
            for site in self.env.sites:     # type: SiteNode
                x_star_a = user.service_a.get_num_server_for_stability(site.service_rate_a)
                self.queue_stable_x_serviceA[user.user_id][site.global_id] = x_star_a
                x_star_r = user.service_r.get_num_server_for_stability(site.service_rate_r)
                self.queue_stable_x_serviceR[user.user_id][site.global_id] = x_star_r

        # 最佳解
        self.best_solution = {
            "association": [],  # [(a1, r1), (a2, r2), ...]
            "service_amount": []  # [(xa1, xr1), (xa2, xr2), ...]
        }

        self.total_iterations = 0
        self.best_iteration = 0

        self.max_iteration = 50     # fixme: fixme: adjust this value

        # 记录 f 和 g 值的变化
        self.f_values = []
        self.g_values = []

        self.debug_flag = False

    def run(self):
        self.start_time = time()

        self.solve()

        self.get_running_time()
        self.get_target_value()
        # self.result_info()

        self.DEBUG("max_delay = {}".format(self.max_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))
        self.DEBUG("local_count = {}".format(self.local_count))
        self.DEBUG("common_count = {}".format(self.common_count))

    def solve(self):
        k = 0
        k_star = 0
        g_best = 0
        f_best = math.inf

        all_user_set = set([i for i in range(self.env.user_num)])

        while (f_best - g_best) >= self.epsilon:
            k = k + 1
            self.DEBUG("----------- iteration {} ------------".format(k))

            """ 1. 计算用户权重，并划分零和非零权重用户集合 """
            self.psi_c_arr = np.sum(self.psi_arr, axis=1)
            self.psi_c_pi_arr = np.sum(self.psi_arr, axis=0)
            assert abs(np.sum(self.psi_c_arr) - 1.0) < 1e-8
            assert abs(np.sum(self.psi_c_pi_arr) - 1.0) < 1e-8
            set_c0, set_c0_pi = set(), set()
            for user_id in range(self.env.user_num):
                if self.psi_c_arr[user_id] == 0:
                    set_c0.add(user_id)
                if self.psi_c_pi_arr[user_id] == 0:
                    set_c0_pi.add(user_id)
            set_c1 = all_user_set - set_c0
            set_c1_pi = all_user_set - set_c0_pi

            self.DEBUG("Length: C0 = {}, C0' = {}, C1 = {}, C1' = {}".format(len(set_c0), len(set_c0_pi), len(set_c1), len(set_c1_pi)))
            # self.DEBUG("C0 = {}".format(set_c0))
            # self.DEBUG("C0_pi = {}".format(set_c0_pi))
            # self.DEBUG("C1 = {}".format(set_c1))
            # self.DEBUG("C1_pi = {}".format(set_c1_pi))

            """ 2. Resource Allocation """
            if self.consider_cost_tq:
                self.resource_allocation(set_c0, set_c0_pi, set_c1, set_c1_pi)

            """ 3. Service Placement """
            self.reset_all_service()
            self.build_graph()
            self.set_edge_weight(set_c0, set_c0_pi, set_c1, set_c1_pi)
            self.assign_users_c_zero(set_c0, set_c0_pi)
            self.get_all_stp(set_c0, set_c0_pi)
            self.assign_users_c_one(set_c1, set_c1_pi)

            """ 4. Keep trace of the best solution found so far """
            self.compute_sub_gradient()
            current_f, cur_max_delay, cur_avg_cost = self.env.compute_target_function_value("min-max")
            current_g = self.compute_g_value()
            self.f_values.append(current_f)
            self.g_values.append(current_g)
            g_best = max(g_best, current_g)
            if current_f < f_best:
                f_best = current_f
                k_star = k
                self.save_best_solution()

            self.DEBUG("max-delay = {:.4f} ms, avg_cost = {:.4f}".format(cur_max_delay * 1000, cur_avg_cost))
            self.DEBUG("cur-f / best-f = {:.4f} / {:.4f}".format(current_f, f_best))
            self.DEBUG("cur-g / best-g = {:.4f} / {:.4f}".format(current_g, g_best))

            """ 5. Update Sub-gradient """
            alpha_k = self.alpha / k
            alpha_k = max(alpha_k, self.alpha / 30)         # fixme
            self.DEBUG("alpha_k = {}".format(alpha_k))

            # only for debug
            if self.debug_flag:
                t_matrix_max = np.max(self.T_matrix)
                t_matrix_min = np.min(self.T_matrix)
                t_matrix_avg = np.mean(self.T_matrix)
                psi_arr_max = np.max(self.psi_arr)
                psi_arr_min = np.min(self.psi_arr)
                psi_arr_avg = np.mean(self.psi_arr)
                maxT_up = np.unravel_index(np.argmax(self.T_matrix), self.T_matrix.shape)
                maxPsi_up = np.unravel_index(np.argmax(self.psi_arr), self.psi_arr.shape)
                print("T-min, T-avg, T-max = {}, {}, {}".format(t_matrix_min, t_matrix_avg, t_matrix_max))
                print("Psi-min, Psi-avg, Psi-max = {}, {}, {}".format(psi_arr_min, psi_arr_avg, psi_arr_max))
                print("[max-user_pair] T: {}, Psi: {}".format(maxT_up, maxPsi_up))
                print("Range of alpha * T = [{}, {}]".format(alpha_k * t_matrix_min, alpha_k * t_matrix_max))

            for i in range(self.env.user_num):
                for j in range(self.env.user_num):
                    self.psi_arr[i][j] += alpha_k * self.T_matrix[i][j]

            """ 6. Projection """
            self.projection()
            self.check_psi_satisfy_constraint()

            if k == self.max_iteration:
                break

        """ 还原最优解 """
        self.set_best_solution()

        self.total_iterations = k
        self.best_iteration = k_star

    """ 根据用户权重计算进行资源分配 """
    def resource_allocation(self, c0: set, c0_pi: set, c1: set, c1_pi: set):
        self.resource_allocation_for_queue_stability(c0, c0_pi)
        self.resource_allocation_normal(c1, c1_pi)

    """ 对于权重为0的用户，满足稳态条件即可 """
    def resource_allocation_for_queue_stability(self, c0: set, c0_pi: set):
        for user_c in c0:
            for site_s in range(self.env.site_num):
                x_cs = self.queue_stable_x_serviceA[user_c][site_s]
                self.best_x_serviceA[user_c][site_s] = x_cs
                self.o_cs_arr[user_c][site_s] = self.env.eta * self.env.sites[site_s].price_a * x_cs / self.n

        for user_cpi in c0_pi:
            for site_spi in range(self.env.site_num):
                x_cpi_spi = self.queue_stable_x_serviceR[user_cpi][site_spi]
                self.best_x_serviceR[user_cpi][site_spi] = x_cpi_spi
                self.o_cpi_spi_arr[user_cpi][site_spi] = self.env.eta * self.env.sites[
                    site_spi].price_r * x_cpi_spi / self.n

    """ 对于权重非0的用户，使用 Resource Allocation 计算最佳服务器个数 """
    def resource_allocation_normal(self, c1: set, c1_pi: set):
        # service A
        for user_c in c1:
            user = self.env.users[user_c]
            for site_s in range(self.env.site_num):
                site = self.env.sites[site_s]

                # 暂时将服务卸载到节点, 方便计算后面的值
                service = user.service_a
                service.assign_to_site(site)

                sigma = self.env.eta * service.price / (self.n * self.psi_c_arr[user_c])
                x_cs, phi_cs = self.get_best_server_num(service.arrival_rate, service.service_rate, sigma)
                self.best_x_serviceA[user_c][site_s] = x_cs
                self.phi_cs_arr[user_c][site_s] = phi_cs

        # service R
        for user_cpi in c1_pi:
            user = self.env.users[user_cpi]
            for site_spi in range(self.env.site_num):
                site = self.env.sites[site_spi]

                # 暂时将服务卸载到节点, 方便计算后面的值
                service = user.service_r
                service.assign_to_site(site)

                sigma = self.env.eta * service.price / (self.n * self.psi_c_pi_arr[user_cpi])
                x_cpi_spi, phi_cpi_spi = self.get_best_server_num(service.arrival_rate, service.service_rate, sigma)
                self.best_x_serviceR[user_cpi][site_spi] = x_cpi_spi
                self.phi_cpi_spi_arr[user_cpi][site_spi] = phi_cpi_spi

    """
        求解最佳服务器个数
    """
    @staticmethod
    def get_best_server_num(lambda_, miu_, sigma):
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
        phi_star = t_x_star + sigma * x_star

        return int(x_star), phi_star

    """
        第一列节点是用户节点 --->  u1i, i = user_id
        第二列节点是所有site节点 ---> s2j, j = site_id
        第三列节点是第二列节点的拷贝 ---> s3j, j = site_id
        第四列节点是第一列的拷贝 ---> u4i, i = user_id
    """
    def build_graph(self):
        # 使用一个新的图
        if self.backup_empty_graph is not None:
            self.graph = copy.deepcopy(self.backup_empty_graph)
            return

        # 添加节点
        for user in self.env.users:  # type: UserNode
            self.graph.add_node("u1{}".format(user.user_id))
            self.graph.add_node("u4{}".format(user.user_id))
        for site in self.env.sites:  # type: SiteNode
            self.graph.add_node("s2{}".format(site.global_id))
            self.graph.add_node("s3{}".format(site.global_id))

        # 添加边
        for i in range(self.env.user_num):
            for j in range(self.env.site_num):
                self.graph.add_edge("u1{}".format(i), "s2{}".format(j), weight=0)

        for i in range(self.env.site_num):
            for j in range(self.env.site_num):
                self.graph.add_edge("s2{}".format(i), "s3{}".format(j), weight=0)

        for j in range(self.env.site_num):
            for i in range(self.env.user_num):
                self.graph.add_edge("s3{}".format(j), "u4{}".format(i), weight=0)

        self.backup_empty_graph = copy.deepcopy(self.graph)

    """
        为权重为 0 的用户放置服务
    """
    def assign_users_c_zero(self, c0: set, c0_pi: set):
        for user_c in c0:
            # 找出使 o_cs 最大的 s
            s_star = np.argmax(self.o_cs_arr[user_c])

            # 将该用户的服务卸载到该节点
            target_site = self.env.sites[s_star]        # type: SiteNode
            user = self.env.users[user_c]               # type: UserNode
            user.service_a.node_id = target_site.global_id
            user.service_a.service_rate = target_site.service_rate_a
            user.service_a.price = target_site.price_a
            user.service_r.node_id = target_site.global_id
            user.service_r.service_rate = target_site.service_rate_r
            user.service_r.price = target_site.price_r
            if self.consider_cost_tq:
                user.service_a.update_num_server(self.queue_stable_x_serviceA[user_c][s_star])
                user.service_r.update_num_server(self.queue_stable_x_serviceR[user_c][s_star])
            else:
                user.service_a.queuing_delay = 0
                user.service_a.num_server = 0
                user.service_r.queuing_delay = 0
                user.service_r.num_server = 0

            # 删除 graph 中的冲突边
            user_c_node_name = "u1{}".format(user_c)
            user_c_u4_node = "u4{}".format(user_c)
            for k in range(self.env.site_num):
                if k != s_star:
                    site_node_name = "s2{}".format(k)
                    self.graph.remove_edge(user_c_node_name, site_node_name)
                    self.reversed_graph.remove_edge(site_node_name, user_c_node_name)

                    site_node_name = "s3{}".format(k)
                    self.graph.remove_edge(site_node_name, user_c_u4_node)
                    self.reversed_graph.remove_edge(user_c_u4_node, site_node_name)

        for user_cpi in c0_pi:
            # 找出使 o_cpi_spi 最大的 spi
            spi_star = np.argmax(self.o_cpi_spi_arr[user_cpi])

            # 将该用户的服务卸载到该节点
            target_site = self.env.sites[spi_star]          # type: SiteNode
            user = self.env.users[user_cpi]                 # type: UserNode
            user.service_r.node_id = target_site.global_id
            user.service_r.service_rate = target_site.service_rate_r
            user.service_r.price = target_site.price_r
            user.service_a.node_id = target_site.global_id
            user.service_a.service_rate = target_site.service_rate_a
            user.service_a.price = target_site.price_a
            if self.consider_cost_tq:
                user.service_r.update_num_server(self.queue_stable_x_serviceR[user_cpi][spi_star])
                user.service_a.update_num_server(self.queue_stable_x_serviceA[user_cpi][spi_star])
            else:
                user.service_r.queuing_delay = 0
                user.service_r.num_server = 0
                user.service_a.queuing_delay = 0
                user.service_a.num_server = 0

            # 删除 graph 中的冲突边
            user_cpi_node_name = "u4{}".format(user_cpi)
            user_cpi_u1_node = "u1{}".format(user_cpi)
            for k in range(self.env.site_num):
                if k != spi_star:
                    site_node_name = "s3{}".format(k)
                    self.graph.remove_edge(site_node_name, user_cpi_node_name)
                    self.reversed_graph.remove_edge(user_cpi_node_name, site_node_name)

                    site_node_name = "s2{}".format(k)
                    self.graph.remove_edge(user_cpi_u1_node, site_node_name)
                    self.reversed_graph.remove_edge(site_node_name, user_cpi_u1_node)

    def set_edge_weight(self, c0: set, c0_pi: set, c1: set, c1_pi: set):
        for user_c0 in c0:
            for site_s in range(self.env.site_num):
                site = self.env.sites[site_s]
                weight = 0
                weight += self.env.tx_user_node[user_c0][site_s] * self.env.data_size[0]
                weight += 1 / site.service_rate_a * 1000
                weight += self.o_cs_arr[user_c0][site_s]
                self.graph["u1{}".format(user_c0)]["s2{}".format(site_s)]['weight'] = weight

        for user_c1 in c1:
            for site_s in range(self.env.site_num):
                site = self.env.sites[site_s]
                weight = 0
                weight += self.env.tx_user_node[user_c1][site_s] * self.env.data_size[0]
                weight += 1 / site.service_rate_a * 1000        # ms
                weight += self.phi_cs_arr[user_c1][site_s]       # 当 self.consider_cost_tq == False 时，这一项是0
                self.graph["u1{}".format(user_c1)]["s2{}".format(site_s)]['weight'] = weight

        for s in range(self.env.site_num):
            for spi in range(self.env.site_num):
                weight = self.env.tx_node_node[s][spi] * self.env.data_size[1]
                self.graph["s2{}".format(s)]["s3{}".format(spi)]['weight'] = weight

        for site_spi in range(self.env.site_num):
            site = self.env.sites[site_spi]
            for user_c0pi in c0_pi:
                weight = 0
                weight += self.env.tx_node_user[site_spi][user_c0pi] * self.env.data_size[2]
                weight += 1 / site.service_rate_r * 1000
                weight += self.phi_cpi_spi_arr[user_c0pi][site_spi]
                self.graph["s3{}".format(site_spi)]["u4{}".format(user_c0pi)]['weight'] = weight

        for site_spi in range(self.env.site_num):
            site = self.env.sites[site_spi]
            for user_c1pi in c1_pi:
                weight = 0
                weight += self.env.tx_node_user[site_spi][user_c1pi] * self.env.data_size[2]
                weight += 1 / site.service_rate_r * 1000        # ms
                weight += self.phi_cpi_spi_arr[user_c1pi][site_spi]
                self.graph["s3{}".format(site_spi)]["u4{}".format(user_c1pi)]['weight'] = weight

        self.reversed_graph = nx.reverse(self.graph, copy=True)

    def get_all_stp(self, c0: set, c0_pi: set):
        self.shortest_paths.clear()
        self.shortest_path_lengths = heapdict()

        for user_from in range(self.n):
            self.shortest_paths["u1{}".format(user_from)] = dict()
            lens, paths = nx.single_source_dijkstra(self.graph, "u1{}".format(user_from))

            for user_to in range(self.n):
                if user_from in c0 and user_to in c0_pi:
                    continue
                self.shortest_paths["u1{}".format(user_from)]["u4{}".format(user_to)] = paths["u4{}".format(user_to)]
                self.shortest_path_lengths[("u1{}".format(user_from), "u4{}".format(user_to))] = lens["u4{}".format(user_to)] * self.psi_arr[user_from][user_to]

    def reset_all_service(self):
        for user in self.env.users:  # type: UserNode
            user.service_a.reset()
            user.service_r.reset()

    def assign_users_c_one(self, c1: set, c1_pi: set):
        service_a_finished_flag = dict()
        service_r_finished_flag = dict()
        for c in c1:
            service_a_finished_flag[c] = False
        for c in c1_pi:
            service_r_finished_flag[c] = False

        assignment_times = 0
        target_times = len(c1) + len(c1_pi)
        while assignment_times < target_times:
            node_i, node_j, min_stp, min_len = self.get_min_stp(service_a_finished_flag, service_r_finished_flag)
            # self.DEBUG("[min-stp-len = {}] ({}, {}): {}".format(min_len, node_i, node_j, min_stp))

            user_i = self.env.users[int(node_i[2:])]
            user_j = self.env.users[int(node_j[2:])]

            need_assign_ui_a = True if user_i.service_a.node_id is None else False  # 本轮是否要决策 user_i 的服务A
            if need_assign_ui_a:
                target_site = self.env.sites[int(min_stp[1][2:])]
                user_i.service_a.node_id = target_site.global_id
                user_i.service_a.service_rate = target_site.service_rate_a
                user_i.service_a.price = target_site.price_a
                user_i.service_r.node_id = target_site.global_id
                user_i.service_r.service_rate = target_site.service_rate_r
                user_i.service_r.price = target_site.price_r
                if self.consider_cost_tq:
                    user_i.service_a.update_num_server(self.best_x_serviceA[user_i.user_id][target_site.global_id])
                    user_i.service_r.update_num_server(self.best_x_serviceR[user_i.user_id][target_site.global_id])
                else:
                    user_i.service_a.queuing_delay = 0
                    user_i.service_a.num_server = 0
                    user_i.service_r.queuing_delay = 0
                    user_i.service_r.num_server = 0

                service_a_finished_flag[user_i.user_id] = True
                service_r_finished_flag[user_i.user_id] = True
                assignment_times += 2
                self.DEBUG("[Assign] user {} service_a --> site {}".format(user_i.user_id, target_site.global_id))
                self.DEBUG("[Assign] user {} service_r --> site {}".format(user_i.user_id, target_site.global_id))

            need_assign_uj_r = True if user_j.service_r.node_id is None else False  # 本轮是否要决策 user_j 的服务R
            if need_assign_uj_r:
                target_site = self.env.sites[int(min_stp[-2][2:])]
                user_j.service_r.node_id = target_site.global_id
                user_j.service_r.service_rate = target_site.service_rate_r
                user_j.service_r.price = target_site.price_r
                user_j.service_a.node_id = target_site.global_id
                user_j.service_a.service_rate = target_site.service_rate_a
                user_j.service_a.price = target_site.price_a
                if self.consider_cost_tq:
                    user_j.service_r.update_num_server(self.best_x_serviceR[user_j.user_id][target_site.global_id])
                    user_j.service_a.update_num_server(self.best_x_serviceA[user_j.user_id][target_site.global_id])
                else:
                    user_j.service_r.queuing_delay = 0
                    user_j.service_r.num_server = 0
                    user_j.service_a.queuing_delay = 0
                    user_j.service_a.num_server = 0

                service_r_finished_flag[user_j.user_id] = True
                service_a_finished_flag[user_j.user_id] = True
                assignment_times += 2
                self.DEBUG("[Assign] user {} service_a --> site {}".format(user_j.user_id, target_site.global_id))
                self.DEBUG("[Assign] user {} service_r --> site {}".format(user_j.user_id, target_site.global_id))

            self.shortest_paths[node_i].pop(node_j)
            self.shortest_path_lengths.__delitem__((node_i, node_j))

            """ 将冲突边删除 """
            if need_assign_ui_a:
                node_i_u4 = "u4{}".format(user_i.user_id)
                user_i_s3_node = "s3{}".format(user_i.service_r.node_id)
                for k in range(self.env.site_num):
                    node_name = "s2{}".format(k)
                    if node_name != min_stp[1]:
                        self.graph.remove_edge(node_i, node_name)
                        self.reversed_graph.remove_edge(node_name, node_i)
                    node_name = "s3{}".format(k)
                    if node_name != user_i_s3_node:
                        self.graph.remove_edge(node_name, node_i_u4)
                        self.reversed_graph.remove_edge(node_i_u4, node_name)

            if need_assign_uj_r:
                node_j_u1 = "u1{}".format(user_j.user_id)
                user_j_s2_node = "s2{}".format(user_j.service_a.node_id)
                for k in range(self.env.site_num):
                    node_name = "s3{}".format(k)
                    if node_name != min_stp[-2]:
                        self.graph.remove_edge(node_name, node_j)
                        self.reversed_graph.remove_edge(node_j, node_name)
                    node_name = "s2{}".format(k)
                    if node_name != user_j_s2_node:
                        self.graph.remove_edge(node_j_u1, node_name)
                        self.reversed_graph.remove_edge(node_name, node_j_u1)

            """ 修改被影响的最短路径 """
            if need_assign_ui_a:
                node_i_u4 = "u4{}".format(user_i.user_id)
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.graph, node_i)
                for k in range(self.n):     # fixme
                    target_node = "u4{}".format(k)
                    if target_node != node_j  and target_node != node_i_u4:
                        self.shortest_path_lengths[(node_i, target_node)] = temp_stp_lengths[target_node] * self.psi_arr[user_i.user_id][k]
                        self.shortest_paths[node_i][target_node] = temp_stp_paths[target_node]

                node_j_u1 = "u1{}".format(user_j.user_id)
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.reversed_graph, node_i_u4)
                for k in range(self.env.user_num):
                    target_node = "u1{}".format(k)
                    if target_node != node_i and target_node != node_j_u1:
                        self.shortest_path_lengths[(target_node, node_i_u4)] = temp_stp_lengths[target_node]
                        self.shortest_paths[target_node][node_i_u4] = temp_stp_paths[target_node][::-1]

            if need_assign_uj_r and user_i != user_j:
                node_j_u1 = "u1{}".format(user_j.user_id)
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.reversed_graph, node_j)
                for k in range(self.n):        # fixme
                    target_node = "u1{}".format(k)
                    if target_node != node_i and target_node != node_j_u1:
                        self.shortest_path_lengths[(target_node, node_j)] = temp_stp_lengths[target_node] * self.psi_arr[k][user_j.user_id]
                        self.shortest_paths[target_node][node_i] = temp_stp_paths[target_node][::-1]

                node_i_u4 = "u4{}".format(user_i.user_id)
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.graph, node_j_u1)
                for k in range(self.env.user_num):
                    target_node = "u4{}".format(k)
                    if target_node != node_j and target_node != node_i_u4:
                        self.shortest_path_lengths[(node_j_u1, target_node)] = temp_stp_lengths[target_node]
                        self.shortest_paths[node_j_u1][target_node] = temp_stp_paths[target_node]

    def get_min_stp(self, service_a_finished_flag, service_r_finished_flag):
        user_pair, min_stp_len, target_stp = None, -1, None
        while len(self.shortest_path_lengths) > 0:
            user_pair, min_stp_len = self.shortest_path_lengths.peekitem()

            user_i = int(user_pair[0].replace("u1", ""))
            user_j = int(user_pair[1].replace("u4", ""))

            if user_i not in service_a_finished_flag.keys() and user_j not in service_r_finished_flag.keys():
                self.shortest_path_lengths.__delitem__(user_pair)
                continue

            if (user_i in service_a_finished_flag.keys() and service_a_finished_flag[user_i]) and \
                    (user_j in service_r_finished_flag.keys() and service_r_finished_flag[user_j]):
                self.shortest_path_lengths.__delitem__(user_pair)
                user_pair = None
            else:
                break

        if user_pair is None:
            return None, None, -1, None

        target_stp = self.shortest_paths[user_pair[0]][user_pair[1]]
        return user_pair[0], user_pair[1], target_stp, min_stp_len

    def compute_sub_gradient(self):
        self.T_matrix = np.zeros((self.env.user_num, self.env.user_num))
        for user_from in self.env.users:  # type: UserNode
            for user_to in self.env.users:  # type: UserNode
                delay = self.env.compute_interactive_delay(user_from, user_to) * 1000  # ms
                self.T_matrix[user_from.user_id][user_to.user_id] = delay

    def compute_g_value(self):
        total = 0
        for i, user_from in enumerate(self.env.users):    # type: UserNode
            for j, user_to in enumerate(self.env.users):  # type: UserNode
                weighted_delay = self.T_matrix[i][j] * self.psi_arr[i][j]
                total += weighted_delay

        avg_cost = self.env.compute_average_cost()
        total += self.env.eta * avg_cost
        return total

    def save_best_solution(self):
        self.best_solution["association"].clear()
        self.best_solution["service_amount"].clear()
        for user in self.env.users:     # type: UserNode
            self.best_solution["association"].append((user.service_a.node_id, user.service_r.node_id))
            self.best_solution["service_amount"].append((user.service_a.num_server, user.service_r.num_server))

    """
        修改 psi值 满足和为 1 的约束
    """
    def projection(self):
        sorted_b = np.sort(self.psi_arr.flatten())[::-1]

        # 计算前缀和
        pre_sum = [0 for _ in range(self.n_2)]
        pre_sum[0] = sorted_b[0]
        for idx in range(1, self.n_2):
            pre_sum[idx] = pre_sum[idx - 1] + sorted_b[idx]

        # 计算rou
        rou = 0
        for j in range(1, self.n_2+1):      # j = [1, n]
            if (sorted_b[j-1] + (1 - pre_sum[j-1]) / j) > 0:
                rou = j

        # 计算 k
        k = (1 - pre_sum[rou-1]) / rou

        # 重新计算权重
        for i in range(self.env.user_num):
            for j in range(self.env.user_num):
                self.psi_arr[i][j] = max(self.psi_arr[i][j] + k, 0)

    def check_psi_satisfy_constraint(self):
        assert abs(np.sum(self.psi_arr) - 1.0) < 1e-8, "Sum of weight doesn't equal to 1."

    def set_best_solution(self):
        for user in self.env.users:     # type: UserNode
            associations = self.best_solution["association"][user.user_id]
            service_amounts = self.best_solution["service_amount"][user.user_id]

            user.service_a.reset()
            user.service_r.reset()

            """ 设置 service A """
            service = user.service_a
            service.node_id = associations[0]
            service.service_rate = self.env.sites[associations[0]].service_rate_a
            service.price = self.env.sites[associations[0]].price_a
            if self.consider_cost_tq:
                service.update_num_server(service_amounts[0])
            else:
                service.num_server = 0
                service.queuing_delay = 0

            """ 设置 service R """
            service = user.service_r
            service.node_id = associations[1]
            service.service_rate = self.env.sites[associations[1]].service_rate_r
            service.price = self.env.sites[associations[1]].price_r
            if self.consider_cost_tq:
                service.update_num_server(service_amounts[1])
            else:
                service.num_server = 0
                service.queuing_delay = 0

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)

if __name__ == "__main__":
    from env.environment2 import Environment
    import random
    from configuration.config import config as conf
    from configuration.config import alpha_initial_values as alpha_values
    from min_max.nearest import NearestAlgorithm
    from min_max.min_max_ours import MinMaxOurs as MinMaxOurs_V1
    from min_max.min_max_ours_v2 import MinMaxOurs_V2
    from min_max.mgreedy import MGreedyAlgorithm
    from min_max.min_avg_for_min_max import MinAvgForMinMax
    from min_max.stp_max_first import StpMaxFirst
    from min_max.min_max_ours_v3 import MinMaxOurs_V3

    print("==================== env  config ===============================")
    print(conf)
    print("================================================================")

    # seed = random.randint(0, 100000)
    # env_seed = 58972            # seed = 999734539, user_num = 30 曲线好看。
    env_seed = 99497

    print("env_seed: ", env_seed)

    num_user = 40

    def draw_fg(f_arr, g_arr):
        from matplotlib import pyplot as plt

        min_f = min(f_arr)
        max_f_idx = f_arr.index(min_f)
        max_g = max(g_arr)
        max_g_idx = g_arr.index(max_g)

        x_ = [(i+1) for i in range(len(f_arr))]
        plt.xlabel("iteration")
        plt.scatter(max_f_idx + 1, min_f, marker='*', color='#ff1f5b', s=120)
        plt.scatter(max_g_idx + 1, max_g, marker='*', color='#ff1f5b', s=120)
        plt.plot(x_, f_arr, label='f', color='#58B272', marker='.')
        plt.plot(x_, g_arr, label='g', color='#009ade', marker='.')
        plt.legend()
        plt.show()

    for i in range(1):
        print("========================= iteration {} ============================".format(i + 1))
        u_seed = random.randint(0, 10000000000)
        # u_seed = 3504109169
        print("user_seed = {}".format(u_seed))

        print("------------- Nearest ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        nearest_alg = NearestAlgorithm(env)
        nearest_alg.run()
        print(nearest_alg.get_results())
        #
        # print("------------- Min-Avg Algorithm ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # min_avg_alg = MinAvgForMinMax(env, consider_cost_tq=True, stable_only=False)
        # min_avg_alg.run()
        # print(min_avg_alg.get_results())
        # temp_f, temp_max_delay, temp_avg_cost = min_avg_alg.env.compute_target_function_value("min-max")
        # print("f_value = {:.4f}, max_delay = {:.4f} ms, avg_cost = {:.4f}".format(temp_f, temp_max_delay * 1000, temp_avg_cost))
        #
        # print("------------- Max Stp First ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # max_first_alg = StpMaxFirst(env, consider_cost_tq=True, stable_only=False)
        # max_first_alg.debug_flag = False
        # max_first_alg.run()
        # print(max_first_alg.get_results())
        #
        # print("------------- MGreedy ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # mg_alg = MGreedyAlgorithm(env, consider_cost_tq=True, stable_only=False)
        # mg_alg.run()
        # print(mg_alg.get_results())

        # print("------------- Ours V1 ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # our_alg_v1 = MinMaxOurs_V1(env)
        # our_alg_v1.debug_flag = True
        # if num_user <= 70:
        #     our_alg_v1.alpha = 5e-5
        # else:
        #     our_alg_v1.alpha = 1e-5
        # our_alg_v1.epsilon = 15
        # our_alg_v1.run()
        # print(our_alg_v1.get_results())
        # print("[iterations = {}, best_iteration = {}]".format(our_alg_v1.total_iterations, our_alg_v1.best_iteration))
        # draw_fg(our_alg_v1.f_values, our_alg_v1.g_values)

        # print("------------- Ours V2 ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # our_alg_v2 = MinMaxOurs_V2(env)
        # our_alg_v2.debug_flag = True
        #
        # our_alg_v2.alpha = alpha_values[conf["eta"]][num_user]
        # print("alpha = {}".format(our_alg_v2.alpha))
        #
        # our_alg_v2.epsilon = 15
        # our_alg_v2.run()
        # print(our_alg_v2.get_results())
        # print("[iterations = {}, best_iteration = {}]".format(our_alg_v2.total_iterations, our_alg_v2.best_iteration))
        # # draw_fg(our_alg_v2.f_values, our_alg_v2.g_values)

        print("------------- Ours V3 ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        our_alg_v3 = MinMaxOurs_V3(env)
        our_alg_v3.debug_flag = False

        our_alg_v3.alpha = alpha_values[conf["eta"]][num_user]
        # our_alg_v3.alpha = 2e-4
        print("alpha = {}".format(our_alg_v3.alpha))

        our_alg_v3.epsilon = 15
        our_alg_v3.run()
        our_alg_v3.result_info()
        print(our_alg_v3.get_results())
        print("[iterations = {}, best_iteration = {}]".format(our_alg_v3.total_iterations, our_alg_v3.best_iteration))
        # draw_fg(our_alg_v3.f_values, our_alg_v3.g_values)

        print("------------- Ours Cent ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        our_alg_cent = MinMaxOursCentralized(env)
        our_alg_cent.debug_flag = False

        our_alg_cent.alpha = alpha_values[conf["eta"]][num_user]
        # our_alg_v3.alpha = 2e-4
        print("alpha = {}".format(our_alg_cent.alpha))

        our_alg_cent.epsilon = 15
        our_alg_cent.run()
        our_alg_cent.result_info()
        print(our_alg_cent.get_results())
        print("[iterations = {}, best_iteration = {}]".format(our_alg_cent.total_iterations, our_alg_cent.best_iteration))
        # draw_fg(our_alg_v3.f_values, our_alg_v3.g_values)

        print("------------------------------")
        print(our_alg_v3.get_results())
        print(our_alg_cent.get_results())





