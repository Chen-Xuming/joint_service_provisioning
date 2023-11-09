import copy
import math

from min_max.base import BaseMinMaxAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
import networkx as nx
import numpy as np

from heapdict import heapdict

class MinMaxSurrogate(BaseMinMaxAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinMaxAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_max_surrogate" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only  # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.n = self.env.user_num
        self.n_2 = self.env.user_num ** 2

        self.all_user_set = set([i for i in range(self.env.user_num)])

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
        # for i in range(self.env.user_num):
        #     for j in range(self.env.user_num):
        #         self.psi_arr[i][j] = 1 / self.n_2

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

        self.max_iteration = self.n + 5     # fixme: adjust this value

        # 记录 f 和 g 值的变化
        self.f_values = []
        self.g_values = []

        self.important_user_pairs = list()      # 时延和开销之和较高的

        self.debug_flag = False

    def run(self):
        self.start_time = time()

        self.solve()

        self.get_running_time()
        self.get_target_value()

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

        self.get_initial_psi()

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
            self.assign_users_c_zero(set_c0, set_c0_pi)
            self.set_edge_weight(set_c1, set_c1_pi)
            self.get_all_stp(set_c1, set_c1_pi)
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

            # """ 5. Update Psi Matrix """
            # self.update_psi_matrix()

            """ 5. Update Sub-gradient """
            alpha_k = self.alpha / k
            alpha_k = max(alpha_k, self.alpha / 20)         # fixme
            self.DEBUG("alpha_k = {}".format(alpha_k))
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

    def get_initial_psi(self):
        self.build_graph()

        self.psi_c_arr = np.ones(self.n)
        self.psi_c_pi_arr = np.ones(self.n)
        self.resource_allocation(set(), set(), self.all_user_set, self.all_user_set)

        self.set_edge_weight(self.all_user_set, self.all_user_set)
        # self.set_edge_weight_for_only_t(self.all_user_set, self.all_user_set)
        self.get_all_stp_max_heap(self.all_user_set, self.all_user_set)

        # 获取交互时延最大的用户对，把它们加入到集合中
        max_n = self.n_2 // 2
        total_len = 0
        user_pairs = []
        lens = []
        for _ in range(max_n):
            user_pair, max_stp_len = self.shortest_path_lengths.peekitem()
            max_stp_len = -max_stp_len
            user_i = int(user_pair[0][2:])
            user_j = int(user_pair[1][2:])
            user_pair = (user_i, user_j)
            total_len += max_stp_len
            user_pairs.append(user_pair)
            lens.append(max_stp_len)
            self.shortest_path_lengths.popitem()

        # 修改权重矩阵
        for idx in range(max_n):
            up = user_pairs[idx]
            self.psi_arr[up[0]][up[1]] = lens[idx] / total_len

        # 重置数据结构
        self.graph = None
        self.phi_cs_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.phi_cpi_spi_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.o_cs_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.o_cpi_spi_arr = np.zeros((self.env.user_num, self.env.site_num))

        self.shortest_paths.clear()
        self.shortest_path_lengths = heapdict()

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
            if self.consider_cost_tq:
                user.service_a.update_num_server(self.queue_stable_x_serviceA[user_c][s_star])
            else:
                user.service_a.queuing_delay = 0
                user.service_a.num_server = 0

            # 删除 graph 中的冲突边
            user_c_node_name = "u1{}".format(user_c)
            for k in range(self.env.site_num):
                if k != s_star:
                    site_node_name = "s2{}".format(k)
                    self.graph.remove_edge(user_c_node_name, site_node_name)
                    # self.reversed_graph.remove_edge(site_node_name, user_c_node_name)

        for user_cpi in c0_pi:
            # 找出使 o_cpi_spi 最大的 spi
            spi_star = np.argmax(self.o_cpi_spi_arr[user_cpi])

            # 将该用户的服务卸载到该节点
            target_site = self.env.sites[spi_star]          # type: SiteNode
            user = self.env.users[user_cpi]                 # type: UserNode
            user.service_r.node_id = target_site.global_id
            user.service_r.service_rate = target_site.service_rate_r
            user.service_r.price = target_site.price_r
            if self.consider_cost_tq:
                user.service_r.update_num_server(self.queue_stable_x_serviceR[user_cpi][spi_star])
            else:
                user.service_r.queuing_delay = 0
                user.service_r.num_server = 0

            # 删除 graph 中的冲突边
            user_cpi_node_name = "u4{}".format(user_cpi)
            for k in range(self.env.site_num):
                if k != spi_star:
                    site_node_name = "s3{}".format(k)
                    self.graph.remove_edge(site_node_name, user_cpi_node_name)
                    # self.reversed_graph.remove_edge(user_cpi_node_name, site_node_name)

    def set_edge_weight_for_only_t(self, c1: set, c1_pi: set):
        for user_c in c1:
            for site_s in range(self.env.site_num):
                site = self.env.sites[site_s]
                weight = 0
                weight += self.env.tx_user_node[user_c][site_s] * self.env.data_size[0]
                weight += 1 / site.service_rate_a * 1000        # ms
                weight += self.env.users[user_c].service_a.queuing_delay * 1000
                self.graph["u1{}".format(user_c)]["s2{}".format(site_s)]['weight'] = weight

        for s in range(self.env.site_num):
            for spi in range(self.env.site_num):
                weight = self.env.tx_node_node[s][spi] * self.env.data_size[1]
                self.graph["s2{}".format(s)]["s3{}".format(spi)]['weight'] = weight

        for site_spi in range(self.env.site_num):
            site = self.env.sites[site_spi]
            for user_cpi in c1_pi:
                weight = 0
                weight += self.env.tx_node_user[site_spi][user_cpi] * self.env.data_size[2]
                weight += 1 / site.service_rate_r * 1000        # ms
                weight += self.env.users[user_cpi].service_r.queuing_delay * 1000
                self.graph["s3{}".format(site_spi)]["u4{}".format(user_cpi)]['weight'] = weight

        self.reversed_graph = nx.reverse(self.graph, copy=True)

    def set_edge_weight(self, c1: set, c1_pi: set):
        for user_c in c1:
            for site_s in range(self.env.site_num):
                site = self.env.sites[site_s]
                weight = 0
                weight += self.env.tx_user_node[user_c][site_s] * self.env.data_size[0]
                weight += 1 / site.service_rate_a * 1000        # ms
                weight += self.phi_cs_arr[user_c][site_s]       # 当 self.consider_cost_tq == False 时，这一项是0
                self.graph["u1{}".format(user_c)]["s2{}".format(site_s)]['weight'] = weight

        for s in range(self.env.site_num):
            for spi in range(self.env.site_num):
                weight = self.env.tx_node_node[s][spi] * self.env.data_size[1]
                self.graph["s2{}".format(s)]["s3{}".format(spi)]['weight'] = weight

        for site_spi in range(self.env.site_num):
            site = self.env.sites[site_spi]
            for user_cpi in c1_pi:
                weight = 0
                weight += self.env.tx_node_user[site_spi][user_cpi] * self.env.data_size[2]
                weight += 1 / site.service_rate_r * 1000        # ms
                weight += self.phi_cpi_spi_arr[user_cpi][site_spi]
                self.graph["s3{}".format(site_spi)]["u4{}".format(user_cpi)]['weight'] = weight

        self.reversed_graph = nx.reverse(self.graph, copy=True)

    def get_all_stp_max_heap(self, c1: set, c1_pi: set):
        self.shortest_paths.clear()
        self.shortest_path_lengths = heapdict()

        for user_c in c1:
            self.shortest_paths["u1{}".format(user_c)] = dict()

            lens, paths = nx.single_source_dijkstra(self.graph, "u1{}".format(user_c))

            for user_cpi in c1_pi:
                self.shortest_paths["u1{}".format(user_c)]["u4{}".format(user_cpi)] = paths["u4{}".format(user_cpi)]
                self.shortest_path_lengths[("u1{}".format(user_c), "u4{}".format(user_cpi))] = -lens["u4{}".format(user_cpi)]   # 大根堆

    def get_all_stp(self, c1: set, c1_pi: set):
        self.shortest_paths.clear()
        self.shortest_path_lengths = heapdict()

        for user_c in c1:
            self.shortest_paths["u1{}".format(user_c)] = dict()

            lens, paths = nx.single_source_dijkstra(self.graph, "u1{}".format(user_c))

            for user_cpi in c1_pi:
                self.shortest_paths["u1{}".format(user_c)]["u4{}".format(user_cpi)] = paths["u4{}".format(user_cpi)]
                self.shortest_path_lengths[("u1{}".format(user_c), "u4{}".format(user_cpi))] = lens["u4{}".format(user_cpi)] * self.psi_arr[user_c][user_cpi]

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
            need_assign_uj_r = True if user_j.service_r.node_id is None else False  # 本轮是否要决策 user_j 的服务R

            if need_assign_ui_a:
                target_site = self.env.sites[int(min_stp[1][2:])]
                # self.DEBUG("[Assign] user {} service_a --> site {}".format(user_i.user_id, target_site.global_id))
                user_i.service_a.node_id = target_site.global_id
                user_i.service_a.service_rate = target_site.service_rate_a
                user_i.service_a.price = target_site.price_a
                if self.consider_cost_tq:
                    user_i.service_a.update_num_server(self.best_x_serviceA[user_i.user_id][target_site.global_id])
                else:
                    user_i.service_a.queuing_delay = 0
                    user_i.service_a.num_server = 0

                service_a_finished_flag[user_i.user_id] = True
                assignment_times += 1

            if need_assign_uj_r:
                target_site = self.env.sites[int(min_stp[-2][2:])]
                # self.DEBUG("[Assign] user {} service_r --> site {}".format(user_j.user_id, target_site.global_id))
                user_j.service_r.node_id = target_site.global_id
                user_j.service_r.service_rate = target_site.service_rate_r
                user_j.service_r.price = target_site.price_r
                if self.consider_cost_tq:
                    user_j.service_r.update_num_server(self.best_x_serviceR[user_j.user_id][target_site.global_id])
                else:
                    user_j.service_r.queuing_delay = 0
                    user_j.service_r.num_server = 0

                service_r_finished_flag[user_j.user_id] = True
                assignment_times += 1

            self.shortest_paths[node_i].pop(node_j)
            self.shortest_path_lengths.__delitem__((node_i, node_j))

            """ 将冲突边删除 """
            if need_assign_ui_a:
                for k in range(self.env.site_num):
                    node_name = "s2{}".format(k)
                    if node_name != min_stp[1]:
                        self.graph.remove_edge(node_i, node_name)
                        self.reversed_graph.remove_edge(node_name, node_i)
            if need_assign_uj_r:
                for k in range(self.env.site_num):
                    node_name = "s3{}".format(k)
                    if node_name != min_stp[-2]:
                        self.graph.remove_edge(node_name, node_j)
                        self.reversed_graph.remove_edge(node_j, node_name)

            """ 修改被影响的最短路径 """
            if need_assign_ui_a:
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.graph, node_i)
                for k in c1_pi:
                    target_node = "u4{}".format(k)
                    if target_node != node_j:
                        self.shortest_path_lengths[(node_i, target_node)] = temp_stp_lengths[target_node] * self.psi_arr[user_i.user_id][k]
                        self.shortest_paths[node_i][target_node] = temp_stp_paths[target_node]

            if need_assign_uj_r:
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.reversed_graph, node_j)
                for k in c1:
                    target_node = "u1{}".format(k)
                    if target_node != node_i:
                        self.shortest_path_lengths[(target_node, node_j)] = temp_stp_lengths[target_node] * self.psi_arr[k][user_j.user_id]
                        self.shortest_paths[target_node][node_i] = reversed(temp_stp_paths[target_node])

    def get_min_stp(self, service_a_finished_flag, service_r_finished_flag):
        user_pair, min_stp_len, target_stp = None, -1, None
        while len(self.shortest_path_lengths) > 0:
            user_pair, min_stp_len = self.shortest_path_lengths.peekitem()

            user_i = int(user_pair[0].replace("u1", ""))
            user_j = int(user_pair[1].replace("u4", ""))
            if service_a_finished_flag[user_i] and service_r_finished_flag[user_j]:
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

    def update_psi_matrix(self):
        # 计算f值最大的用户对
        max_user_pair, max_f = self.compute_max_latency_and_weighted_cost_user_pair()
        if max_user_pair in self.important_user_pairs:
            self.DEBUG("user_pair {} is already in important_user_pairs!".format(max_user_pair))
        self.DEBUG("max_user_pair = {}, max_f = {}".format(max_user_pair, max_f))

        self.important_user_pairs.append(max_user_pair)

        # 更新psi
        self.psi_arr = np.zeros((self.env.user_num, self.env.user_num))
        size = len(self.important_user_pairs)
        weight = 1 / size
        for up in self.important_user_pairs:
            self.psi_arr[up[0]][up[1]] += weight

    def compute_max_latency_and_weighted_cost_user_pair(self):
        user_pair = None
        max_f = -1
        for i in range(self.n):
            user_from = self.env.users[i]       # type: UserNode

            for j in range(self.n):
                user_to = self.env.users[j]     # type: UserNode

                f = 0
                f += self.env.tx_user_node[i][user_from.service_a.node_id] * self.env.data_size[0]    # ms
                f += 1 / user_from.service_a.service_rate * 1000                                      # ms
                f += user_from.service_a.queuing_delay * 1000                                         # ms
                f += self.env.eta * (user_from.service_a.price * user_from.service_a.num_server)

                f += self.env.tx_node_node[user_from.service_a.node_id][user_to.service_r.node_id] * self.env.data_size[1]

                f += self.env.tx_node_user[user_to.service_r.node_id][j] * self.env.data_size[2]
                f += 1 / user_to.service_r.service_rate * 1000
                f += user_to.service_r.queuing_delay * 1000
                f += self.env.eta * (user_to.service_r.price * user_to.service_r.num_server)

                if f > max_f:
                    max_f = f
                    user_pair = (i, j)
        return user_pair, max_f

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)

if __name__ == "__main__":
    from env.environment_old import Environment
    import random
    from configuration.config import config as conf
    from min_max.nearest import NearestAlgorithm
    from min_max.min_max_ours_v2 import MinMaxOurs_V2 as MinMaxOurs
    from min_max.MGreedy import MGreedyAlgorithm
    from min_avg.min_avg_ours import MinAvgOurs
    from min_max.stp_max_first import StpMaxFirst

    print("==================== env  config ===============================")
    print(conf)
    print("================================================================")

    # seed = random.randint(0, 100000)
    env_seed = 58972            # seed = 999734539, user_num = 30 曲线好看。
    # env_seed = 99497

    print("env_seed: ", env_seed)

    num_user = 50

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
        # u_seed = 4486628981   8010712550
        print("user_seed = {}".format(u_seed))

        print("------------- Nearest ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        nearest_alg = NearestAlgorithm(env)
        nearest_alg.run()
        print(nearest_alg.get_results())

        # print("------------- Min-Avg Algorithm ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # min_avg_alg = MinAvgOurs(env, consider_cost_tq=True, stable_only=False)
        # min_avg_alg.run()
        # print(min_avg_alg.get_results())
        # temp_f, temp_max_delay, temp_avg_cost = min_avg_alg.env.compute_target_function_value("min-max")
        # print("f_value = {:.4f}, max_delay = {:.4f} ms, avg_cost = {:.4f}".format(temp_f, temp_max_delay * 1000, temp_avg_cost))

        print("------------- Max Stp First ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        max_first_alg = StpMaxFirst(env, consider_cost_tq=True, stable_only=False)
        max_first_alg.run()
        print(max_first_alg.get_results())
        #
        # print("------------- MGreedy ------------------------")
        # env = Environment(conf, env_seed)
        # env.reset(num_user=num_user, user_seed=u_seed)
        # mg_alg = MGreedyAlgorithm(env, consider_cost_tq=True, stable_only=False)
        # mg_alg.run()
        # print(mg_alg.get_results())

        print("------------- Ours ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        surrogate_alg = MinMaxSurrogate(env)
        surrogate_alg.debug_flag = True
        surrogate_alg.alpha = 5e-5

        surrogate_alg.run()
        print(surrogate_alg.get_results())
        print("[iterations = {}, best_iteration = {}]".format(surrogate_alg.total_iterations, surrogate_alg.best_iteration))
        draw_fg(surrogate_alg.f_values, surrogate_alg.g_values)