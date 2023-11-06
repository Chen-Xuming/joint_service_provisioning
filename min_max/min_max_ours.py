import copy
import math

from min_max.base import BaseMinMaxAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
import networkx as nx
import numpy as np

from heapdict import heapdict


class MinMaxOurs(BaseMinMaxAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinMaxAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_max_ours" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only      # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.backup_empty_graph = None
        self.graph = nx.DiGraph()  # 有向图
        self.reversed_graph = None  # graph的反向图

        self.shortest_paths = dict()
        self.shortest_path_lengths = heapdict()

        self.T_matrix = np.zeros((self.env.user_num, self.env.user_num))    # 次梯度（时延矩阵）

        self.best_x_serviceA = np.zeros((self.env.user_num, self.env.site_num))  # 存储每个服务卸载到各个站点，性价比最大时对应服务器个数
        self.best_x_serviceR = np.zeros((self.env.user_num, self.env.site_num))
        self.phi_cs_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.phi_cpi_spi_arr = np.zeros((self.env.user_num, self.env.site_num))

        # self.psi_cs_arr = np.zeros((self.env.user_num, self.env.site_num))          # 权重矩阵
        # self.psi_cpi_spi_arr = np.zeros((self.env.user_num, self.env.site_num))
        self.psi_c_arr = np.zeros(self.env.user_num)
        self.psi_c_pi_arr = np.zeros(self.env.user_num)

        self.n_2 = self.env.user_num ** 2

        self.alpha = 0.005
        self.epsilon = 5
        self.psi_arr = np.zeros((self.env.user_num, self.env.user_num))     # 权重矩阵
        for i in range(self.env.user_num):
            for j in range(self.env.user_num):
                self.psi_arr[i][j] = 1 / self.n_2

        # 最佳解
        self.best_solution = {
            "association": [],          # [(a1, r1), (a2, r2), ...]
            "service_amount": []        # [(xa1, xr1), (xa2, xr2), ...]
        }

        self.total_iterations = 0
        self.best_iteration = 0

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
        g_best = 0
        f_best = math.inf
        k = 0
        k_star = 0

        alpha_k = self.alpha
        update_alpha_step = 3
        update_cur_step = 1
        update_counter = 2

        while f_best - g_best >= self.epsilon:
        # max_iteration = 100
        # while k < max_iteration:
            k = k + 1
            self.DEBUG("----------- iteration {} ------------".format(k))

            # 计算 psi_c 和 psi_c_pi
            self.psi_c_arr = np.sum(self.psi_arr, axis=1)
            self.psi_c_pi_arr = np.sum(self.psi_arr, axis=0)
            assert abs(np.sum(self.psi_c_arr) - 1.0) < 1e-8
            assert abs(np.sum(self.psi_c_pi_arr) - 1.0) < 1e-8

            """ 1. Resource Allocation(找出最佳服务器个数) """
            if self.consider_cost_tq:
                self.resource_allocation()

            """ 2. Server Association """
            self.build_graph()
            self.set_edge_weight()
            self.get_all_stp()
            self.reset_all_service()
            self.assign_users()

            """ 3. Keep trace of the best solution found so far """
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
            self.DEBUG("[cur_max_delay, cur_avg_cost] = {:.4f} ms, {:.4f}".format(cur_max_delay * 1000, cur_avg_cost))
            self.DEBUG("[cur_f, best_f] = {:.4f}, {:.4f}, [cur_g, best_g] = {:.4f}, {:.4f}".format(current_f,
                                                                                                   f_best,
                                                                                                   current_g,
                                                                                                   g_best))

            """ 4. Sub-gradient Update """
            alpha_k = self.alpha / k
            alpha_k = max(alpha_k, self.alpha / 20)
            #
            # alpha_k = self.alpha

            # if update_cur_step == update_alpha_step:
            #     alpha_k = self.alpha / update_counter
            #     update_counter += 1
            #     update_cur_step = 1
            # else:
            #     update_cur_step += 1
            # alpha_k = max(alpha_k, self.alpha / 15)

            self.DEBUG("alpha_k = {}".format(alpha_k))
            for i in range(self.env.user_num):
                for j in range(self.env.user_num):
                    self.psi_arr[i][j] += alpha_k * self.T_matrix[i][j]

            """ 5. Projection """
            self.projection()
            self.check_psi_satisfy_constraint()

            if k == 100:
                break

        """ 还原最优解 """
        self.set_best_solution()

        self.total_iterations = k
        self.best_iteration = k_star

    def resource_allocation(self):
        # service A
        for i in range(self.env.user_num):
            user_i = self.env.users[i]  # type: UserNode
            for j in range(self.env.site_num):
                site_j = self.env.sites[j]  # type: SiteNode

                # 暂时user_i的服务A卸载到节点site_j, 方便计算后面的值
                service = user_i.service_a
                service.assign_to_site(site_j)
                if not self.stable_only:
                    x_cs, phi_cs = 0, 0
                    if self.psi_c_arr[i] > 0:
                        sigma = self.env.eta * service.price / (self.env.user_num * self.psi_c_arr[i])
                        x_cs, phi_cs = self.get_best_server_num(service.arrival_rate,
                                                                service.service_rate,
                                                                sigma)
                    elif self.psi_c_arr[i] == 0:
                        self.DEBUG("self.psi_c_arr[i] == 0")
                        x_cs = service.get_num_server_for_stability(service.service_rate)
                        # x_cs = service.get_num_server_for_stability(service.service_rate) + 2      # fixme: 避免刚好满足稳态条件使得排队时延很大

                    else:
                        raise Exception("psi < 0!")

                    self.phi_cs_arr[i][j] = phi_cs

                else:
                    x_cs = service.get_num_server_for_stability(service.service_rate)
                self.best_x_serviceA[i][j] = x_cs

        # service R
        for i in range(self.env.user_num):
            user_i = self.env.users[i]  # type: UserNode
            for j in range(self.env.site_num):
                site_j = self.env.sites[j]  # type: SiteNode

                service = user_i.service_r
                service.assign_to_site(site_j)
                if not self.stable_only:
                    x_cpi_spi, phi_cpi_spi = 0, 0
                    if self.psi_c_pi_arr[i] > 0:
                        sigma = self.env.eta * service.price / (self.env.user_num * self.psi_c_pi_arr[i])
                        x_cpi_spi, phi_cpi_spi = self.get_best_server_num(service.arrival_rate,
                                                                          service.service_rate,
                                                                          sigma)
                    elif self.psi_c_pi_arr[i] == 0:
                        self.DEBUG("self.psi_c_arr[i] == 0")
                        x_cpi_spi = service.get_num_server_for_stability(service.service_rate)
                        # x_cpi_spi = service.get_num_server_for_stability(service.service_rate) + 2      # fixme: 避免刚好满足稳态条件使得排队时延很大

                    else:
                        raise Exception("psi < 0!")

                    self.phi_cpi_spi_arr[i][j] = phi_cpi_spi
                else:
                    x_cpi_spi = service.get_num_server_for_stability(service.service_rate)
                self.best_x_serviceR[i][j] = x_cpi_spi

    """
        求解最佳服务器个数
    """
    @staticmethod
    # fixme !
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
        为图的边设置权重
    """
    def set_edge_weight(self):
        for i in range(self.env.user_num):
            for j in range(self.env.site_num):
                site_j = self.env.sites[j]  # type: SiteNode

                weight = 0
                weight += self.env.tx_user_node[i][j] * self.env.data_size[0]       # TX
                weight += 1 / site_j.service_rate_a * 1000  # ms                    # TP
                weight += self.phi_cs_arr[i][j]     # 当 self.consider_cost_tq == False 时，这一项是0

                self.graph["u1{}".format(i)]["s2{}".format(j)]['weight'] = weight

        for i in range(self.env.site_num):
            site_i = self.env.sites[i]  # type: SiteNode
            for j in range(self.env.site_num):
                site_j = self.env.sites[j]  # type: SiteNode

                weight = self.env.tx_node_node[site_i.global_id][site_j.global_id] * self.env.data_size[1]
                self.graph.add_edge("s2{}".format(i), "s3{}".format(j), weight=weight)      # fixme

        for j in range(self.env.site_num):
            site_j = self.env.sites[j]  # type: SiteNode
            for i in range(self.env.user_num):

                weight = 0
                weight += self.env.tx_node_user[j][i] * self.env.data_size[2]
                weight += 1 / site_j.service_rate_r * 1000  # ms
                weight += self.phi_cpi_spi_arr[i][j]

                self.graph.add_edge("s3{}".format(j), "u4{}".format(i), weight=weight)      # fixme

        self.reversed_graph = nx.reverse(self.graph, copy=True)

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

    """ 求所有用户对之间的最短路径 """
    # fixme: 对于 psi = 0 的时延对，它们的 phi = 0，这会影响最短路径的选择！（尽管长度乘以 psi 后等于 0）
    def get_all_stp(self):
        # 重置记录最短路径及其长度的数据结构
        # 注意：这里的长度是加权的！
        self.shortest_paths.clear()
        assert len(self.shortest_paths) == 0, "self.shortest_paths is not empty!"
        self.shortest_path_lengths = heapdict()
        assert self.shortest_path_lengths.__len__() == 0, "self.shortest_path_lengths is not empty!"

        for i in range(self.env.user_num):
            self.shortest_paths["u1{}".format(i)] = dict()

            # 求解 u1i 节点到其它节点的最短路径和距离
            lens, paths = nx.single_source_dijkstra(self.graph, "u1{}".format(i))

            for j in range(self.env.user_num):
                self.shortest_paths["u1{}".format(i)]["u4{}".format(j)] = paths["u4{}".format(j)]
                self.shortest_path_lengths[("u1{}".format(i), "u4{}".format(j))] = lens["u4{}".format(j)] * self.psi_arr[i][j]      # heapdict 是小根堆

    def assign_users(self):
        service_a_finished_flag = [False for _ in range(self.env.user_num)]
        service_r_finished_flag = [False for _ in range(self.env.user_num)]

        assignment_times = 0
        iteration = 0
        while assignment_times < self.env.user_num * 2:
            iteration += 1
            # self.DEBUG("[assign_users] iteration = {}".format(iteration))

            node_i, node_j, min_stp, min_len = self.get_min_stp(service_a_finished_flag, service_r_finished_flag)

            # self.DEBUG("[min-stp-len = {}] ({}, {}): {}".format(min_len, node_i, node_j, min_stp))

            user_i = self.env.users[int(node_i[2:])]
            user_j = self.env.users[int(node_j[2:])]
            need_assign_ui_a = True if user_i.service_a.node_id is None else False  # 本轮是否要决策 user_i 的服务A
            need_assign_uj_r = True if user_j.service_r.node_id is None else False  # 本轮是否要决策 user_j 的服务R

            if need_assign_ui_a:
                target_site = self.env.sites[int(min_stp[1][2:])]
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
                # self.DEBUG("[Assign] user {} service_a --> site {}".format(user_i.user_id, target_site.global_id))

            if need_assign_uj_r:
                target_site = self.env.sites[int(min_stp[-2][2:])]
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
                # self.DEBUG("[Assign] user {} service_r --> site {}".format(user_j.user_id, target_site.global_id))

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
                for k in range(self.env.user_num):
                    target_node = "u4{}".format(k)
                    if target_node != node_j:
                        self.shortest_path_lengths[(node_i, target_node)] = temp_stp_lengths[target_node] * self.psi_arr[user_i.user_id][k]     # fixme: 不一定对
                        self.shortest_paths[node_i][target_node] = temp_stp_paths[target_node]

            if need_assign_uj_r:
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.reversed_graph, node_j)
                for k in range(self.env.user_num):
                    target_node = "u1{}".format(k)
                    if target_node != node_i:
                        self.shortest_path_lengths[(target_node, node_j)] = temp_stp_lengths[target_node] * self.psi_arr[k][user_j.user_id]      # fixme: 不一定对
                        self.shortest_paths[target_node][node_i] = reversed(temp_stp_paths[target_node])

    def reset_all_service(self):
        for user in self.env.users:     # type: UserNode
            user.service_a.reset()
            user.service_r.reset()

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

    def compute_g_value(self):
        total = 0
        for i, user_from in enumerate(self.env.users):    # type: UserNode
            for j, user_to in enumerate(self.env.users):  # type: UserNode
                weighted_delay = self.T_matrix[i][j] * self.psi_arr[i][j]
                total += weighted_delay

        avg_cost = self.env.compute_average_cost()
        total += self.env.eta * avg_cost
        return total

    def compute_sub_gradient(self):
        self.T_matrix = np.zeros((self.env.user_num, self.env.user_num))
        for user_from in self.env.users:        # type: UserNode
            for user_to in self.env.users:      # type: UserNode
                delay = self.env.compute_interactive_delay(user_from, user_to) * 1000  # ms
                self.T_matrix[user_from.user_id][user_to.user_id] = delay

    def save_best_solution(self):
        self.best_solution["association"].clear()
        self.best_solution["service_amount"].clear()
        for user in self.env.users:     # type: UserNode
            self.best_solution["association"].append((user.service_a.node_id, user.service_r.node_id))
            self.best_solution["service_amount"].append((user.service_a.num_server, user.service_r.num_server))

    def check_psi_satisfy_constraint(self):
        assert abs(np.sum(self.psi_arr) - 1.0) < 1e-8, "Sum of weight doesn't equal to 1."

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
    from env.environment import Environment
    import random
    from configuration.config import config as conf
    from min_max.nearest import NearestAlgorithm

    print("==================== env  config ===============================")
    print(conf)
    print("================================================================")

    # seed = random.randint(0, 100000)
    # env_seed = 58972
    env_seed = 99497
    print("env_seed: ", env_seed)

    num_user = 40

    def draw_fg(f_arr, g_arr):
        from matplotlib import pyplot as plt

        x_ = [(i+1) for i in range(len(f_arr))]
        plt.plot(x_, f_arr, marker='.', label='f')
        plt.plot(x_, g_arr, marker='*', label='g')
        plt.xlabel("iteration")
        plt.legend()
        plt.show()

    for i in range(1):
        print("========================= iteration {} ============================".format(i + 1))
        u_seed = random.randint(0, 10000000000)
        # u_seed = 5201314
        print("user_seed = {}".format(u_seed))

        print("------------- Nearest ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        nearest_alg = NearestAlgorithm(env)
        nearest_alg.run()
        print(nearest_alg.get_results())

        print("------------- Ours ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        our_alg = MinMaxOurs(env)
        our_alg.debug_flag = True
        our_alg.alpha = 5e-5
        our_alg.epsilon = 15
        our_alg.run()
        print(our_alg.get_results())
        print("[iterations = {}, best_iteration = {}]".format(our_alg.total_iterations, our_alg.best_iteration))

        draw_fg(our_alg.f_values, our_alg.g_values)

