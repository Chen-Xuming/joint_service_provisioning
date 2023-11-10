from min_max.base import BaseMinMaxAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from env.service import Service
from time import time
import networkx as nx
import numpy as np

from heapdict import heapdict


class MinAvgForMinMax(BaseMinMaxAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinMaxAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_avg_for_min_max" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only      # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.graph = nx.DiGraph()  # 有向图
        self.reversed_graph = None  # graph的反向图

        self.shortest_paths = dict()
        self.shortest_path_lengths = heapdict()

        self.best_x_serviceA = np.zeros((self.env.user_num, self.env.site_num))  # 存储每个服务卸载到各个站点，性价比最大时对应服务器个数
        self.best_x_serviceR = np.zeros((self.env.user_num, self.env.site_num))

        self.debug_flag = False

        self.build_graph_time = 0
        self.get_all_stp_time = 0
        self.get_max_stp_total_running_time = 0
        self.assign_user_time = 0
        self.allocation_time = 0

        # 以下值仅用于debug
        self.max_delay_arr = []
        self.avg_cost_arr = []
        self.f_arr = []

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
        self.DEBUG("on_local = {}, on_common = {}".format(self.local_count, self.common_count))

        self.DEBUG("build-graph-time: {}ms".format(self.build_graph_time * 1000))
        self.DEBUG("get-all-stp-time: {}ms".format(self.get_all_stp_time * 1000))
        self.DEBUG("assign_user_time: {}ms".format(self.assign_user_time * 1000))
        self.DEBUG("get-max-stp-running-time: {}ms".format(self.get_max_stp_total_running_time * 1000))
        self.DEBUG("allocation-time: {}ms".format(self.allocation_time * 1000))

    def solve(self):
        self.build_graph()
        self.get_all_stp()
        self.assign_users()

        # self.server_allocation()

    """
        第一列节点是用户节点 --->  u1i, i = user_id
        第二列节点是所有site节点 ---> s2j, j = site_id
        第三列节点是第二列节点的拷贝 ---> s3j, j = site_id
        第四列节点是第一列的拷贝 ---> u4i, i = user_id
    """
    def build_graph(self):
        start = time()

        # 添加节点
        for user in self.env.users:  # type: UserNode
            self.graph.add_node("u1{}".format(user.user_id))
            self.graph.add_node("u4{}".format(user.user_id))
        for site in self.env.sites:  # type: SiteNode
            self.graph.add_node("s2{}".format(site.global_id))
            self.graph.add_node("s3{}".format(site.global_id))

        """
            添加边，权重如下(时延统一使用ms为单位):
            第一列到第二列的权重是 tx * data_size + tp_a + tq(性价比最大) + N * eta * price(单价*个数)
            第二列到第三列的权重是 tx * data_size
            第三列到第四列的权重是 tx * data_size + tp_r + tq(性价比最大) + N * eta * price(单价*个数)
        """
        for i in range(self.env.user_num):
            user_i = self.env.users[i]  # type: UserNode
            for j in range(self.env.site_num):
                site_j = self.env.sites[j]  # type: SiteNode

                weight = 0
                weight += self.env.tx_user_node[i][j] * self.env.data_size[0]       # TX
                weight += 1 / site_j.service_rate_a * 1000  # ms                    # TP

                # 当考虑排队时延、开销时，才把它们加上
                if self.consider_cost_tq:
                    # 暂时user_i的服务A卸载到节点site_j, 方便计算后面的值
                    service = user_i.service_a
                    service.assign_to_site(site_j)

                    if not self.stable_only:
                        x_a = self.resource_allocation(user_i.service_a.arrival_rate, user_i.service_a.service_rate, self.env.eta, user_i.service_a.price)
                    else:
                        x_a = user_i.service_a.get_num_server_for_stability(user_i.service_a.service_rate)
                    self.best_x_serviceA[i][j] = x_a  # 把最佳服务器个数记录下来，后面要用
                    service.update_num_server(x_a)

                    weight += service.queuing_delay * 1000  # 性价比最高时的排队时延（ms）
                    weight += self.env.eta * (service.price * service.num_server)
                    service.reset()

                self.graph.add_edge("u1{}".format(i), "s2{}".format(j), weight=weight)

        for i in range(self.env.site_num):
            site_i = self.env.sites[i]  # type: SiteNode
            for j in range(self.env.site_num):
                site_j = self.env.sites[j]  # type: SiteNode

                weight = self.env.tx_node_node[site_i.global_id][site_j.global_id] * self.env.data_size[1]
                self.graph.add_edge("s2{}".format(i), "s3{}".format(j), weight=weight)

        for j in range(self.env.site_num):
            site_j = self.env.sites[j]  # type: SiteNode
            for i in range(self.env.user_num):
                user_i = self.env.users[i]  # type: UserNode

                weight = 0
                weight += self.env.tx_node_user[j][i] * self.env.data_size[2]
                weight += 1 / site_j.service_rate_r * 1000  # ms

                if self.consider_cost_tq:
                    # 暂时user_i的服务R卸载到节点site_j, 方便计算后面的值
                    service = user_i.service_r
                    service.assign_to_site(site_j)

                    if not self.stable_only:
                        x_r = self.resource_allocation(user_i.service_r.arrival_rate, user_i.service_r.service_rate, self.env.eta, user_i.service_r.price)
                    else:
                        x_r = user_i.service_r.get_num_server_for_stability(user_i.service_r.service_rate)
                    self.best_x_serviceR[i][j] = x_r  # 把最佳服务器个数记录下来，后面要用
                    service.update_num_server(x_r)

                    weight += service.queuing_delay * 1000  # 性价比最高时的排队时延（ms）
                    weight += self.env.eta * (service.price * service.num_server)
                    service.reset()

                self.graph.add_edge("s3{}".format(j), "u4{}".format(i), weight=weight)

        self.reversed_graph = nx.reverse(self.graph, copy=True)

        end = time()
        self.build_graph_time = end - start

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
        C_old = x * B / (x - a + a * B)     # 排队时延

        while True:
            x = x + 1
            B = a * B / (x + a * B)
            C_new = x * B / (x - a + a * B)
            delta_t = C_old / ((x - 1 - a) * miu_) - C_new / ((x - a) * miu_)
            delta_t *= 1000
            if delta_t <= eta_f or (C_new * 1000 <= 1.0):       # fixme !
                break

            C_old = C_new

        x_star = x - 1
        return int(x_star)

    """ 求所有用户对之间的最短路径 """
    def get_all_stp(self):
        start = time()

        for i in range(self.env.user_num):
            self.shortest_paths["u1{}".format(i)] = dict()

            # 求解 u1i 节点到其它节点的最短路径和距离
            lens, paths = nx.single_source_dijkstra(self.graph, "u1{}".format(i))

            for j in range(self.env.user_num):
                self.shortest_paths["u1{}".format(i)]["u4{}".format(j)] = paths["u4{}".format(j)]
                self.shortest_path_lengths[("u1{}".format(i), "u4{}".format(j))] = lens["u4{}".format(j)]       # heapdict 是小根堆

        end = time()
        self.get_all_stp_time = end - start

    def assign_users(self):
        start = time()
        modify_stp_time = 0

        source_set = set()
        for u in range(self.env.user_num):
            source_set.add("u1{}".format(u))

        service_a_finished_flag = [False for _ in range(self.env.user_num)]
        service_r_finished_flag = [False for _ in range(self.env.user_num)]

        assignment_times = 0
        count_running_loop = 0
        no_need_count = 0
        while assignment_times < self.env.user_num * 2:
            count_running_loop += 1
            node_i, node_j, min_stp, min_len = self.get_min_stp(service_a_finished_flag, service_r_finished_flag)

            self.DEBUG("[min-stp-len = {}] ({}, {}): {}".format(min_len, node_i, node_j, min_stp))

            user_i = self.env.users[int(node_i[2:])]
            user_j = self.env.users[int(node_j[2:])]
            need_assign_ui_a = True if user_i.service_a.node_id is None else False  # 本轮是否要决策 user_i 的服务A
            need_assign_uj_r = True if user_j.service_r.node_id is None else False  # 本轮是否要决策 user_j 的服务R

            if need_assign_ui_a is None and need_assign_uj_r is None:
                no_need_count += 1

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
                self.DEBUG("[Assign] user {} service_a --> site {}".format(user_i.user_id, target_site.global_id))

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
                self.DEBUG("[Assign] user {} service_r --> site {}".format(user_j.user_id, target_site.global_id))

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

            """ 修改被影响的最短路径 version 2 """
            modify_stp_start = time()
            if need_assign_ui_a:
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.graph, node_i)
                for k in range(self.env.user_num):
                    target_node = "u4{}".format(k)
                    if target_node != node_j:
                        self.shortest_path_lengths[(node_i, target_node)] = temp_stp_lengths[target_node]
                        self.shortest_paths[node_i][target_node] = temp_stp_paths[target_node]

            if need_assign_uj_r:
                temp_stp_lengths, temp_stp_paths = nx.single_source_dijkstra(self.reversed_graph, node_j)
                for k in range(self.env.user_num):
                    target_node = "u1{}".format(k)
                    if target_node != node_i:
                        self.shortest_path_lengths[(target_node, node_j)] = temp_stp_lengths[target_node]
                        self.shortest_paths[target_node][node_i] = reversed(temp_stp_paths[target_node])

            modify_stp_end = time()
            modify_stp_time += modify_stp_end - modify_stp_start

            if self.debug_flag:
                self.record_values_for_debug(service_a_finished_flag, service_r_finished_flag)

        end = time()
        self.assign_user_time = end - start
        self.DEBUG("[assign_user] running in loop: {}".format(count_running_loop))
        self.DEBUG("[assign_user] no_need_count: {}".format(no_need_count))
        self.DEBUG("[assign_user] modify_stp_time: {}ms".format(modify_stp_time * 1000))

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

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)

    def record_values_for_debug(self, a_flags, r_flags):
        # 计算 max_delay
        max_delay = -1
        for ia, fa in enumerate(a_flags):
            if not fa:
                continue
            user_from = self.env.users[ia]  # type: UserNode
            for ir, fr in enumerate(r_flags):
                if not fr:
                    continue
                user_to = self.env.users[ir]  # type: UserNode

                delay = self.env.compute_interactive_delay(user_from, user_to)
                if delay > max_delay:
                    max_delay = delay
        max_delay = max_delay * 1000
        self.max_delay_arr.append(max_delay)

        # 计算 avg_cost
        total_cost = 0
        count = 0
        for ia, fa in enumerate(a_flags):
            if fa:
                user_from = self.env.users[ia]  # type: UserNode
                count += 1
                total_cost += user_from.service_a.num_server * user_from.service_a.price
        for ir, fr in enumerate(r_flags):
            if fr:
                user_to = self.env.users[ir]  # type: UserNode
                count += 1
                total_cost += user_to.service_r.num_server * user_to.service_r.price

        avg_cost = total_cost / count
        self.avg_cost_arr.append(avg_cost)

        f = max_delay + self.env.eta * avg_cost
        self.f_arr.append(f)

    def draw(self, target: str):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel(target)
        iters = len(self.f_arr)
        x = [i+1 for i in range(iters)]
        if target == "Max Latency":
            plt.plot(x, self.max_delay_arr, color='#58B272', marker='.')
        if target == "Average Cost":
            plt.plot(x, self.avg_cost_arr, color='#58B272', marker='.')
        if target == "F Value":
            plt.plot(x, self.f_arr, color='#58B272', marker='.')
        if target == "All":
            plt.ylabel("Values")
            plt.plot(x, self.max_delay_arr, color='green', marker='.', label="Max Latency")
            plt.plot(x, self.avg_cost_arr, color='blue', marker='.', label="Average Cost")
            plt.plot(x, self.f_arr, color='red', marker='.', label="F Value")
        leg = plt.legend()
        leg.set_draggable(state=True)
        plt.show()

