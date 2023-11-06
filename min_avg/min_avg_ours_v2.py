from min_avg.base import BaseMinAvgAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from env.service import Service
from time import time
import networkx as nx
import numpy as np


class MinAvgOursV2(BaseMinAvgAlgorithm):
    def __init__(self, env, *args, **kwargs):
        BaseMinAvgAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_avg_ours_v2" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.sorted_path_length = []

        self.graph = nx.DiGraph()  # 有向图

        self.best_x_serviceA = np.zeros((self.env.user_num, self.env.site_num))  # 存储每个服务卸载到各个站点，性价比最大时对应服务器个数
        self.best_x_serviceR = np.zeros((self.env.user_num, self.env.site_num))

        self.debug_flag = False

        self.build_graph_time = 0
        self.assign_user_time = 0

    def run(self):
        self.start_time = time()

        self.solve()

        self.get_running_time()
        self.get_target_value()
        # self.result_info()

        self.DEBUG("avg_delay = {}".format(self.avg_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))

        self.DEBUG("build-graph-time: {}ms".format(self.build_graph_time * 1000))
        self.DEBUG("assign_user_time: {}ms".format(self.assign_user_time * 1000))

    def solve(self):
        self.build_graph()
        self.record_all_simple_path()
        self.assign_users_v2()

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
                weight += self.env.tx_user_node[i][j] * self.env.data_size[0]
                weight += 1 / site_j.service_rate_a * 1000  # ms

                # 暂时user_i的服务A卸载到节点site_j, 方便计算后面的值
                service = user_i.service_a
                service.assign_to_site(site_j)

                x_a = self.resource_allocation(user_i.service_a.arrival_rate, user_i.service_a.service_rate, self.env.eta, user_i.service_a.price)
                self.best_x_serviceA[i][j] = x_a  # 把最佳服务器个数记录下来，后面要用
                service.update_num_server(x_a)


                weight += service.queuing_delay * 1000  # 性价比最高时的排队时延（ms）

                weight += self.env.eta * (service.price * service.num_server)

                self.graph.add_edge("u1{}".format(i), "s2{}".format(j), weight=weight)

                service.reset()

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

                # 暂时user_i的服务R卸载到节点site_j, 方便计算后面的值
                service = user_i.service_r
                service.assign_to_site(site_j)

                x_r = self.resource_allocation(user_i.service_r.arrival_rate, user_i.service_r.service_rate, self.env.eta, user_i.service_r.price)
                self.best_x_serviceR[i][j] = x_r  # 把最佳服务器个数记录下来，后面要用
                service.update_num_server(x_r)

                weight += service.queuing_delay * 1000  # 性价比最高时的排队时延（ms）

                weight += self.env.eta * (service.price * service.num_server)

                self.graph.add_edge("s3{}".format(j), "u4{}".format(i), weight=weight)

                service.reset()

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
            if delta_t <= eta_f or (C_new * 1000 <= 1.0):
                break

            C_old = C_new

        x_star = x - 1
        return int(x_star)

    """
        记录所有简单路径，及其长度. 并按长度排升序
    """
    def record_all_simple_path(self):
        for user_u in range(self.env.user_num):
            node_u = "u1{}".format(user_u)
            for user_v in range(self.env.user_num):
                node_v = "u4{}".format(user_v)
                for site_sa in range(self.env.site_num):
                    node_sa = "s2{}".format(site_sa)
                    for site_sr in range(self.env.site_num):
                        node_sr = "s3{}".format(site_sr)

                        length = 0
                        length += self.graph[node_u][node_sa]["weight"]
                        length += self.graph[node_sa][node_sr]["weight"]
                        length += self.graph[node_sr][node_v]["weight"]

                        path_ = (user_u, site_sa, site_sr, user_v)
                        self.sorted_path_length.append((path_, length))

        self.sorted_path_length = sorted(self.sorted_path_length, key=lambda x: x[1])

    def assign_users_v2(self):
        start = time()

        finished_count = 0
        assigned_services_a = [False for _ in range(self.env.user_num)]
        assigned_services_r = [False for _ in range(self.env.user_num)]
        for path, _ in self.sorted_path_length:
            c, s, s_, c_ = path

            if not assigned_services_a[c]:
                assigned_services_a[c] = True
                finished_count += 1
                user_c = self.env.users[c]
                site_s = self.env.sites[s]
                user_c.service_a.node_id = site_s.global_id
                user_c.service_a.service_rate = site_s.service_rate_a
                user_c.service_a.price = site_s.price_a
                user_c.service_a.update_num_server(self.best_x_serviceA[user_c.user_id][site_s.global_id])
                self.DEBUG("[Assign] user {} service_a --> site {}".format(user_c.user_id, site_s.global_id))

            if not assigned_services_r[c_]:
                assigned_services_r[c_] = True
                finished_count += 1
                user_c_ = self.env.users[c_]
                site_s_ = self.env.sites[s_]
                user_c_.service_r.node_id = site_s_.global_id
                user_c_.service_r.service_rate = site_s_.service_rate_r
                user_c_.service_r.price = site_s_.price_r
                user_c_.service_r.update_num_server(self.best_x_serviceR[user_c_.user_id][site_s_.global_id])
                self.DEBUG("[Assign] user {} service_r --> site {}".format(user_c_.user_id, site_s_.global_id))

            if finished_count == (2 * self.env.user_num):
                break

        end = time()
        self.assign_user_time = end - start

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)

if __name__ == "__main__":
    from env.environment import Environment
    import random
    from configuration.config import config as conf
    from min_avg.min_avg_ours import MinAvgOurs as MinAvgOursV1

    print("==================== env  config ===============================")
    print(conf)
    print("================================================================")

    # seed = random.randint(0, 100000)
    env_seed = 58972
    print("env_seed: ", env_seed)

    num_user = 100

    def check_same(val1, val2):
        return abs(val1 - val2) < 1e-3

    for i in range(10):
        print("========================= iteration {} ============================".format(i + 1))
        u_seed = random.randint(0, 10000000000)
        # u_seed = 132049593
        print("user_seed = {}".format(u_seed))

        print("------------- our_v1 ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        alg_v1 = MinAvgOursV1(env)
        alg_v1.run()
        print(alg_v1.get_results())

        print("------------- our_v2 ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        alg_v2 = MinAvgOursV2(env)
        alg_v2.run()
        print(alg_v2.get_results())

        assert check_same(alg_v1.avg_delay, alg_v2.avg_delay)
        assert check_same(alg_v1.final_avg_cost, alg_v2.final_avg_cost)
        assert check_same(alg_v1.target_value, alg_v2.target_value)


