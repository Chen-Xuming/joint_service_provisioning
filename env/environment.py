import copy
import random
from numpy.random import default_rng
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

from env.site_node import SiteNode
from env.user_node import UserNode
from env.area import Area
from env.service import Service

from configuration.config import config

class Environment:
    def __init__(self, env_conf, env_seed):
        self.conf = env_conf
        self.env_rng = default_rng(env_seed)

        self.debug_flag = False

        """ ------------ Network Topo --------------- """
        self.topo = nx.Graph()      # 无向图，边是节点之间的传输时延
        self.sites = []
        self.common_site_positions = []    # 坐标
        self.edge_site_positions = []
        self.areas = []

        self.common_site_num = self.conf["common_area_site_num"]
        self.edge_site_num = self.conf["edge_area_site_num"]
        self.edge_num = self.conf["edge_area_num"]
        self.site_num = self.common_site_num + self.edge_site_num * self.edge_num

        """ ------------ Environment ---------------- """
        self.tx_node_node = np.zeros((self.site_num, self.site_num))    # 节点到节点之间的传输时延（基于最短路径）
        self.tx_node_user = None    # 节点到用户的传输时延
        self.tx_user_node = None    # 用户到节点的传输时延（跟tx_node_user矩阵仅是维度不同）

        self.data_size = self.conf["data_size"]  # 传输数据量

        self.eta = self.conf["eta"]

        self.user_num = None    # 用户数量待定，可变化
        self.users = []
        self.user_positions = []  # 坐标
        self.topo_with_users = None

        # 用户到达率取值范围
        self.total_arrival_rate = None

        self.init_topo()
        self.build_tx_matrix_node_to_node()
        self.init_site_attr()


    """
        初始化网络拓扑
    """
    def init_topo(self):
        self.generate_common_sites()
        self.generate_edge_sites()

        # 检查图的连通性
        assert nx.is_connected(self.topo), "The graph is not connected"

        if self.conf["show_graph"]:
            positions = self.common_site_positions + self.edge_site_positions
            nx.draw(self.topo, pos=positions, with_labels=True)
            nx.draw_networkx_nodes(self.topo, pos=nx.get_node_attributes(self.topo, 'pos'), node_color=list(nx.get_node_attributes(self.topo, 'color')))
            nx.draw_networkx_edge_labels(self.topo, nx.get_node_attributes(self.topo, 'pos'), edge_labels=nx.get_edge_attributes(self.topo, 'weight'))
            # nx.draw_networkx_edges(self.topo, nx.get_node_attributes(self.topo, 'pos'))
            plt.axis('equal')
            plt.show()

    """ 生成common节点 """
    def generate_common_sites(self):
        common_area = Area(0, (0, 0))
        self.areas.append(common_area)

        center = (0, 0)
        radius = self.conf["global_radius"]
        min_distance = self.conf["global_min_distance"]

        # 距离变换
        def transform(length):
            delay = length * 1.8  # fixme
            return int(delay)

        num_points = self.common_site_num

        sid = 0
        while sid < num_points:
            # 生成一个随机半径和极坐标角度
            r = self.env_rng.uniform(0, radius)
            angle = self.env_rng.uniform(0, 2 * math.pi)
            # 计算随机点的坐标
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            # 检查新点与已有点之间的距离
            too_close = False
            for (x_existing, y_existing) in self.common_site_positions:
                if self.distance((x, y), (x_existing, y_existing)) < min_distance:
                    too_close = True
                    break
            if not too_close:
                self.common_site_positions.append((x, y))
                site = SiteNode(sid, sid, 0, pos=(x, y))
                self.sites.append(site)
                common_area.site_nodes.append(site)
                self.topo.add_node(sid, pos=(x, y), color="red")

                sid += 1

        """ 每个节点与其最近的若干个节点生成边（双向边） """
        distances = {}
        for i in range(self.common_site_num):
            distances[i] = {}
            coor_i = self.common_site_positions[i]
            for j in range(self.common_site_num):
                coor_j = self.common_site_positions[j]
                dis = np.linalg.norm(np.array(coor_i) - np.array(coor_j))
                distances[i][j] = dis
        for u in range(self.common_site_num):
            sorted_distances = sorted(distances[u].items(), key=lambda x: x[1])
            nearest_neighbors = sorted_distances[1:4]
            for v, d in nearest_neighbors:
                self.topo.add_edge(u, v, weight=transform(d))

    """ 
        生成边缘节点 
        每次随机生成一个坐标，表示一个 edge-area 的位置pos，这个edge-area内的节点与pos最近的两个common节点直连.
        生成各个边的传输时延.
    """
    def generate_edge_sites(self):
        center = (0, 0)
        radius = self.conf["global_radius"]
        edge_radius = self.conf["edge_radius"]

        count = 0
        edge_id = 1
        global_id = self.common_site_num
        exist_set = set()
        while count < self.edge_num:
            # 生成一个随机半径和极坐标角度
            r = self.env_rng.uniform(0, radius)
            angle = self.env_rng.uniform(0, 2 * math.pi)
            # 计算随机点的坐标
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)

            # 找出与其最近的两个common节点
            dis_to_common = []
            for j, comm_pos in enumerate(self.common_site_positions):
                dis = self.distance((x, y), comm_pos)
                dis_to_common.append((j, dis))
            dis_to_common = sorted(dis_to_common, key=lambda x: x[1], reverse=False)

            link_common_sites = (dis_to_common[0][0], dis_to_common[1][0])
            if link_common_sites in exist_set:
                continue
            else:
                exist_set.add(link_common_sites)

            # 生成edge-site，它们互相直连，而且与上面找出的节点直连
            edge_center = (x, y)
            edge_area = Area(edge_id, edge_center)
            self.areas.append(edge_area)
            for j in range(self.edge_site_num):
                # 生成一个随机半径和极坐标角度
                r = self.env_rng.uniform(0, edge_radius)
                angle = self.env_rng.uniform(0, 2 * math.pi)
                # 计算随机点的坐标
                x = edge_center[0] + r * math.cos(angle)
                y = edge_center[1] + r * math.sin(angle)

                site = SiteNode(global_id, j, edge_id, (x, y))
                self.edge_site_positions.append((x, y))
                self.sites.append(site)
                edge_area.site_nodes.append(site)
                self.topo.add_node(global_id, pos=(x, y), color='green')
                global_id += 1

            for j in range(self.edge_site_num):
                for k in range(j + 1, self.edge_site_num):
                    tx = self.env_rng.integers(self.conf["tx_edge_edge_range"][0],
                                               self.conf["tx_edge_edge_range"][1]+1)
                    self.topo.add_edge(edge_area.site_nodes[j].global_id, edge_area.site_nodes[k].global_id, weight=tx)
                for c in link_common_sites:
                    tx = self.env_rng.integers(self.conf["tx_edge_common_range"][0],
                                               self.conf["tx_edge_common_range"][1] + 1)
                    self.topo.add_edge(edge_area.site_nodes[j].global_id, c, weight=tx)

            edge_id += 1
            count += 1

    def distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    """ 构建节点间的传输时延矩阵 """
    def build_tx_matrix_node_to_node(self):
        len_path = nx.all_pairs_dijkstra(self.topo)
        for node, (len_, path_) in len_path:
            for i in range(self.site_num):
                self.tx_node_node[node][i] = len_[i]

    """ 初始化各个节点的服务率、处理时延和服务单价 """
    def init_site_attr(self):
        # 初始化 common site 的各个属性
        for site in self.areas[0].site_nodes:   # type: SiteNode
            site.service_rate_a = self.env_rng.integers(self.conf["minmax_service_rate_A_common"][0],
                                                        self.conf["minmax_service_rate_A_common"][1] + 1)
            site.service_rate_r = self.env_rng.integers(self.conf["minmax_service_rate_R_common"][0],
                                                        self.conf["minmax_service_rate_R_common"][1] + 1)
            site.tp_a = 1 / site.service_rate_a     # 秒
            site.tp_r = 1 / site.service_rate_r
            site.price_a = round(self.env_rng.uniform(self.conf["minmax_price_A_common"][0], self.conf["minmax_price_A_common"][1]), 1)
            site.price_r = round(self.env_rng.uniform(self.conf["minmax_price_R_common"][0], self.conf["minmax_price_R_common"][1]), 1)

        # 初始化 edge-site 的各个属性
        for area in self.areas[1:]:     # type: Area
            for site in area.site_nodes:
                site.service_rate_a = self.env_rng.integers(self.conf["minmax_service_rate_A_edge"][0],
                                                            self.conf["minmax_service_rate_A_edge"][1] + 1)
                site.service_rate_r = self.env_rng.integers(self.conf["minmax_service_rate_R_edge"][0],
                                                            self.conf["minmax_service_rate_R_edge"][1] + 1)
                site.tp_a = 1 / site.service_rate_a  # 秒
                site.tp_r = 1 / site.service_rate_r
                site.price_a = round(self.env_rng.uniform(self.conf["minmax_price_A_edge"][0], self.conf["minmax_price_A_edge"][1]), 1)
                site.price_r = round(self.env_rng.uniform(self.conf["minmax_price_R_edge"][0], self.conf["minmax_price_R_edge"][1]), 1)

    """ 重置用户分布 """
    def reset(self, num_user, user_seed):
        self.total_arrival_rate = 0
        self.user_num = num_user
        self.users.clear()
        self.user_positions.clear()
        self.topo_with_users = copy.deepcopy(self.topo)
        self.tx_user_node = np.zeros((self.user_num, self.site_num))
        self.tx_node_user = np.zeros((self.site_num, self.user_num))
        self.generate_random_users(num_user, user_seed)

    """ 
    生成随机用户 
    随机将用户分配到某个edge-area，这个用户与所属区域内的节点有一条直连的边.
    """
    def generate_random_users(self, num_user, user_seed):
        user_rng = default_rng(user_seed)

        self.total_arrival_rate = 0
        for i in range(num_user):
            area_id = user_rng.integers(1, self.edge_num + 1)
            user = UserNode(i, area_id)
            user.arrival_rate = user_rng.integers(self.conf["minmax_arrival_rate"][0],
                                                  self.conf["minmax_arrival_rate"][1] + 1)
            self.total_arrival_rate += user.arrival_rate

            user.service_a = Service("A", i)
            user.service_r = Service("R", i)
            user.service_a.arrival_rate = user.arrival_rate

            self.topo_with_users.add_node(user.node_name)
            for site in self.areas[area_id].site_nodes:     # type: SiteNode
                tx = user_rng.integers(self.conf["tx_user_edge_range"][0], self.conf["tx_user_edge_range"][1] + 1)
                self.topo_with_users.add_edge(user.node_name, site.global_id, weight=tx)

            self.users.append(user)

        self.DEBUG("total_arrival_rate = {}".format(self.total_arrival_rate * self.conf["trigger_probability"]))
        # 各个Service R的到达率
        for u in self.users:        # type: UserNode
            u.service_r.arrival_rate = self.total_arrival_rate * self.conf["trigger_probability"]

        # 基于最短路径构建用户与节点的时延矩阵
        # 为了避免最短路径可能经过其它用户节点的情况，不能直接在包含用户节点的图上求最短路径
        for user in self.users:         # type: UserNode
            # 在 topo 上临时添加与该用户相关的点和边
            self.topo.add_node(user.node_name)
            for site in self.areas[user.area_id].site_nodes:        # type: SiteNode
                tx = self.topo_with_users[user.node_name][site.global_id]['weight']
                self.topo.add_edge(user.node_name, site.global_id, weight=tx)

            for site in self.sites:     # type: SiteNode
                stp_len = nx.dijkstra_path_length(self.topo, user.node_name, site.global_id)
                self.tx_user_node[user.user_id][site.global_id] = stp_len
                self.tx_node_user[site.global_id][user.user_id] = stp_len

            # 撤销临时点和边
            for site in self.areas[user.area_id].site_nodes:        # type: SiteNode
                self.topo.remove_edge(user.node_name, site.global_id)
            self.topo.remove_node(user.node_name)

    """ 
        计算系统开销
    """
    def compute_cost(self):
        cost = 0
        for user in self.users:     # type: UserNode
            cost += user.service_a.num_server * self.sites[user.service_a.node_id].price_a
            cost += user.service_r.num_server * self.sites[user.service_r.node_id].price_r
        return round(cost)

    def compute_average_cost(self):
        cost = self.compute_cost()
        avg_cost = cost / self.user_num
        return avg_cost

    """
        计算目标函数值(min-avg / min-max)
    """
    def compute_target_function_value(self, target):
        if target == "min-max":
            _, _, max_delay = self.compute_max_interactive_delay(self.users)
            avg_cost = self.compute_average_cost()
            func_value = max_delay * 1000 + self.conf["eta"] * avg_cost     # 这里的时延单位转化为ms
            return func_value, max_delay, avg_cost

        elif target == "min-avg":
            avg_delay = self.compute_avg_interactive_delay()
            avg_cost = self.compute_average_cost()
            func_value = avg_delay * 1000 + self.conf["eta"] * avg_cost
            return func_value, avg_delay, avg_cost

        else:
            raise Exception("Target type invalid.")

    """
        计算两个用户之间的交互时延.
        传输时延需要乘data_size.
    """
    def compute_interactive_delay(self, user_from: UserNode, user_to: UserNode) -> float:
        # 传输时延(ms)
        transmission_delay = self.tx_user_node[user_from.user_id][user_from.service_a.node_id] * self.data_size[0] + \
                             self.tx_node_node[user_from.service_a.node_id][user_to.service_r.node_id] * self.data_size[1] + \
                             self.tx_node_user[user_to.service_r.node_id][user_to.user_id] * self.data_size[2]
        transmission_delay = transmission_delay / 1000  # 将单位换算为秒

        # 排队时延(s)
        queuing_delay = user_from.service_a.queuing_delay + user_to.service_r.queuing_delay

        # 处理时延(s)
        processing_delay = 1 / user_from.service_a.service_rate + 1 / user_to.service_r.service_rate

        interactive_delay = transmission_delay + queuing_delay + processing_delay
        return interactive_delay

    """ 
        计算最大交互时延
    """
    def compute_max_interactive_delay(self, assigned_user_list: list) -> (UserNode, UserNode, float):
        max_delay = -1
        user_pair = (-1, -1)

        for user_from in assigned_user_list:  # type: UserNode
            for user_to in assigned_user_list:  # type: UserNode
                delay = self.compute_interactive_delay(user_from, user_to)
                if delay > max_delay:
                    max_delay = delay
                    user_pair = (user_from, user_to)

        return user_pair[0], user_pair[1], max_delay

    """
        计算平均交互时延
    """
    def compute_avg_interactive_delay(self):
        total_delay = 0
        user_pair_count = self.user_num * self.user_num
        for user_from in self.users:        # type: UserNode
            for user_to in self.users:      # type: UserNode
                delay = self.compute_interactive_delay(user_from, user_to)
                total_delay += delay

        avg_delay = total_delay / user_pair_count
        return avg_delay

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)


if __name__ == "__main__":
    # env_seed = random.randint(0, 100000)
    env_seed = 58972
    print("env_seed: ", env_seed)
    env = Environment(config, env_seed)

    useed = random.randint(0, 10000000000)
    print("user_seed: ", useed)

    num_user = 50
    env.reset(num_user=num_user, user_seed=useed)
# 58972, 52593, 54849, 2543