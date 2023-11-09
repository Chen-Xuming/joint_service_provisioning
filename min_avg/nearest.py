from min_avg.base import BaseMinAvgAlgorithm
from env.site_node import SiteNode
from env.user_node import UserNode
from time import time
from env.service import Service


class NearestAlgorithm(BaseMinAvgAlgorithm):
    def __init__(self, env, consider_cost_tq=True, stable_only=False, *args, **kwargs):
        BaseMinAvgAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "nearest" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.consider_cost_tq = consider_cost_tq
        self.stable_only = stable_only      # 仅当 consider_cost_tq=True 时有效
        if self.env.eta == 0:
            self.consider_cost_tq = False
        if not self.consider_cost_tq:
            self.stable_only = False

        self.debug_flag = False

    def run(self):
        self.start_time = time()

        # 每个用户关联到与其最近的直连节点，并初始化满足稳态条件
        for user in self.env.users:
            min_tx = 10000000
            target_site = None
            for site in self.env.areas[user.area_id].site_nodes:    # type: SiteNode
                tx = self.env.tx_user_node[user.user_id][site.global_id]
                if tx < min_tx:
                    min_tx = tx
                    target_site = site

            user.service_a.node_id = target_site.global_id
            user.service_a.service_rate = target_site.service_rate_a
            user.service_a.price = target_site.price_a

            user.service_r.node_id = target_site.global_id
            user.service_r.service_rate = target_site.service_rate_r
            user.service_r.price = target_site.price_r

        if not self.consider_cost_tq:
            for user in self.env.users:   # type: UserNode
                user.service_a.num_server = 0
                user.service_a.queuing_delay = 0
                user.service_r.num_server = 0
                user.service_r.queuing_delay = 0

        # 资源分配
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

        self.get_running_time()
        self.get_target_value()
        # self.result_info()

        self.DEBUG("avg_delay = {}".format(self.avg_delay))
        self.DEBUG("final_cost = {}".format(self.final_avg_cost))
        self.DEBUG("target_value = {}".format(self.target_value))
        self.DEBUG("running_time = {}".format(self.running_time))

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

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)

if __name__ == "__main__":
    from env.environment_old import Environment
    import random
    from configuration.config import config as conf
    from min_avg.min_avg_ours import MinAvgOurs

    print("==================== env  config ===============================")
    print(conf)
    print("================================================================")

    # seed = random.randint(0, 100000)
    env_seed = 58972
    print("env_seed: ", env_seed)

    num_user = 100

    for i in range(10):
        print("========================= iteration {} ============================".format(i + 1))
        u_seed = random.randint(0, 10000000000)
        # u_seed = 132049593
        print("user_seed = {}".format(u_seed))

        print("------------- nearest ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        near_alg = NearestAlgorithm(env, consider_cost_tq=False, stable_only=False)
        near_alg.run()
        print(near_alg.get_results())

        print("------------- our ------------------------")
        env = Environment(conf, env_seed)
        env.reset(num_user=num_user, user_seed=u_seed)
        our_alg = MinAvgOurs(env, consider_cost_tq=False)
        our_alg.run()
        print(our_alg.get_results())