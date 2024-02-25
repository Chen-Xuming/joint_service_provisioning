from min_avg.min_avg_ours import MinAvgOurs
from min_avg.nearest import NearestAlgorithm
from min_avg.modify_assignment import ModifyAssignmentAlgorithm
from min_avg.modify_assignment_v2 import ModifyAssignmentAlgorithm as ModifyAssignmentAlgorithm_V2
from min_avg.greedy_server_provisioning import GreedyServerProvisioningAlgorithm

from env.environment2 import Environment
from configuration.config import config as conf
from matplotlib import pyplot as plt
import random

env_seed = 99497
print("env_seed: ", env_seed)

num_user = 70

def get_offloading_distribution(algos):
    dist = []
    for algo in algos:  # type: MinAvgOurs
        n_tier1 = 0
        n_tier2 = 0
        for user in algo.env.users:
            if algo.env.sites[user.service_a.node_id].area_id == 0:
                n_tier2 += 1
            else:
                n_tier1 += 1

            if algo.env.sites[user.service_r.node_id].area_id == 0:
                n_tier2 += 1
            else:
                n_tier1 += 1
        dist.append([n_tier1, n_tier2])

    return dist

def find_offloading_difference(algos):
    print("========= offloading_difference ============")

    users = []

    def check(tiers, nodes, u):
        # if tiers[0] == (1, 1) and tiers[1] == (2, 2) and (tiers[2] == (1, 2) or tiers[2] == (1, 1)):
        if tiers[0] == (1, 1) and (tiers[1] == (2, 2) or tiers[1] == (1, 2)) and (tiers[2] == (2, 2) or tiers[2] == (1, 2))\
                and (tiers[3] == (1, 2) or tiers[3] == (1, 1)):
            users.append(u)
            print("User #{} tiers:{}, nodes:{}".format(u, tiers, nodes))

    for i in range(num_user):
        offload_tier = []
        offload_node = []
        for algo in algos:  # type: MinAvgOurs
            node_a_id = algo.env.users[i].service_a.node_id
            node_r_id = algo.env.users[i].service_r.node_id
            offload_node.append((node_a_id, node_r_id))
            tier_a = 2 if algo.env.sites[node_a_id].area_id == 0 else 1
            tier_r = 2 if algo.env.sites[node_r_id].area_id == 0 else 1
            offload_tier.append((tier_a, tier_r))

        # print("[user {}] tiers:{} nodes:{}".format(i, offload_tier, offload_node))
        check(offload_tier, offload_node, i)

def print_user_info_a(algos, uid):
    def get_total_tx_from(uid, alg:MinAvgOurs):
        tx = 0
        for u_to in alg.env.users:
            uf = alg.env.users[uid]
            ut = u_to

            site_a = uf.service_a.node_id
            site_r = ut.service_r.node_id
            tx += alg.env.tx_user_node[uf.user_id][site_a] * conf["data_size"][0]
            tx += alg.env.tx_node_node[site_a][site_r] * conf["data_size"][1]
            tx += alg.env.tx_node_user[site_r][ut.user_id] * conf["data_size"][2]
        return tx / alg.env.user_num

    print("========= user {} ============".format(uid))

    etas = [0.1, 0.25, 0.5, 0.75]
    for i, algo in enumerate(algos):
        print(" ---------- eta = {} ----------".format(etas[i]))

        sa = algo.env.users[uid].service_a
        node_a_id = algo.env.users[uid].service_a.node_id
        site_a = algo.env.sites[node_a_id]

        tier_a = 2 if algo.env.sites[node_a_id].area_id == 0 else 1
        print("location: node({}) tier({})".format(node_a_id, tier_a))

        print("n_serv = {}".format(sa.num_server))
        print("price = {}".format(sa.price))

        Tx_a = get_total_tx_from(uid, algo)
        print("Tx = {}".format(Tx_a))

        print("Tp = {}".format(round(site_a.tp_a * 1000, 3)))

        print("Tq = {}".format(round(sa.queuing_delay * 1000, 3)))

        H = sa.num_server * sa.price
        print("H = {}".format(H))

        print("eta * H = {}".format(etas[i] * H))

        T = Tx_a + site_a.tp_a * 1000 + sa.queuing_delay * 1000
        print("T = {}".format(T))

        total = etas[i] * H + T
        print("f = {}".format(total))

def print_user_info_a(algos, uid):
    def get_total_tx_from(uid, alg:MinAvgOurs):
        tx = 0
        for u_to in alg.env.users:
            uf = alg.env.users[uid]
            ut = u_to

            site_a = uf.service_a.node_id
            site_r = ut.service_r.node_id
            tx += alg.env.tx_user_node[uf.user_id][site_a] * conf["data_size"][0]
            tx += alg.env.tx_node_node[site_a][site_r] * conf["data_size"][1]
            tx += alg.env.tx_node_user[site_r][ut.user_id] * conf["data_size"][2]
        return tx / alg.env.user_num

    print("========= user {} ============".format(uid))

    etas = [0.1, 0.25, 0.5, 0.75]
    for i, algo in enumerate(algos):
        print(" ---------- eta = {} ----------".format(etas[i]))

        sa = algo.env.users[uid].service_a
        node_a_id = algo.env.users[uid].service_a.node_id
        site_a = algo.env.sites[node_a_id]

        tier_a = 2 if algo.env.sites[node_a_id].area_id == 0 else 1
        print("location: node({}) tier({})".format(node_a_id, tier_a))

        print("n_serv = {}".format(sa.num_server))
        print("price = {}".format(sa.price))

        Tx_a = get_total_tx_from(uid, algo)
        print("Tx = {}".format(Tx_a))

        print("Tp = {}".format(round(site_a.tp_a * 1000, 3)))

        print("Tq = {}".format(round(sa.queuing_delay * 1000, 3)))

        H = sa.num_server * sa.price
        print("H = {}".format(H))

        print("eta * H = {}".format(etas[i] * H))

        T = Tx_a + site_a.tp_a * 1000 + sa.queuing_delay * 1000
        print("T = {}".format(T))

        total = etas[i] * H + T
        print("f = {}".format(total))

def print_user_info(algos, uid):
    def get_total_tx_from(uid, alg:MinAvgOurs):
        tx = 0
        for u_to in alg.env.users:
            uf = alg.env.users[uid]
            ut = u_to

            site_a = uf.service_a.node_id
            site_r = ut.service_r.node_id
            tx += alg.env.tx_user_node[uf.user_id][site_a] * conf["data_size"][0]
            tx += alg.env.tx_node_node[site_a][site_r] * conf["data_size"][1]
            tx += alg.env.tx_node_user[site_r][ut.user_id] * conf["data_size"][2]
        return tx / alg.env.user_num

    def get_total_tx_to(uid, alg:MinAvgOurs):
        tx = 0
        for u_from in alg.env.users:
            ut = alg.env.users[uid]
            uf = u_from

            site_a = uf.service_a.node_id
            site_r = ut.service_r.node_id
            tx += alg.env.tx_user_node[uf.user_id][site_a] * conf["data_size"][0]
            tx += alg.env.tx_node_node[site_a][site_r] * conf["data_size"][1]
            tx += alg.env.tx_node_user[site_r][ut.user_id] * conf["data_size"][2]
        return tx / alg.env.user_num

    print("========= user {} ============".format(uid))

    etas = [0.1, 0.25, 0.5, 0.75]
    for i, algo in enumerate(algos):
        print(" ---------- eta = {} ----------".format(etas[i]))

        sa = algo.env.users[uid].service_a
        sr = algo.env.users[uid].service_r
        node_a_id = algo.env.users[uid].service_a.node_id
        node_r_id = algo.env.users[uid].service_r.node_id
        site_a = algo.env.sites[node_a_id]
        site_r = algo.env.sites[node_r_id]

        tier_a = 2 if algo.env.sites[node_a_id].area_id == 0 else 1
        tier_r = 2 if algo.env.sites[node_r_id].area_id == 0 else 1
        print("location: node({}, {}) tier({}, {})".format(node_a_id, node_r_id, tier_a, tier_r))

        print("n_serv = ({}, {})".format(sa.num_server, sr.num_server))
        print("price = ({}, {})".format(sa.price, sr.price))

        # Tx_a = algo.env.tx_user_node[uid][node_a_id] * 2
        # Tx_r = algo.env.tx_user_node[uid][node_r_id] * 3
        Tx_a = get_total_tx_from(uid, algo)
        Tx_r = get_total_tx_to(uid, algo)
        print("Tx = ({}, {})".format(Tx_a, Tx_r))

        print("Tp = ({}, {})".format(round(site_a.tp_a * 1000, 3), round(site_r.tp_r * 1000, 3)))

        print("Tq = ({}, {})".format(round(sa.queuing_delay * 1000, 3), round(sr.queuing_delay * 1000, 3)))

        H = sa.num_server * sa.price + sr.num_server * sr.price
        print("H = {}".format(H))

        print("eta * H = {}".format(etas[i] * H))

        T = Tx_a + Tx_r + site_a.tp_a * 1000 + site_r.tp_r * 1000 + sa.queuing_delay * 1000 + sr.queuing_delay * 1000
        print("T = {}".format(T))

        total = etas[i] * H + T
        print("f = {}".format(total))

def print_user_info_r(algos, uid):
    def get_total_tx_to(alg:MinAvgOurs, uid):
        tx = 0
        for u_from in alg.env.users:
            ut = alg.env.users[uid]
            uf = u_from

            site_a = uf.service_a.node_id
            site_r = ut.service_r.node_id
            tx += alg.env.tx_user_node[uf.user_id][site_a] * conf["data_size"][0]
            tx += alg.env.tx_node_node[site_a][site_r] * conf["data_size"][1]
            tx += alg.env.tx_node_user[site_r][ut.user_id] * conf["data_size"][2]
        return tx / alg.env.user_num

    print("========= user {} ============".format(uid))

    etas = [0.1, 0.25, 0.5, 0.75]
    for i, algo in enumerate(algos):
        print(" ---------- eta = {} ----------".format(etas[i]))

        sr = algo.env.users[uid].service_r
        node_r_id = algo.env.users[uid].service_r.node_id
        site_r = algo.env.sites[node_r_id]

        tier_r = 2 if algo.env.sites[node_r_id].area_id == 0 else 1
        print("location: node({}) tier({})".format(node_r_id, tier_r))

        print("n_serv = ({})".format(sr.num_server))
        print("price = ({})".format(sr.price))

        Tx_r = get_total_tx_to(algo, uid)
        print("Tx = ({})".format(Tx_r))

        print("Tp = ({})".format(round(site_r.tp_r * 1000, 3)))

        print("Tq = ({})".format(round(sr.queuing_delay * 1000, 3)))

        H = sr.num_server * sr.price
        print("H = {}".format(H))

        print("eta * H = {}".format(etas[i] * H))

        T = Tx_r + site_r.tp_r * 1000 + sr.queuing_delay * 1000
        print("T = {}".format(T))

        total = etas[i] * H + T
        print("f = {}".format(total))

def count_same_tier_offloading(algos):
    count_ = 0

    def check(tiers):
        if tiers[0] == tiers[1] and tiers[1] == tiers[2] and tiers[2] == tiers[3]:
            return True
        else:
            return False

    for i in range(num_user):
        offload_tier = []
        for algo in algos:  # type: MinAvgOurs
            node_a_id = algo.env.users[i].service_a.node_id
            node_r_id = algo.env.users[i].service_r.node_id
            tier_a = 2 if algo.env.sites[node_a_id].area_id == 0 else 1
            tier_r = 2 if algo.env.sites[node_r_id].area_id == 0 else 1
            offload_tier.append((tier_a, tier_r))

        # print("[user {}] tiers:{} nodes:{}".format(i, offload_tier, offload_node))
        if check(offload_tier):
            count_ += 1

    print("same tier offloading: {}".format(count_))

sim_times = 1
for sim_id in range(sim_times):
    print("========================= iteration {} ============================".format(sim_id + 1))
    # u_seed = random.randint(0, 10000000000)
    u_seed = 8257592327

    print("user_seed = {}".format(u_seed))

    print("------------- Ours (eta = 0.1) ------------------------")
    conf["eta"] = 0.1
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg_1 = MinAvgOurs(env)
    min_avg_alg_1.run()

    # sa = min_avg_alg_1.env.users[4].service_a
    # sr = min_avg_alg_1.env.users[4].service_r
    # target_site = min_avg_alg_1.env.sites[0]
    # sa.assign_to_site(target_site)
    # sr.assign_to_site(target_site)
    # sa.update_num_server(1)
    # sr.update_num_server(7)

    min_avg_alg_1.result_info()
    print(min_avg_alg_1.get_results())

    print("------------- Ours (eta = 0.25) ------------------------")
    conf["eta"] = 0.25
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg_2 = MinAvgOurs(env)
    min_avg_alg_2.run()

    # sa = min_avg_alg_2.env.users[4].service_a
    # sr = min_avg_alg_2.env.users[4].service_r
    # target_site = min_avg_alg_2.env.sites[34]
    # sa.assign_to_site(target_site)
    # sr.assign_to_site(target_site)
    # sa.update_num_server(1)
    # sr.update_num_server(13)

    min_avg_alg_2.result_info()
    print(min_avg_alg_2.get_results())

    print("------------- Ours (eta = 0.5) ------------------------")
    conf["eta"] = 0.5
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg_3 = MinAvgOurs(env)
    min_avg_alg_3.run()

    # sr = min_avg_alg_3.env.users[4].service_r
    # sr.update_num_server(sr.num_server - 1)
    # min_avg_alg_3.get_target_value()

    min_avg_alg_3.result_info()
    print(min_avg_alg_3.get_results())

    print("------------- Ours (eta = 0.75) ------------------------")
    conf["eta"] = 0.75
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg_4 = MinAvgOurs(env)
    min_avg_alg_4.run()

    # sa = min_avg_alg_4.env.users[42].service_a
    # sr = min_avg_alg_4.env.users[42].service_r
    # target_site = min_avg_alg_4.env.sites[35]
    # sa.assign_to_site(target_site)
    # sr.assign_to_site(target_site)
    # sa.update_num_server(1)
    # sr.update_num_server(11)

    min_avg_alg_4.result_info()
    print(min_avg_alg_4.get_results())

    # ------ debug ---------
    print("\n\n")
    print("seed: {}".format(u_seed))
    distribution = get_offloading_distribution([min_avg_alg_1, min_avg_alg_2, min_avg_alg_3, min_avg_alg_4])
    print(distribution)
    find_offloading_difference([min_avg_alg_1, min_avg_alg_2, min_avg_alg_3, min_avg_alg_4])

    print("\n--------- user info --------")
    # print_user_info([min_avg_alg_1, min_avg_alg_2, min_avg_alg_3, min_avg_alg_4], uid=4)
    print_user_info_a([min_avg_alg_1, min_avg_alg_2, min_avg_alg_3, min_avg_alg_4], uid=4)
    # print_user_info_r([min_avg_alg_1, min_avg_alg_2, min_avg_alg_3, min_avg_alg_4], uid=4)

    count_same_tier_offloading([min_avg_alg_1, min_avg_alg_2, min_avg_alg_3, min_avg_alg_4])

""" records 

5734711073 100
8913473348 100

8257592327 70
9630719332 70

e = 0.1
若采用 e = 0.25 时的卸载方案，
eta * H = 13.5 (↓ 2.3)
T = 74.9 (↑ 7.8)
f = 88.4 (↑ 5.5)

e = 0.25
若采用 e = 0.1 时的卸载方案，
eta * H = 39.5 (↑ 5.75)
T = 67.1 (↓ 7.8)
f = 106.6

---------------------------------------
e = 0.1
若采用 e = 0.25 时的卸载方案，
eta * H = 15.5 (↓ 4.7)
T = 72.3 (↑ 6.8)
f = 88.4 (↑ 2.7)

e = 0.25
若采用 e = 0.1 时的卸载方案，
eta * H = 50.5 (↑ 11.8)
T = 65.5 (↓ 6.8)
f = 112 (↑ 1)

e = 0.5
若采用 e = 0.25 时的卸载方案，
eta * H = 77.5 (↑ 10)
T = 72.3 (↓ 5.8)
f = 149.8 (↑ 4.2)
------------------------------
若采用 e = 0.75 时的卸载方案，
eta * H = 59.5 (↓ 8)
T = 82.3 (↑ 4.2)
f = 149.8 (↓ 3.8)

e = 0.75
若采用 e = 0.5 时的卸载方案，
eta * H = 101.25 (↑ 12)
T = 78.1 (↓ 4.2)
f = 149.8 (↑ 7.8)

"""
