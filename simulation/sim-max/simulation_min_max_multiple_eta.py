from env.environment2 import Environment
from numpy.random import default_rng
import random
import os
import json
from datetime import datetime

from min_max.nearest import NearestAlgorithm
from min_max.min_max_ours_v3 import MinMaxOurs_V3
from min_max.mgreedy import MGreedyAlgorithm
from min_max.mgreedy_v2 import MGreedyAlgorithm_V2
from configuration.config import config as conf
from configuration.config import alpha_initial_values as alpha_list

print("Script started at {}.".format(datetime.now()))

""" 创建文件夹 """
description = "multi_eta"        # fixme
res_dir = "../../result/min_max/1-24_{}".format(description)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

print("res_dir = {}".format(res_dir))

env_seed = 99497

simulation_no = 39  # 文件号
print("simulation_no = {}".format(simulation_no))

# 用户数及测试次数
user_range = (40, 100)
user_range_step = 30
simulation_times_each_num_user = 4

# etas = [0, 0.1]
etas = [0, 0.1, 0.25, 0.5, 0.75]

algorithms = ["Nearest", "M-Greedy", "M-Greedy-V2(Tx+Tp+Tq)", "Ours"]


""" 保存实验配置 """
if not os.path.exists(res_dir + "/config.txt"):
    config_str = json.dumps(conf, indent=4)
    with open(res_dir + "/config.txt", 'w') as conf_file:
        conf_file.write("Simulation Description: {}\n".format(description))
        conf_file.write(config_str)
        conf_file.write("\n")
        conf_file.write("user_range: {}\n".format(user_range))
        conf_file.write("user_range_step: {}\n".format(user_range_step))
        conf_file.write("etas: {}\n".format(etas))
        conf_file.write("algorithms: \n")
        for alg in algorithms:
            conf_file.write("\t{}\n".format(alg))

""" 创建json文件 """
file = "{}/user{}-{}_{}.json".format(res_dir, user_range[0], user_range[1], simulation_no)
if os.path.exists(file):
    raise Exception("File {} is already existed!".format(file))

result = {}

def save_result_to_dict(num_u, sim_id, eta, algo_name, algo):
    algo_res = algo.get_results()
    result[num_u][sim_id][eta][algo_name] = {}
    for attr, value in algo_res.items():
        result[num_u][sim_id][eta][algo_name][attr] = value

for num_user in range(user_range[0], user_range[1] + user_range_step, user_range_step):
    result[num_user] = {}

    for sim_id in range(simulation_times_each_num_user):
        sim_id_str = "sim{}".format(sim_id)
        result[num_user][sim_id_str] = dict()

        user_seed = random.randint(0, 1000000000)
        result[num_user][sim_id_str]["simulation_id"] = sim_id
        result[num_user][sim_id_str]["user_seed"] = user_seed

        for eta_ in etas:
            conf["eta"] = eta_      # 这里要手动修改
            result[num_user][sim_id_str][eta_] = {}

            for alg_name in algorithms:
                if alg_name == "Nearest":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    nearest_alg = NearestAlgorithm(env)
                    nearest_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, nearest_alg)

                elif alg_name == "M-Greedy(No Limitation)" or alg_name == "M-Greedy":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    mg_alg = MGreedyAlgorithm(env)
                    mg_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, mg_alg)

                elif alg_name == "M-Greedy-V2(Tx+Tp)":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    mgv2_110_alg = MGreedyAlgorithm_V2(env, max_t_compositions=0b110)
                    mgv2_110_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, mgv2_110_alg)

                elif alg_name == "M-Greedy-V2(Tx+Tp+Tq)":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    mgv2_111_alg = MGreedyAlgorithm_V2(env, max_t_compositions=0b111)
                    mgv2_111_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, mgv2_111_alg)

                elif alg_name == "Ours":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    our_alg = MinMaxOurs_V3(env)
                    our_alg.debug_flag = False
                    if conf["eta"] in alpha_list.keys():
                        our_alg.alpha = alpha_list[conf["eta"]][num_user]
                    else:
                        our_alg.alpha = 1e-5
                    our_alg.epsilon = 15
                    our_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, our_alg)

        print("-----------------------------------")
        print("num_user: {}, simulation #{}".format(num_user, sim_id))
        for eta_ in etas:
            print("[eta = {}]".format(eta_))
            for k, v in result[num_user][sim_id_str][eta_].items():
                print("\t{}: {}".format(k, v))
        print("")

print(result)
import simplejson
with open(file, 'w') as f:
    simplejson.dump(result, f)

print("Script finished at {}.".format(datetime.now()))