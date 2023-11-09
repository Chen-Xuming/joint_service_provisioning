"""
    仿真单种eta值下的实验。
    与 ”simulation_min_max.py“ 的不同：
    1. 记录结果的 json 格式不同
    2. 记录每组实验的 user_seed，便于复现
"""

from env.environment2 import Environment
from numpy.random import default_rng
import random
import os
import json
from datetime import datetime

from min_max.nearest import NearestAlgorithm
from min_max.stp_max_first import StpMaxFirst
from min_max.min_max_ours_v2 import MinMaxOurs_V2
from min_max.MGreedy import MGreedyAlgorithm
from min_max.surrogate import MinMaxSurrogate

from configuration.config import config as conf
from configuration.config import alpha_initial_values as alpha_list

print("Script started at {}.".format(datetime.now()))

""" 创建文件夹 """
description = "min-max-new_site_attr"        # fixme
res_dir = "../../result/min_max/11-09_eta{}_{}".format(conf["eta"], description)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

print("res_dir = {}".format(res_dir))

env_seed = 99497

simulation_no = 1  # 文件号
print("simulation_no = {}".format(simulation_no))

# 用户数及测试次数
user_range = (40, 50)
user_range_step = 10
simulation_times_each_num_user = 2

# algorithms = ["Nearest", "Modify-Assignment", "M-Greedy", "Shortest-Path", "Shortest-Path-V2"]

algorithms = ["Nearest", "M-Greedy(4)", "M-Greedy(8)", "M-Greedy(No Limitation)", "Max-First", "Ours"]
# algorithms = ["Nearest", "M-Greedy(No Limitation)", "Max-First"]

do_RA = True
stable_only = False

""" 保存实验配置 """
if not os.path.exists(res_dir + "/config.txt"):
    config_str = json.dumps(conf, indent=4)
    with open(res_dir + "/config.txt", 'w') as conf_file:
        conf_file.write("Simulation Description: {}\n".format(description))
        conf_file.write(config_str)
        conf_file.write("\n")
        conf_file.write("user_range: {}\n".format(user_range))
        conf_file.write("user_range_step: {}\n".format(user_range_step))
        conf_file.write("algorithms: \n")
        for alg in algorithms:
            conf_file.write("\t{}\n".format(alg))

""" 创建json文件 """
file = "{}/user{}-{}_{}.json".format(res_dir, user_range[0], user_range[1], simulation_no)
if os.path.exists(file):
    raise Exception("File {} is already existed!".format(file))

result = {}

def save_result_to_dict(n, sid, algo_name, algo):
    algo_res = algo.get_results()
    result[n][sid][algo_name] = {}
    for attr, value in algo_res.items():
        result[num_user][sim_id_str][alg_name][attr] = value

for num_user in range(user_range[0], user_range[1] + user_range_step, user_range_step):
    result[num_user] = {}

    for sim_id in range(simulation_times_each_num_user):
        sim_id_str = "sim{}".format(sim_id)
        result[num_user][sim_id_str] = dict()

        user_seed = random.randint(0, 100000000)
        result[num_user][sim_id_str]["simulation_id"] = sim_id
        result[num_user][sim_id_str]["user_seed"] = user_seed

        for alg_name in algorithms:
            if alg_name == "Nearest":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env)
                nearest_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, nearest_alg)

            elif alg_name == "M-Greedy(4)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                mg4_alg = MGreedyAlgorithm(env)
                mg4_alg.M = 4
                mg4_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, mg4_alg)

            elif alg_name == "M-Greedy(8)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                mg8_alg = MGreedyAlgorithm(env)
                mg8_alg.M = 8
                mg8_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, mg8_alg)

            elif alg_name == "M-Greedy(No Limitation)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                mg_alg = MGreedyAlgorithm(env)
                mg_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, mg_alg)

            elif alg_name == "Max-First":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                max_first_alg = StpMaxFirst(env)
                max_first_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, max_first_alg)

            elif alg_name == "Surrogate":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                surrogate_alg = MinMaxSurrogate(env)
                surrogate_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, surrogate_alg)

            elif alg_name == "Ours":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                our_alg = MinMaxOurs_V2(env)
                our_alg.alpha = alpha_list[conf["eta"]][num_user]
                our_alg.epsilon = 15
                our_alg.run()
                our_alg.debug_flag = False
                save_result_to_dict(num_user, sim_id_str, alg_name, our_alg)

        print("-----------------------------------")
        print("num_user: {}, simulation #{}".format(num_user, sim_id))
        for k, v in result[num_user][sim_id_str].items():
            if k in algorithms:
                print("{}: {}".format(k, v))
        print("")

print(result)
import simplejson
with open(file, 'w') as f:
    simplejson.dump(result, f)

print("Script finished at {}.".format(datetime.now()))


























