"""
    仿真单种eta值下的实验。
    与 ”simulation_min_avg.py“ 的不同：
    1. 记录结果的 json 格式不同
    2. 记录每组实验的 user_seed，便于复现
"""

from env.environment2 import Environment
from numpy.random import default_rng
import random
import os
import json
from datetime import datetime

from min_avg.nearest import NearestAlgorithm
from min_avg.modify_assignment import ModifyAssignmentAlgorithm
from min_avg.modify_assignment_v2 import ModifyAssignmentAlgorithm as ModifyAssignmentAlgorithm_V2
from min_avg.greedy_server_provisioning import GreedyServerProvisioningAlgorithm
from min_avg.min_avg_ours import MinAvgOurs
from min_avg.min_avg_centralized import MinAvgCentralized

from configuration.config import config as conf

print("Script started at {}.".format(datetime.now()))

""" 创建文件夹 """
description = "small_eta"        # fixme
res_dir = "../../result/min_avg/1-19_eta{}_{}".format(conf["eta"], description)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

print("res_dir = {}".format(res_dir))

env_seed = 99497

simulation_no = 15  # 文件号
print("simulation_no = {}".format(simulation_no))

# 用户数及测试次数
user_range = (40, 100)
user_range_step = 10
simulation_times_each_num_user = 8

# algorithms = ["Nearest", "Modify-Assignment", "M-Greedy", "Shortest-Path", "Shortest-Path-V2"]

# algorithms = ["Nearest", "M-Greedy(4)", "M-Greedy(8)", "M-Greedy(No Limitation)", "Min-Avg", "Max-First", "Ours"]
# algorithms = ["Nearest", "M-Greedy", "M-Greedy-V2", "Min-Avg", "Max-First", "Ours"]
algorithms = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]

# algorithms = ["Ours", "Ours_centralized"]

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

            if alg_name == "Modify-Assignment(Tx)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env)
                nearest_alg.run()
                ma1_alg = ModifyAssignmentAlgorithm(env)
                ma1_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, ma1_alg)

            if alg_name == "Modify-Assignment(Tx+Tp)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env)
                nearest_alg.run()
                ma2_alg = ModifyAssignmentAlgorithm_V2(env, t_compositions=0b110)
                ma2_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, ma2_alg)

            if alg_name == "Modify-Assignment(Tx+Tp+Tq)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env)
                nearest_alg.run()
                ma3_alg = ModifyAssignmentAlgorithm_V2(env, t_compositions=0b111)
                ma3_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, ma3_alg)

            if alg_name == "GSP":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                gsp_alg = GreedyServerProvisioningAlgorithm(env, avg_t_compositions=0b111)
                gsp_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, gsp_alg)

            if alg_name == "Ours":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                our_alg = MinAvgOurs(env)
                our_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, our_alg)

            if alg_name == "Ours_centralized":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                our_centralized_alg = MinAvgCentralized(env)
                our_centralized_alg.run()
                save_result_to_dict(num_user, sim_id_str, alg_name, our_centralized_alg)

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
