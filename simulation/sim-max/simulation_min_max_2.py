import copy

from env.environment_old import Environment
from numpy.random import default_rng
import random
from min_max.nearest import NearestAlgorithm
from min_max.stp_max_first import StpMaxFirst
from min_max.min_max_ours_v2 import MinMaxOurs_V2
from min_max.MGreedy import MGreedyAlgorithm
from configuration.config import config as conf

eta_range = (0.25, 1.0)
eta_step = 0.25

""" 创建文件夹 """
description = "minmax-v2"        # fixme
res_dir = "../../result/min_max/11-05_eta{}-{}_{}".format(eta_range[0], eta_range[1], description)
import os
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

env_seed = 99497

simulation_no = 4  # 文件号

# 用户数及测试次数
user_range = (40, 100)
user_range_step = 10
simulation_times_each_num_user = 10

# algorithms = ["Nearest", "Modify-Assignment", "M-Greedy", "Shortest-Path", "Shortest-Path-V2"]
# algorithms = ["Shortest-Path", "Shortest-Path-Stable-Only"]

algorithms = ["Nearest", "M-Greedy", "SP-Max-First", "Ours"]

do_RA = True
stable_only = False


""" 保存实验配置 """
import json
config_str = json.dumps(conf, indent=4)
with open(res_dir + "/config.txt", 'w') as conf_file:
    conf_file.write(config_str)
    conf_file.write("\n")
    conf_file.write("eta=[{}, {}], step={}\n".format(eta_range[0], eta_range[1], eta_step))
    conf_file.write("user_range: {}\n".format(user_range))
    conf_file.write("user_range_step: {}\n".format(user_range_step))
    conf_file.write("algorithms: \n")
    for alg in algorithms:
        conf_file.write("\t{}\n".format(alg))


results = {}

eta = eta_range[0]
while eta <= eta_range[1]:
    result_each_eta = {
        "eta": eta,
    }
    conf["eta"] = eta

    for num_user in range(user_range[0], user_range[1] + user_range_step, user_range_step):
        result_each_eta[num_user] = {}
        result_each_alg = {
            "max_delay": [],
            "avg_cost": [],
            "target_value": [],
            "running_time": [],
            "local_count": [],
            "common_count": []
        }
        for alg in algorithms:
            result_each_eta[num_user][alg] = copy.deepcopy(result_each_alg)

        for i in range(simulation_times_each_num_user):
            user_seed = random.randint(0, 1000000000)
            for j, alg_name in enumerate(algorithms):
                if alg_name == "Nearest":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    nearest_alg = NearestAlgorithm(env, do_RA=True, stable_only=False)
                    nearest_alg.run()
                    result_each_eta[num_user][alg_name]["max_delay"].append(nearest_alg.max_delay)
                    result_each_eta[num_user][alg_name]["avg_cost"].append(nearest_alg.final_avg_cost)
                    result_each_eta[num_user][alg_name]["target_value"].append(nearest_alg.target_value)
                    result_each_eta[num_user][alg_name]["running_time"].append(nearest_alg.running_time)
                    result_each_eta[num_user][alg_name]["local_count"].append(nearest_alg.local_count)
                    result_each_eta[num_user][alg_name]["common_count"].append(nearest_alg.common_count)

                if alg_name == "M-Greedy":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    mg_alg = MGreedyAlgorithm(env, do_RA=True, stable_only=False)
                    mg_alg.run()
                    result_each_eta[num_user][alg_name]["max_delay"].append(mg_alg.max_delay)
                    result_each_eta[num_user][alg_name]["avg_cost"].append(mg_alg.final_avg_cost)
                    result_each_eta[num_user][alg_name]["target_value"].append(mg_alg.target_value)
                    result_each_eta[num_user][alg_name]["running_time"].append(mg_alg.running_time)
                    result_each_eta[num_user][alg_name]["local_count"].append(mg_alg.local_count)
                    result_each_eta[num_user][alg_name]["common_count"].append(mg_alg.common_count)

                if alg_name == "SP-Max-First":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    spmf_alg = StpMaxFirst(env, do_RA=True, stable_only=False)
                    spmf_alg.run()
                    result_each_eta[num_user][alg_name]["max_delay"].append(spmf_alg.max_delay)
                    result_each_eta[num_user][alg_name]["avg_cost"].append(spmf_alg.final_avg_cost)
                    result_each_eta[num_user][alg_name]["target_value"].append(spmf_alg.target_value)
                    result_each_eta[num_user][alg_name]["running_time"].append(spmf_alg.running_time)
                    result_each_eta[num_user][alg_name]["local_count"].append(spmf_alg.local_count)
                    result_each_eta[num_user][alg_name]["common_count"].append(spmf_alg.common_count)

                if alg_name == "Ours":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    our_alg = MinMaxOurs_V2(env, do_RA=True, stable_only=False)
                    if num_user <= 60:
                        our_alg.alpha = 5e-5
                    elif 60 < num_user <= 80:
                        our_alg.alpha = 3e-5
                    elif num_user > 80:
                        our_alg.alpha = 1e-5
                    our_alg.epsilon = 15
                    our_alg.run()
                    result_each_eta[num_user][alg_name]["max_delay"].append(our_alg.max_delay)
                    result_each_eta[num_user][alg_name]["avg_cost"].append(our_alg.final_avg_cost)
                    result_each_eta[num_user][alg_name]["target_value"].append(our_alg.target_value)
                    result_each_eta[num_user][alg_name]["running_time"].append(our_alg.running_time)
                    result_each_eta[num_user][alg_name]["local_count"].append(our_alg.local_count)
                    result_each_eta[num_user][alg_name]["common_count"].append(our_alg.common_count)



            print("---------------------")
            print("eta = {}, num_user = {}, simulation #{}".format(eta, num_user, i))
            for j in range(len(algorithms)):
                print("algorithm: {}, max_delay = {}, cost = {}, target = {}, running_time = {}, local_count = {}, common_count = {}".format(algorithms[j],
                                                                                                        result_each_eta[num_user][algorithms[j]]["max_delay"][-1],
                                                                                                        result_each_eta[num_user][algorithms[j]]["avg_cost"][-1],
                                                                                                        result_each_eta[num_user][algorithms[j]]["target_value"][-1],
                                                                                                        result_each_eta[num_user][algorithms[j]]["running_time"][-1],
                                                                                                        result_each_eta[num_user][algorithms[j]]["local_count"][-1],
                                                                                                        result_each_eta[num_user][algorithms[j]]["common_count"][-1]))
            print("----------------------")

    results[eta] = result_each_eta
    eta += eta_step

print(results)

import simplejson
file = "{}/sim_{}.json".format(res_dir, simulation_no)
with open(file, "a") as fjson:
    simplejson.dump(results, fjson)