from env.environment import Environment
from numpy.random import default_rng
import random

from min_max.nearest import NearestAlgorithm
from min_max.stp_max_first import StpMaxFirst
from min_max.min_max_ours_v2 import MinMaxOurs_V2
from min_max.MGreedy import MGreedyAlgorithm
from min_max.min_avg_for_min_max import MinAvgForMinMax
from min_max.surrogate import MinMaxSurrogate
from configuration.config import config as conf

""" 创建文件夹 """
description = "min-max-surrogate"        # fixme
res_dir = "../../result/min_max/11-08_eta{}_{}".format(conf["eta"], description)
import os
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

print("target direction: {}".format(res_dir))

env_seed = 99497

simulation_no = 5  # 文件号
print("simulation num: {}".format(simulation_no))

# 用户数及测试次数
user_range = (40, 100)
user_range_step = 10
simulation_times_each_num_user = 3

# algorithms = ["Nearest", "Modify-Assignment", "M-Greedy", "Shortest-Path", "Shortest-Path-V2"]

# algorithms = ["Nearest", "M-Greedy", "SP-Max-First", "Ours"]
# algorithms = ["Nearest", "M-Greedy(4)", "M-Greedy(8)", "M-Greedy(No Limitation)", "Min-Avg", "SP-Max-First", "Ours"]

algorithms = ["Nearest", "Surrogate", "SP-Max-First"]

do_RA = True
stable_only = False


""" 保存实验配置 """
import json
config_str = json.dumps(conf, indent=4)
with open(res_dir + "/config.txt", 'w') as conf_file:
    conf_file.write(config_str)
    conf_file.write("\n")
    conf_file.write("user_range: {}\n".format(user_range))
    conf_file.write("user_range_step: {}\n".format(user_range_step))
    conf_file.write("algorithms: \n")
    for alg in algorithms:
        conf_file.write("\t{}\n".format(alg))


results = {
    "max_delay": {},
    "cost": {},
    "target_value": {},
    "running_time": {},
    "local_count": {},
    "common_count": {}
}

for num_user in range(user_range[0], user_range[1] + user_range_step, user_range_step):
    res_max_delay = [[] for _ in range(len(algorithms))]
    res_cost = [[] for _ in range(len(algorithms))]
    res_target_value = [[] for _ in range(len(algorithms))]
    res_running_time = [[] for _ in range(len(algorithms))]
    res_local_count = [[] for _ in range(len(algorithms))]
    res_common_count = [[] for _ in range(len(algorithms))]

    for i in range(simulation_times_each_num_user):

        user_seed = random.randint(0, 100000000)

        for j, alg_name in enumerate(algorithms):
            if alg_name == "Nearest":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env, do_RA=True, stable_only=False)
                nearest_alg.run()
                res_max_delay[j].append(nearest_alg.max_delay)
                res_cost[j].append(nearest_alg.final_avg_cost)
                res_target_value[j].append(nearest_alg.target_value)
                res_running_time[j].append(nearest_alg.running_time)
                res_local_count[j].append(nearest_alg.local_count)
                res_common_count[j].append(nearest_alg.common_count)

            if alg_name == "M-Greedy(4)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                mg_alg = MGreedyAlgorithm(env, do_RA=True, stable_only=False)
                mg_alg.M = 4
                mg_alg.run()
                res_max_delay[j].append(mg_alg.max_delay)
                res_cost[j].append(mg_alg.final_avg_cost)
                res_target_value[j].append(mg_alg.target_value)
                res_running_time[j].append(mg_alg.running_time)
                res_local_count[j].append(mg_alg.local_count)
                res_common_count[j].append(mg_alg.common_count)

            if alg_name == "M-Greedy(8)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                mg_alg = MGreedyAlgorithm(env, do_RA=True, stable_only=False)
                mg_alg.M = 8
                mg_alg.run()
                res_max_delay[j].append(mg_alg.max_delay)
                res_cost[j].append(mg_alg.final_avg_cost)
                res_target_value[j].append(mg_alg.target_value)
                res_running_time[j].append(mg_alg.running_time)
                res_local_count[j].append(mg_alg.local_count)
                res_common_count[j].append(mg_alg.common_count)

            if alg_name == "M-Greedy(No Limitation)":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                mg_alg = MGreedyAlgorithm(env, do_RA=True, stable_only=False)
                mg_alg.run()
                res_max_delay[j].append(mg_alg.max_delay)
                res_cost[j].append(mg_alg.final_avg_cost)
                res_target_value[j].append(mg_alg.target_value)
                res_running_time[j].append(mg_alg.running_time)
                res_local_count[j].append(mg_alg.local_count)
                res_common_count[j].append(mg_alg.common_count)

            if alg_name == "Min-Avg":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                min_avg_alg = MinAvgForMinMax(env, do_RA=True, stable_only=False)
                min_avg_alg.run()
                res_max_delay[j].append(min_avg_alg.max_delay)
                res_cost[j].append(min_avg_alg.final_avg_cost)
                res_target_value[j].append(min_avg_alg.target_value)
                res_running_time[j].append(min_avg_alg.running_time)
                res_local_count[j].append(min_avg_alg.local_count)
                res_common_count[j].append(min_avg_alg.common_count)

            if alg_name == "SP-Max-First":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                spmf_alg = StpMaxFirst(env, do_RA=True, stable_only=False)
                spmf_alg.run()
                res_max_delay[j].append(spmf_alg.max_delay)
                res_cost[j].append(spmf_alg.final_avg_cost)
                res_target_value[j].append(spmf_alg.target_value)
                res_running_time[j].append(spmf_alg.running_time)
                res_local_count[j].append(spmf_alg.local_count)
                res_common_count[j].append(spmf_alg.common_count)

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
                res_max_delay[j].append(our_alg.max_delay)
                res_cost[j].append(our_alg.final_avg_cost)
                res_target_value[j].append(our_alg.target_value)
                res_running_time[j].append(our_alg.running_time)
                res_local_count[j].append(our_alg.local_count)
                res_common_count[j].append(our_alg.common_count)

            if alg_name == "Surrogate":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                surrogate_alg = MinMaxSurrogate(env, do_RA=True, stable_only=False)

                surrogate_alg.run()
                res_max_delay[j].append(surrogate_alg.max_delay)
                res_cost[j].append(surrogate_alg.final_avg_cost)
                res_target_value[j].append(surrogate_alg.target_value)
                res_running_time[j].append(surrogate_alg.running_time)
                res_local_count[j].append(surrogate_alg.local_count)
                res_common_count[j].append(surrogate_alg.common_count)

        print("---------------------")
        print("num_user = {}, simulation #{}".format(num_user, i))
        for j in range(len(algorithms)):
            print("algorithm: {}, max_delay = {}, cost = {}, target = {}, running_time = {}, local_count = {}, common_count = {}".format(algorithms[j],
                                                                                                                                         res_max_delay[j][i],
                                                                                                                                         res_cost[j][i],
                                                                                                                                         res_target_value[j][i],
                                                                                                                                         res_running_time[j][i],
                                                                                                                                         res_local_count[j][i],
                                                                                                                                         res_common_count[j][i]))
        print("----------------------")

    results["max_delay"][num_user] = res_max_delay
    results["cost"][num_user] = res_cost
    results["target_value"][num_user] = res_target_value
    results["running_time"][num_user] = res_running_time
    results["local_count"][num_user] = res_local_count
    results["common_count"][num_user] = res_common_count

print(results)

import simplejson
file = "{}/user{}-{}_{}.json".format(res_dir, user_range[0], user_range[1], simulation_no)
with open(file, "a") as fjson:
    simplejson.dump(results, fjson)