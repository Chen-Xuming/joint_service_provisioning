from env.environment_old import Environment
from numpy.random import default_rng
import random
from min_avg.nearest import NearestAlgorithm
from min_avg.modify_assignment import ModifyAssignmentAlgorithm
from min_avg.modify_assignment_v2 import ModifyAssignmentAlgorithm as ModifyAssignmentAlgorithm_V2
from min_avg.min_avg_ours import MinAvgOurs
from configuration.config import config as conf

""" 创建文件夹 """
description = "ma-v2"        # fixme
res_dir = "../../result/min_avg/10-24_eta{}_{}".format(conf["eta"], description)
import os
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

print("res_dir = {}".format(res_dir))

env_seed = 99497

simulation_no = 4  # 文件号

# 用户数及测试次数
user_range = (40, 100)
user_range_step = 10
simulation_times_each_num_user = 10

# algorithms = ["Nearest", "Modify-Assignment", "M-Greedy", "Shortest-Path", "Shortest-Path-V2"]
# algorithms = ["Shortest-Path", "Shortest-Path-Stable-Only"]

algorithms = ["Nearest", "Modify-Assignment", "Modify-Assignment-V2", "RA-UA"]

do_RA = False
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
    "avg_delay": {},
    "cost": {},
    "target_value": {},
    "running_time": {},
    "local_count": {},
    "common_count": {}
}

for num_user in range(user_range[0], user_range[1] + user_range_step, user_range_step):
    res_avg_delay = [[] for _ in range(len(algorithms))]
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
                res_avg_delay[j].append(nearest_alg.avg_delay)
                res_cost[j].append(nearest_alg.final_avg_cost)
                res_target_value[j].append(nearest_alg.target_value)
                res_running_time[j].append(nearest_alg.running_time)
                res_local_count[j].append(nearest_alg.local_count)
                res_common_count[j].append(nearest_alg.common_count)


            if alg_name == "Modify-Assignment":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env, do_RA=True, stable_only=False)
                nearest_alg.run()
                mdf_alg = ModifyAssignmentAlgorithm(env, do_RA=True, stable_only=False)
                mdf_alg.run()
                res_avg_delay[j].append(mdf_alg.avg_delay)
                res_cost[j].append(mdf_alg.final_avg_cost)
                res_target_value[j].append(mdf_alg.target_value)
                res_running_time[j].append(mdf_alg.running_time)
                res_local_count[j].append(mdf_alg.local_count)
                res_common_count[j].append(mdf_alg.common_count)

            if alg_name == "Modify-Assignment-V2":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                nearest_alg = NearestAlgorithm(env, do_RA=True, stable_only=False)
                nearest_alg.run()
                mdf_alg_v2 = ModifyAssignmentAlgorithm_V2(env, do_RA=True, stable_only=False)
                mdf_alg_v2.run()
                res_avg_delay[j].append(mdf_alg_v2.avg_delay)
                res_cost[j].append(mdf_alg_v2.final_avg_cost)
                res_target_value[j].append(mdf_alg_v2.target_value)
                res_running_time[j].append(mdf_alg_v2.running_time)
                res_local_count[j].append(mdf_alg_v2.local_count)
                res_common_count[j].append(mdf_alg_v2.common_count)


            if alg_name == "RA-UA":
                env = Environment(conf, env_seed)
                env.reset(num_user=num_user, user_seed=user_seed)
                our_alg = MinAvgOurs(env, do_RA=True, stable_only=False)
                our_alg.run()
                res_avg_delay[j].append(our_alg.avg_delay)
                res_cost[j].append(our_alg.final_avg_cost)
                res_target_value[j].append(our_alg.target_value)
                res_running_time[j].append(our_alg.running_time)
                res_local_count[j].append(our_alg.local_count)
                res_common_count[j].append(our_alg.common_count)

        print("---------------------")
        print("num_user = {}, simulation #{}".format(num_user, i))
        for j in range(len(algorithms)):
            print("algorithm: {}, avg_delay = {}, cost = {}, target = {}, running_time = {}, local_count = {}, common_count = {}".format(algorithms[j],
                                                                                                                                         res_avg_delay[j][i],
                                                                                                                                         res_cost[j][i],
                                                                                                                                         res_target_value[j][i],
                                                                                                                                         res_running_time[j][i],
                                                                                                                                         res_local_count[j][i],
                                                                                                                                         res_common_count[j][i]))
        print("----------------------")

    results["avg_delay"][num_user] = res_avg_delay
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