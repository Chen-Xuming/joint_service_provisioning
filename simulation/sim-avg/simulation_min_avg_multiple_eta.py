"""
    对于每个用户数，不同eta使用同一种用户
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
from min_avg.min_avg_ours import MinAvgOurs

from configuration.config import config as conf

print("Script started at {}.".format(datetime.now()))

""" 创建文件夹 """
description = "multi_eta"        # fixme
res_dir = "../../result/min_avg/1-24_{}".format(description)
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

algorithms = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]

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

"""
    format:
    {
        user_num_1: {
            sim_id_1: {
                "simulation_id": xxx,
                ""user_seed": xxx,
                eta_1: {
                    algo_1: {
                        result values
                    },
                    algo_2: {
                        ...
                    },...
                },...
            }
        },
        user_num_2: {
            ...
        }, ...
    }
"""
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

                if alg_name == "Modify-Assignment(Tx)":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    nearest_alg = NearestAlgorithm(env)
                    nearest_alg.run()
                    ma1_alg = ModifyAssignmentAlgorithm(env)
                    ma1_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, ma1_alg)

                if alg_name == "Modify-Assignment(Tx+Tp)":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    nearest_alg = NearestAlgorithm(env)
                    nearest_alg.run()
                    ma2_alg = ModifyAssignmentAlgorithm_V2(env, t_compositions=0b110)
                    ma2_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, ma2_alg)

                if alg_name == "Modify-Assignment(Tx+Tp+Tq)":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    nearest_alg = NearestAlgorithm(env)
                    nearest_alg.run()
                    ma3_alg = ModifyAssignmentAlgorithm_V2(env, t_compositions=0b111)
                    ma3_alg.run()
                    save_result_to_dict(num_user, sim_id_str, eta_, alg_name, ma3_alg)

                if alg_name == "Ours":
                    env = Environment(conf, env_seed)
                    env.reset(num_user=num_user, user_seed=user_seed)
                    our_alg = MinAvgOurs(env)
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
