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

sim_times = 1
for sim_id in range(sim_times):
    print("========================= iteration {} ============================".format(sim_id + 1))
    # u_seed = random.randint(0, 10000000000)
    u_seed = 7175262417

    print("user_seed = {}".format(u_seed))

    print("------------- Ours (eta = 0.1) ------------------------")
    conf["eta"] = 0.1
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg = MinAvgOurs(env)
    min_avg_alg.run()
    min_avg_alg.result_info()
    print(min_avg_alg.get_results())

    print("------------- Ours (eta = 0.25) ------------------------")
    conf["eta"] = 0.25
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg = MinAvgOurs(env)
    min_avg_alg.run()
    min_avg_alg.result_info()
    print(min_avg_alg.get_results())

    print("------------- Ours (eta = 0.5) ------------------------")
    conf["eta"] = 0.5
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg = MinAvgOurs(env)
    min_avg_alg.run()
    min_avg_alg.result_info()
    print(min_avg_alg.get_results())

    print("------------- Ours (eta = 0.75) ------------------------")
    conf["eta"] = 0.75
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg = MinAvgOurs(env)
    min_avg_alg.run()
    min_avg_alg.result_info()
    print(min_avg_alg.get_results())

