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
    u_seed = random.randint(0, 10000000000)
    # u_seed = 92238814

    print("user_seed = {}".format(u_seed))

    print("------------- Nearest ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    nearest_alg = NearestAlgorithm(env)
    nearest_alg.run()
    print(nearest_alg.get_results())
    #
    # print("------------- Modify-Assignment(original) ------------------------")
    # env = Environment(conf, env_seed)
    # env.reset(num_user=num_user, user_seed=u_seed)
    # nearest_alg = NearestAlgorithm(env, do_RA=True, stable_only=False)
    # nearest_alg.run()
    # ma_alg = ModifyAssignmentAlgorithm(env)
    # ma_alg.run()
    # ma_alg.result_info()
    # print(ma_alg.get_results())
    #
    # print("------------- Modify-Assignment(Tx+Tp+Tq) ------------------------")
    # env = Environment(conf, env_seed)
    # env.reset(num_user=num_user, user_seed=u_seed)
    # nearest_alg = NearestAlgorithm(env, do_RA=True, stable_only=False)
    # nearest_alg.run()
    # ma3_alg = ModifyAssignmentAlgorithm_V2(env, t_compositions=0b111)
    # ma3_alg.debug_flag = False
    # ma3_alg.run()
    # ma3_alg.result_info()
    # print(ma3_alg.get_results())

    print("------------- Ours ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    min_avg_alg = MinAvgOurs(env)
    min_avg_alg.run()
    min_avg_alg.result_info()
    print(min_avg_alg.get_results())

