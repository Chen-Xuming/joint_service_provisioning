from min_max.min_avg_for_min_max import MinAvgForMinMax
from min_max.mgreedy import MGreedyAlgorithm
from min_max.mgreedy_v2 import MGreedyAlgorithm_V2
from min_max.min_max_ours_v2 import MinMaxOurs_V2
from min_max.stp_max_first import StpMaxFirst
from min_max.nearest import NearestAlgorithm
from env.environment2 import Environment
from configuration.config import config as conf
from configuration.config import alpha_initial_values
from matplotlib import pyplot as plt
import random

env_seed = 99497
print("env_seed: ", env_seed)

num_user = 60

sim_times = 10
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

    # print("------------- Max-First ------------------------")
    # env = Environment(conf, env_seed)
    # env.reset(num_user=num_user, user_seed=u_seed)
    # max_first_alg = StpMaxFirst(env)
    # max_first_alg.run()
    # print(max_first_alg.get_results())
    #
    # print("------------- Min-First ------------------------")
    # env = Environment(conf, env_seed)
    # env.reset(num_user=num_user, user_seed=u_seed)
    # min_first_alg = MinAvgForMinMax(env)
    # min_first_alg.run()
    # print(min_first_alg.get_results())

    print("------------- M-Greedy ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    m_greedy_alg = MGreedyAlgorithm(env)
    m_greedy_alg.run()
    print(m_greedy_alg.get_results())

    print("------------- M-Greedy-V2(110) ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    m_greedy_alg_v2 = MGreedyAlgorithm_V2(env, max_t_compositions=0b110)
    m_greedy_alg_v2.run()
    print(m_greedy_alg_v2.get_results())

    print("------------- M-Greedy-V2(111) ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    m_greedy_alg_v2 = MGreedyAlgorithm_V2(env, max_t_compositions=0b111)
    m_greedy_alg_v2.run()
    print(m_greedy_alg_v2.get_results())

    print("------------- Ours ------------------------")
    # env = Environment(conf, env_seed)
    # env.reset(num_user=num_user, user_seed=u_seed)
    # our_alg = MinMaxOurs_V2(env)
    # our_alg.debug_flag = True
    # our_alg.alpha = alpha_initial_values[conf["eta"]][num_user]
    # our_alg.epsilon = 15
    # our_alg.run()
    # print(our_alg.get_results())
