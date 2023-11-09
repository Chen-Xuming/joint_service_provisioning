from env.environment_old import Environment
from numpy.random import default_rng
import random
from min_avg.nearest import NearestAlgorithm

# from min_avg.modify_assignment import ModifyAssignmentAlgorithm
from min_avg.modify_assignment_v2 import ModifyAssignmentAlgorithm

from min_avg.min_avg_ours import MinAvgOurs

from configuration.config import config as conf

print("==================== env  config ===============================")
print(conf)
print("================================================================")

# seed = random.randint(0, 100000)
env_seed = 58972
print("env_seed: ", env_seed)

num_user = 40

for i in range(1):
    print("========================= iteration {} ============================".format(i + 1))
    # u_seed = random.randint(0, 10000000000)
    u_seed = i
    print("user_seed = {}".format(u_seed))

    print("------------- nearest ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    nearest_alg = NearestAlgorithm(env, stable_only=False)
    nearest_alg.run()
    print(nearest_alg.get_results())

    print("------------- modify-assignment ------------------------")
    modify_assignment_alg = ModifyAssignmentAlgorithm(env)
    modify_assignment_alg.run()
    print(modify_assignment_alg.get_results())

    print("------------- Our ------------------------")
    env = Environment(conf, env_seed)
    env.reset(num_user=num_user, user_seed=u_seed)
    our_alg = MinAvgOurs(env)
    our_alg.run()
    print(our_alg.get_results())

