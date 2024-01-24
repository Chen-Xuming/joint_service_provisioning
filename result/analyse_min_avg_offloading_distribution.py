import random

import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]

# 获取一组实验的json文件的路径
def get_json_file_list(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix == ".json":
            json_file = os.path.join(dir_path, file_name)
            files.append(json_file)
    return files

user_range = (40, 100)
user_step = 30
etas = [0, 0.1, 0.25, 0.5, 0.75]
num_algorithms = len(algorithm_list)

def process_data(dir_path):
    json_file_list = get_json_file_list(dir_path)

    """
        target format:
        {
            user_num_1: {
                eta_1: {
                    algo_1: {
                        "local_count": [...],
                        "common_count": [...]
                    },
                    ...
                },
                ...
            },
            ...
        }
    """
    target = {}
    for num_u in range(user_range[0], user_range[1] + user_step, user_step):
        target[num_u] = {}
        for eta_ in etas:
            target[num_u][eta_] = {}
            for algo in algorithm_list:
                target[num_u][eta_][algo] = {
                    "local_count": [],
                    "common_count": []
                }

    for file in json_file_list:
        print("processing file: {}".format(file))
        raw_data = json.load(open(file))

        for u_str, u_data in raw_data.items():
            user_num = int(u_str)
            for sim_id_str, sim_data in u_data.items():
                for eta_ in etas:
                    for algo in algorithm_list:
                        target[user_num][eta_][algo]["local_count"].append(sim_data[eta_][algo]["local_count"])
                        target[user_num][eta_][algo]["common_count"].append(sim_data[eta_][algo]["local_count"])

    for u, u_data in target.items():
        for e, e_data in u_data.items():
            for algo, algo_data in e_data.items():
                total_count = np.sum(target[u][e][algo]["local_count"]) + np.sum(target[u][e][algo]["common_count"])
                target[u][e][algo]["local_ratio"] = np.sum(target[u][e][algo]["local_count"]) / total_count
                target[u][e][algo]["common_ratio"] = np.sum(target[u][e][algo]["common_ratio"]) / total_count

                if algo in ["Nearest", "Modify-Assignment(Tx)"]:
                    for element in target[u][e][algo]["common_count"]:
                        assert element == 0, "Algorithm {} offloads services to tier-2 nodes.".format(algo)

    return target

if __name__ == "__main__":
    directory = "min_avg/1-24_multi_eta"
    res = process_data(directory)
    print(res)