import random

import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

algorithm_list = ["Nearest", "M-Greedy", "M-Greedy-V2(Tx+Tp+Tq)", "Ours"]

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
                        target[user_num][eta_][algo]["local_count"].append(sim_data[str(eta_)][algo]["local_count"])
                        target[user_num][eta_][algo]["common_count"].append(sim_data[str(eta_)][algo]["common_count"])

                        if algo in ["Nearest", "Modify-Assignment(Tx)"]:
                            assert sim_data[str(eta_)][algo]["common_count"] == 0, \
                                "Algorithm {} offloads services to tier-2 nodes.".format(algo)

    for u, u_data in target.items():
        for e, e_data in u_data.items():
            for algo, algo_data in e_data.items():
                total_count = np.sum(target[u][e][algo]["local_count"]) + np.sum(target[u][e][algo]["common_count"])
                target[u][e][algo]["local_count"] = np.round(np.sum(target[u][e][algo]["local_count"]) / total_count * 100, decimals=1)
                target[u][e][algo]["common_count"] = np.round(np.sum(target[u][e][algo]["common_count"]) / total_count * 100, decimals=1)

                # if algo in ["Nearest", "Modify-Assignment(Tx)"]:
                #     for element in target[u][e][algo]["common_count"]:
                #         assert element == 0, "Algorithm {} offloads services to tier-2 nodes.".format(algo)

    return target

def draw_offloading_distribution(res_dict: dict):
    from matplotlib import pyplot as plt

    plt.grid(linestyle='--')
    plt.tight_layout()

    plt.xlabel(r"$\eta$")
    plt.ylabel("Offloading Ratio (%)")

    y_40 = [43.6, 41.8, 48.1, 34.2, 31.4]
    y_70 = [53.7, 59.7, 46.2, 39.5, 32.9]
    y_100 = [55.1, 58.6, 46.6, 40.4, 32.6]

    x = [0, 0.1, 0.25, 0.5, 0.75]

    markers = ['d', '^', 'X']
    plt.xticks(x)
    plt.plot(x, y_40, label="N=40", marker=markers[0])
    plt.plot(x, y_70, label="N=70", marker=markers[1])
    plt.plot(x, y_100, label="N=100", marker=markers[2])

    leg = plt.legend(loc='best')
    leg.set_draggable(state=True)
    plt.show()

if __name__ == "__main__":
    directory = "min_max/1-24_multi_eta"
    res = process_data(directory)
    print(res)

    draw_offloading_distribution(res)

"""  ------------- Results -----------------------

user_num = 40
                    0       0.1      0.25       0.5     0.75      
Min-Max-SP          43.6    41.8     48.1       34.2    31.4
M-Greedy-RA         43.6    43.6     43.6       43.6    43.6
M-Greedy-V2-RA      59.1    72.6     75.2       79.8    80.4

user_num = 70
                    0       0.1      0.25       0.5     0.75   
Min-Max-SP          53.7    59.7     46.2       39.5    32.9    
M-Greedy-RA         45.9    45.9     45.9       45.9    45.9
M-Greedy-V2-RA      62.0    76.5     75.3       79.7    80.5

user_num = 100
                    0       0.1      0.25       0.5     0.75   
Min-Max-SP          55.1    58.6     46.6       40.4    32.6    
M-Greedy-RA         53.9    53.9     53.9       53.9    53.9
M-Greedy-V2-RA      71.3    76.1     76.1       80.0    80.4

"""