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

        # print(raw_data)

        for u_str, u_data in raw_data.items():
            user_num = int(u_str)
            for sim_id_str, sim_data in u_data.items():
                for eta_ in etas:
                    for algo in algorithm_list:
                        target[user_num][eta_][algo]["local_count"].append(sim_data[str(eta_)][algo]["local_count"])
                        target[user_num][eta_][algo]["common_count"].append(sim_data[str(eta_)][algo]["common_count"])

                        if algo in ["Nearest", "Modify-Assignment(Tx)"]:
                            assert sim_data[str(eta_)][algo]["common_count"] == 0, "Algorithm {} offloads services to tier-2 nodes.".format(algo)

    for u, u_data in target.items():
        for e, e_data in u_data.items():
            for algo, algo_data in e_data.items():
                total_count = np.sum(target[u][e][algo]["local_count"]) + np.sum(target[u][e][algo]["common_count"])
                target[u][e][algo]["local_count"] = np.round(np.sum(target[u][e][algo]["local_count"]) / total_count * 100, decimals=1)    # 转化为百分比
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

    y_40 = [24.2, 27.1, 32.7, 33.4, 31.8]
    y_70 = [26.3, 29.1, 38.9, 35.8, 32.4]
    y_100 = [26.2, 29.9, 39.4, 35.2, 32.0]

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
    directory = "min_avg/1-24_multi_eta"
    res = process_data(directory)
    print(res)

    draw_offloading_distribution(res)

"""  ------------- Results -----------------------

user_num = 40
                0       0.1      0.25       0.5     0.75       
Min-Avg-SP      24.2    27.1     32.7       33.4    31.8    
MA-V2-RA        31.4    28.1     32.8       37.8    38.1

user_num = 70
                0       0.1      0.25       0.5     0.75       
Min-Avg-SP      26.3    29.1     38.9       35.8    32.4    
MA-V2-RA        32.4    29.4     34.5       39.2    39.2

user_num = 70
                0       0.1      0.25       0.5     0.75       
Min-Avg-SP      26.2    29.9     39.4       35.2    32.0    
MA-V2-RA        33.4    30.1     34.9       39.6    39.7

"""