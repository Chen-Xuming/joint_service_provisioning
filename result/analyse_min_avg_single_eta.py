import random

import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

fontsize = 20
linewidth = 3
markersize = 10
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 20
# color_list = ['#2878b5',  '#F28522', '#58B272', '#FF1F5B', '#991a4e', '#1f77b4', '#A6761D', '#009ADE', '#AF58BA']
# color_list = ['#002c53', '#ffa510', '#0c84c6', '#ffbd66', '#f74d4d', '#2455a4', '#41b7ac']
color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b']
# color_list = ['#002c53', '#9c403d', '#8983BF', '#58B272', '#f28522', '#009ade', '#ff1f5b']

# color_list = ['#f28522', '#58B272', '#9c403d', '#009ade', '#ff1f5b']


marker_list = ['d', '^', 'X', 'o', 's', 'v', 'P',  '*','>','<','x']

figure_size = (10, 10)
dpi = 60

# algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]
# algorithm_in_fig = ["Nearest", "Modify-Assignment", "Modify-Assignment-V2", "Min-Avg"]

# algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "GSP", "Ours"]
# algorithm_in_fig = ["Nearest", "M-Greedy", "M-Greedy-V2", "GSP", "Min-Max"]

algorithm_list = ["Ours", "Ours_centralized"]
algorithm_in_fig = ["Min-Avg(Independent)", "Min-Avg(Centralized)"]


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
user_step = 10
num_algorithms = len(algorithm_list)


def process_data(dir_path):
    json_file_list = get_json_file_list(dir_path)

    """
        data = {
            40: {
                "Nearest": {
                    "avg_delay": 100,
                    "cost:: 101,
                    ...
                },
                ...
            },
            ...
        }
    """
    data = {}
    for u in range(user_range[0], user_range[1] + user_step, user_step):
        data[u] = {}
        for algo in algorithm_list:
            data[u][algo] = {
                "avg_delay": [],
                "cost": [],
                "target_value": [],
                "running_time": [],
                "local_count": [],
                "common_count": []
            }

    """ 逐个处理json文件，读取数据到data字典里 """
    for file in json_file_list:
        print(file)
        raw_data = json.load(open(file))

        for u_str, u_data in raw_data.items():
            user_num = int(u_str)
            for sim_id_str, sim_data in u_data.items():
                for algo in algorithm_list:
                    algo_data = sim_data[algo]

                    for attr, attr_val in algo_data.items():
                        if attr in data[user_num][algo].keys():
                            data[user_num][algo][attr].append(attr_val)

    """ 对 data 中的每个数组求平均值 """
    for u in range(user_range[0], user_range[1] + user_step, user_step):
        for algo in algorithm_list:
            for attr, attr_array in data[u][algo].items():
                data[u][algo][attr] = np.mean(data[u][algo][attr])

            # 单位转换
            data[u][algo]["avg_delay"] = data[u][algo]["avg_delay"] * 1000              # ms
            data[u][algo]["running_time"] = data[u][algo]["running_time"] / 1000        # s

    """
        final_data = {
            "Nearest": {
                "avg_delay": [...],     # 不同用户数下的平均最大时延
                ..
            },
            ...
        }
    """
    final_data = {}
    for algo in algorithm_list:
        final_data[algo] = {
            "avg_delay": [],
            "cost": [],
            "target_value": [],
            "running_time": [],
            "local_count": [],
            "common_count": []
        }
    for u, u_data in data.items():
        for algo, algo_data in u_data.items():
            if algo in algorithm_list:
                for attr, attr_value in algo_data.items():
                    final_data[algo][attr].append(attr_value)

    return final_data

def draw_avg_delay(data: dict):
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.ylabel("Average Interaction Latency (ms)", fontsize=fontsize+10, labelpad=10)
    plt.xlabel("Number of Users", fontsize=fontsize+10, labelpad=10)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize+8)
    plt.yticks(fontsize=fontsize+8)

    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["avg_delay"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    leg = plt.legend(fontsize=fontsize_legend+2, loc='best')
    leg.set_draggable(state=True)
    plt.show()

def draw_avg_cost(data: dict):
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.ylabel("Average Cost", fontsize=fontsize+10, labelpad=10)
    plt.xlabel("Number of Users", fontsize=fontsize+10, labelpad=10)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize+8)
    plt.yticks(fontsize=fontsize+8)

    # y = [i for i in range(75, 350+25, 25)]
    # plt.yticks(ticks=y)

    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["cost"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    leg = plt.legend(fontsize=fontsize_legend+2, loc='best')
    leg.set_draggable(state=True)
    plt.show()

def draw_target_value(data: dict):
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.ylabel("Weighted Sum of Average\nLatency and Cost", fontsize=fontsize+10, labelpad=10)
    plt.xlabel("Number of Users", fontsize=fontsize+10, labelpad=10)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize + 8)
    plt.yticks(fontsize=fontsize + 8)

    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["target_value"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    leg = plt.legend(fontsize=fontsize_legend+2, loc='best')
    leg.set_draggable(state=True)
    plt.show()

def draw_running_time(data: dict):
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.ylabel("Average Running Time (s)")
    plt.xlabel("Number of Users")
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)

    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["running_time"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    leg = plt.legend(fontsize=fontsize_legend, loc='best')
    leg.set_draggable(state=True)
    plt.show()

def draw_figures_shared_legend(data: dict):
    fig = plt.figure()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]

    # ------------------------- target value ---------------------------
    plt.subplot(1, 3, 1)
    plt.ylabel("Weighted Sum of Average\nLatency and Cost")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(a)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["target_value"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    # -------------------------- avg delay --------------------------
    plt.subplot(1, 3, 2)
    plt.ylabel("Average Interaction Latency (ms)")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(b)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["avg_delay"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    # ------------------------- average cost ----------------------------
    plt.subplot(1, 3, 3)
    plt.ylabel("Average Cost")
    plt.xlabel("Number of Users")
    plt.text(0.48, -0.25, "(c)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    for idx, algo in enumerate(algorithm_list):
        plt.plot(x, data[algo]["cost"], label=algorithm_in_fig[idx], color=color_list[idx], marker=marker_list[idx])

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    leg = fig.legend(lines, labels, bbox_to_anchor=(0.74, 0.96), ncol=4, framealpha=1)
    leg.set_draggable(state=True)
    plt.show()

if __name__ == '__main__':
    eta = 0.5
    if eta == 0:
        algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]
        algorithm_in_fig = ["Nearest", "Modify-Assignment", "Modify-Assignment-V2", "Min-Avg"]

    raw_data_path = "min_avg/12-8_eta{}_min-avg-centralized".format(eta)
    res_dict = process_data(raw_data_path)
    # print(res_dict)

    # draw_avg_delay(res_dict)
    # draw_avg_cost(res_dict)
    # draw_target_value(res_dict)
    # draw_running_time(res_dict)
    # draw_figures_shared_legend(res_dict)

    if eta == 0:
        draw_avg_delay(res_dict)
    else:
        # draw_figures_shared_legend(res_dict)

        # draw_avg_delay(res_dict)
        # draw_avg_cost(res_dict)
        draw_target_value(res_dict)