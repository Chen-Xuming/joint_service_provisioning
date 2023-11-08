import random

import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

fontsize = 20
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 20
# color_list = ['#2878b5',  '#F28522', '#58B272', '#FF1F5B', '#991a4e', '#1f77b4', '#A6761D', '#009ADE', '#AF58BA']
# color_list = ['#002c53', '#ffa510', '#0c84c6', '#ffbd66', '#f74d4d', '#2455a4', '#41b7ac']
# color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b']
color_list = ['#002c53', '#9c403d', '#8983BF', '#58B272', '#f28522', '#009ade', '#ff1f5b']

marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

# algorithm_list = ["Nearest", "M-Greedy", "SP-Max-First", "Ours"]
algorithm_list = ["Nearest", "M-Greedy(4)", "M-Greedy(8)", "M-Greedy(No Limitation)", "Min-Avg", "SP-Max-First", "Ours"]    # simulation name
algorithm_in_fig = ["Nearest", "M-Greedy(4)", "M-Greedy(8)", "M-Greedy(No Limitation)", "Min-Avg", "Max-First", "Ours"]


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

    max_delays = {}
    costs = {}
    target_values = {}
    running_times = {}
    for u in range(user_range[0], user_range[1] + user_step, user_step):
        max_delays[u] = [[] for _ in range(num_algorithms)]
        costs[u] = [[] for _ in range(num_algorithms)]
        target_values[u] = [[] for _ in range(num_algorithms)]
        running_times[u] = [[] for _ in range(num_algorithms)]

    u_range = [x for x in range(user_range[0], user_range[1] + user_step, user_step)]

    for file in json_file_list:
        print(file)
        raw_data = json.load(open(file))

        raw_max_delay = raw_data['max_delay']
        for u_num, max_delay_arr in raw_max_delay.items():
            if int(u_num) not in u_range:
                continue

            for i in range(num_algorithms):
                max_delays[int(u_num)][i].extend(max_delay_arr[i])

        raw_running_time = raw_data['running_time']
        for u_num, running_time_arr in raw_running_time.items():
            if int(u_num) not in u_range:
                continue

            for i in range(num_algorithms):
                running_times[int(u_num)][i].extend(running_time_arr[i])

        raw_cost = raw_data['cost']
        for u_num, cost_arr in raw_cost.items():
            if int(u_num) not in u_range:
                continue

            for i in range(num_algorithms):
                costs[int(u_num)][i].extend(cost_arr[i])

        raw_target_value = raw_data['target_value']
        for u_num, target_value_arr in raw_target_value.items():
            if int(u_num) not in u_range:
                continue

            for i in range(num_algorithms):
                target_values[int(u_num)][i].extend(target_value_arr[i])

    print(max_delays)
    print(costs)
    print(target_values)
    print(running_times)

    avg_max_delay = {}
    avg_cost = {}
    avg_target_value = {}
    avg_running_times = {}
    for u in range(user_range[0], user_range[1] + user_step, user_step):
        avg_max_delay[u] = []
        avg_cost[u] = []
        avg_target_value[u] = []
        avg_running_times[u] = []

    for u_num, arr in max_delays.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            avg_md = np.mean(arr[i])
            avg_max_delay[u_num].append(avg_md * 1000)      # ms

    for u_num, arr in running_times.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            avg_r = np.mean(arr[i]) / 1000      # 秒
            avg_running_times[u_num].append(avg_r)

    for u_num, arr in costs.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            avg_c = np.mean(arr[i])
            avg_cost[u_num].append(avg_c)

    for u_num, arr in target_values.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            avg_t = np.mean(arr[i])
            avg_target_value[u_num].append(avg_t)

    print("---------------------")
    print("avg_max_delay: ", avg_max_delay)
    print("avg_cost: ", avg_cost)
    print("avg_target_value: ", avg_target_value)
    print("avg_running_time: ", avg_running_times)
    print("---------------------")

    res_max_delay = [[] for _ in range(num_algorithms)]
    for u_num, md in avg_max_delay.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            res_max_delay[i].append(md[i])

    res_running_time = [[] for _ in range(num_algorithms)]
    for u_num, r in avg_running_times.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            res_running_time[i].append(r[i])

    res_cost = [[] for _ in range(num_algorithms)]
    for u_num, md in avg_cost.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            res_cost[i].append(md[i])

    res_target_value = [[] for _ in range(num_algorithms)]
    for u_num, md in avg_target_value.items():
        if int(u_num) not in u_range:
            continue

        for i in range(num_algorithms):
            res_target_value[i].append(md[i])
    print("res_max_delay: ", res_max_delay)
    print("res_cost: ", res_cost)
    print("res_target_value: ", res_target_value)
    print("res_running_time: ", res_running_time)

    return res_max_delay, res_cost, res_target_value, res_running_time

def draw_max_delay(max_delays):
    # for i in range(num_algorithms):
    #     for j in range(len(max_delays[i])):
    #         max_delays[i][j] += random.uniform(-0.3, 0.3)

    plt.figure(figsize=(5, 5))
    # plt.figure()
    plt.ylabel("Maximum Interaction Latency (ms)", fontsize=fontsize+3)
    plt.xlabel("Number of Users", fontsize=fontsize+3)
    plt.grid(linestyle='--')
    plt.tight_layout()

    algs = algorithm_in_fig

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    # plt.yticks(ticks=[i for i in range(0, 400, 20)])
    #
    # min_y = np.min(cost)
    # max_y = np.max(cost)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 20))

    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, max_delays[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    # plt.legend(fontsize=fontsize_legend, bbox_to_anchor=(1, 0.2))   # eta = 0
    leg = plt.legend(fontsize=fontsize_legend, loc='best')
    leg.set_draggable(state=True)

    plt.show()

def draw_cost(costs):
    plt.figure()
    plt.ylabel("Average Cost")
    plt.xlabel("Number of Users")
    plt.grid(linestyle='--')
    plt.tight_layout()

    algs = algorithm_in_fig

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    # plt.yticks(ticks=[i for i in range(0, 180, 10)])

    # min_y = np.min(cost)
    # max_y = np.max(cost)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 50))

    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, costs[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    plt.legend(fontsize=fontsize_legend)
    # plt.savefig(save_file, bbox_inches="tight")
    plt.show()

def draw_target_value(target_values):
    plt.figure()
    plt.ylabel("Weighted Sum of Maximum\nLatency and Cost")
    plt.xlabel("Number of Users")
    plt.grid(linestyle='--')
    plt.tight_layout()

    algs = algorithm_in_fig

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    # plt.yticks(ticks=[i for i in range(0, 180, 10)])

    # min_y = np.min(cost)
    # max_y = np.max(cost)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 50))

    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, target_values[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    plt.legend(fontsize=fontsize_legend)
    # plt.savefig(save_file, bbox_inches="tight")
    plt.show()

def draw_running_time(times):
    plt.figure()
    plt.ylabel("Average Running Time (s)")
    plt.xlabel("Number of Users")
    plt.grid(linestyle='--')
    plt.tight_layout()

    # algs = ["Random", "Nearest", "Greedy", "RL"]
    # algs = ["Random", "Nearest", "Greedy", "RL(train40-50)", "RL(train50-60)", "SL(train40-70)", "SL(train60-70)", "A3C+GCN"]
    algs = algorithm_in_fig

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    # plt.yticks(ticks=[i for i in range(0, 15000, 1000)])

    # min_y = np.min(cost)
    # max_y = np.max(cost)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 50))

    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, times[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    plt.legend(fontsize=fontsize_legend)
    # plt.savefig(save_file, bbox_inches="tight")
    plt.show()

def draw_figures_shared_legend(max_delays, avg_costs, target_values):
    algs = algorithm_in_fig
    fig = plt.figure()

    # ------------------------- target value ---------------------------
    plt.subplot(1, 3, 1)
    # plt.ylabel("Weighted Sum of Average Latency and Cost")
    plt.ylabel("Weighted Sum of Maximum\nLatency and Cost")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(a)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, target_values[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    # -------------------------- max delay --------------------------
    plt.subplot(1, 3, 2)
    plt.ylabel("Maximum Interaction Latency (ms)")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(b)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, max_delays[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    # ------------------------- average cost ----------------------------
    plt.subplot(1, 3, 3)
    plt.ylabel("Average Cost")
    plt.xlabel("Number of Users")
    plt.text(0.48, -0.25, "(c)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, avg_costs[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    leg = fig.legend(lines, labels, bbox_to_anchor=(0.74, 0.96), ncol=4, framealpha=1)
    leg.set_draggable(state=True)

    plt.show()

def draw_figures(max_delays, avg_costs, target_values):
    algs = algorithm_in_fig

    # ------------------------- target value ---------------------------
    plt.subplot(1, 3, 1)
    # plt.ylabel("Weighted Sum of Average Latency and Cost")
    plt.ylabel("Weighted Sum of Maximum\nLatency and Cost")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(a)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, target_values[i], label=algs[i], color=color_list[i], marker=marker_list[i])
    leg1 = plt.legend(fontsize=fontsize_legend)
    leg1.set_draggable(state=True)

    # -------------------------- max delay --------------------------
    plt.subplot(1, 3, 2)
    plt.ylabel("Maximum Interaction Latency (ms)")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(b)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    # plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, max_delays[i], label=algs[i], color=color_list[i], marker=marker_list[i])
    # plt.legend(fontsize=fontsize_legend, bbox_to_anchor=(1, 0.5))   # eta = 0
    leg2 = plt.legend(fontsize=fontsize_legend)
    leg2.set_draggable(state=True)

    # ------------------------- average cost ----------------------------
    plt.subplot(1, 3, 3)
    plt.ylabel("Average Cost")
    plt.xlabel("Number of Users")
    plt.text(0.48, -0.25, "(c)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    # plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    for i in range(num_algorithms):
        if algs[i] == "None":
            continue
        plt.plot(x, avg_costs[i], label=algs[i], color=color_list[i], marker=marker_list[i])
    leg3 = plt.legend(fontsize=fontsize_legend)
    leg3.set_draggable(state=True)

    plt.show()


if __name__ == '__main__':
    raw_data_path = "min_max/11-07_eta0.5_min-max-7algs"

    max_delay, cost, target_value, running_time = process_data(raw_data_path)
    # draw_target_value(target_value)
    # draw_max_delay(max_delay)
    # draw_cost(cost)
    draw_running_time(running_time)

    # draw_figures(avg_delay, cost, target_value)
    draw_figures_shared_legend(max_delay, cost, target_value)