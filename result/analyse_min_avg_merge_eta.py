from analyse_min_avg_single_eta import process_data as process_data_for_single_data

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
color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b']
# color_list = ['#002c53', '#9c403d', '#8983BF', '#58B272', '#f28522', '#009ade', '#ff1f5b']

marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]

etas = [0, 0.25, 0.5, 0.75, 1.0]    # fixme

user_range = (40, 100)
user_step = 10

def process_data(file_base_name):
    res = {}
    for eta_ in etas:
        file = file_base_name.format(eta_)
        res[eta_] = process_data_for_single_data(file)
    return res

"""
    {
        eta1: [...],
        eta2: [...],
        ...
    }
"""
def get_reduction_ratio(res_dict, attribute: str, original_algorithm="Modify-Assignment(Tx+Tp+Tq)", target_algorithm="Ours"):
    reduction_ratios = {}
    for eta, dict_ in res_dict.items():
        if eta == 0:
            continue
        reduction_ratios[eta] = []
        list_len = len(res_dict[eta][target_algorithm][attribute])
        for i in range(list_len):
            origin = res_dict[eta][original_algorithm][attribute][i]
            current = res_dict[eta][target_algorithm][attribute][i]
            reduction_ratio = (origin - current) / origin
            reduction_ratio *= 100
            reduction_ratio = round(reduction_ratio, 1)     # 转化为%，保留一位小数
            reduction_ratios[eta].append(reduction_ratio)

    return reduction_ratios

def draw_reduction_ratio(reduction_ratios, attribute: str):
    plt.figure()
    plt.ylabel("Reduction Ratio (%)", fontsize=fontsize+3)
    plt.xlabel("Number of Users", fontsize=fontsize+3)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)

    idx = 0
    for eta, ratios in reduction_ratios.items():
        plt.plot(x, ratios, label=r'$\eta={}$'.format(eta), color=color_list[idx], marker=marker_list[idx])
        idx += 1

    leg = plt.legend(fontsize=fontsize_legend)
    leg.set_draggable(state=True)

    plt.show()

def get_offloading_ratio(res_dict):
    res = {}
    for eta, eta_dict in res_dict.items():
        res[eta] = {}
        for alg, alg_dict in eta_dict.items():
            local_count = sum(alg_dict["local_count"])
            common_count = sum(alg_dict["common_count"])
            total = local_count + common_count
            res[eta][alg] = [local_count, common_count, total, round(local_count/total*100, 1), round(common_count/total*100, 1)]
    return res

if __name__ == '__main__':
    file_name = "min_avg/11-14_eta{}_min-avg-ma-3kinds"
    raw_result = process_data(file_name)
    # print(raw_result)

    target_value_reduction_ratios = get_reduction_ratio(raw_result, "target_value")
    # target_value_reduction_ratios = get_reduction_ratio(raw_result, "target_value", original_algorithm="Nearest")
    draw_reduction_ratio(target_value_reduction_ratios, "target_value")

    offloading_distribution = get_offloading_ratio(raw_result)
    print(offloading_distribution)