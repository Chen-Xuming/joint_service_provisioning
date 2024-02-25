from analyse_min_avg_single_eta import process_data as process_data_for_single_data

import simplejson as json
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import os

fontsize = 20
linewidth = 3
markersize = 12
fontsize_legend = 20
# color_list = ['#2878b5',  '#F28522', '#58B272', '#FF1F5B', '#991a4e', '#1f77b4', '#A6761D', '#009ADE', '#AF58BA']
# color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b']
color_list = ['#58B272', '#009ade', '#ff1f5b']

# color_list = ['#002c53', '#9c403d', '#8983BF', '#58B272', '#f28522', '#009ade', '#ff1f5b']

# color_list = ['#ff1f5b', '#009ade', '#f28522', '#58B272', '#B22222', '#4B65AF']


# marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']
marker_list = ['d', 'X', 'o', '^', 's', 'v', 'P',  '*','>','<','x']

# figure_size = (12, 9)
# dpi = 60

# MEng
figure_size = (12, 6)
dpi = 60

shared_legend = True

x_label = "Number of Users"
y_label_f = "Reduction Ratio of " + r"$T_{avg}+\eta H$" + " (%)"
y_label_delay = "Reduction Ratio of\nAverage Interaction Delay (%)"
y_label_cost = "Reduction Ratio of Average Cost (%)"

# 黑白图
black_and_white_style = False
if black_and_white_style:
    color_list = ["#0d0d0d" for c in color_list]
    markersize = 13

# 中文
in_chinese = True
if in_chinese:
    from matplotlib import rcParams
    rcParams['font.family'] = 'SimSun'

    x_label = "用户数"
    y_label_f = r"$T_{avg}+\eta H$" + " 减小率（%）"
    y_label_delay = "平均交互时延减小率（%）"
    y_label_cost = "平均开销减小率（%）"

# ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42,
                     "mathtext.fontset" : "cm"})

algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "Ours"]
# algorithm_list = ["Nearest", "Modify-Assignment(Tx)", "Modify-Assignment(Tx+Tp+Tq)", "GSP", "Ours"]
# etas = [0, 0.25, 0.5, 0.75, 1.0]    # fixme
#
# etas = [0.5, 0.75, 1.0]    # fixme
# etas = [0.25, 0.5, 0.75]

etas = [0, 0.25, 0.5, 0.75]

user_range = (40, 100)
user_step = 10

# def process_data(file_base_name):
#     res = {}
#     for eta_ in etas:
#         file = file_base_name.format(eta_)
#         res[eta_] = process_data_for_single_data(file)
#     return res

def process_data(files):
    res = {}
    for eta_, file in files.items():
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
    plt.figure(figsize=figure_size, dpi=dpi)

    y_label = ""
    if attribute == "target_value":
        y_label = y_label_f
    if attribute == "avg_delay":
        y_label = y_label_delay
    if attribute == "cost":
        y_label = y_label_cost

    ylabel_font = {
        'usetex': False,
        'size': fontsize + 10
    }

    # if attribute == "target_value":
    #     ylabel_font = {
    #         'usetex': False,
    #         'size': fontsize + 12,
    #
    #     }

    plt.ylabel(y_label, ylabel_font, labelpad=10)
    plt.xlabel(x_label, fontsize=fontsize+10, labelpad=10)
    plt.grid(linestyle='--')

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize+8)
    plt.yticks(fontsize=fontsize + 8)

    if attribute == "target_value":
        y = [i for i in range(4, 16+2, 2)]
        plt.yticks(y)
    if attribute == "cost":
        y = [i for i in range(12, 26+1, 1)]
        plt.yticks(y)

    idx = 0
    for eta, ratios in reduction_ratios.items():
        plt.plot(x, ratios, label=r'$\eta={}$'.format(eta), color=color_list[idx], marker=marker_list[idx])
        idx += 1

    if shared_legend and attribute in ["cost", "avg_delay"]:
        plt.tight_layout()
        plt.show()
        return
    else:
        if shared_legend and attribute == "target_value":
            leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 6},
                             bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, mode="expand", ncol=3)
            frame = leg.get_frame()
            frame.set_linewidth(1.0)
            frame.set_edgecolor('black')
        else:
            if in_chinese:
                leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 5}, loc='best')
            else:
                leg = plt.legend(fontsize=fontsize_legend + 6, loc='best')
        leg.set_draggable(state=True)
        plt.tight_layout()
        plt.show()

def get_and_draw_all_reduction_ratios(res_dict, origin_algo="Modify-Assignment(Tx+Tp+Tq)"):
    target_value_rr = get_reduction_ratio(raw_result, original_algorithm=origin_algo, attribute="target_value")
    avg_delay_rr = get_reduction_ratio(raw_result, original_algorithm=origin_algo, attribute="avg_delay")
    avg_cost_rr = get_reduction_ratio(raw_result, original_algorithm=origin_algo, attribute="cost")

    fig = plt.figure()
    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]

    # ------------------------- target value ---------------------------
    plt.subplot(1, 3, 1)
    plt.ylabel("Reduction Ratio of Weighted Sum of \nAverage Latency and Cost (%)")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(a)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    idx = 0
    for eta, ratios in target_value_rr.items():
        if eta == 0:
            continue
        plt.plot(x, ratios, label=r'$\eta={}$'.format(eta), color=color_list[idx], marker=marker_list[idx])
        idx += 1

    # -------------------------- avg delay --------------------------
    plt.subplot(1, 3, 2)
    plt.ylabel("Reduction Ratio of\nAverage Interaction Latency (%)")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(b)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    idx = 0
    for eta, ratios in avg_delay_rr.items():
        if eta == 0:
            continue
        plt.plot(x, ratios, label=r'$\eta={}$'.format(eta), color=color_list[idx], marker=marker_list[idx])
        idx += 1

    # ------------------------- average cost ----------------------------
    plt.subplot(1, 3, 3)
    plt.ylabel("Reduction Ratio of Average Cost (%)")
    plt.xlabel("Number of Users")
    plt.text(0.48, -0.25, "(c)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    idx = 0
    for eta, ratios in avg_cost_rr.items():
        if eta == 0:
            continue
        plt.plot(x, ratios, label=r'$\eta={}$'.format(eta), color=color_list[idx], marker=marker_list[idx])
        idx += 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    leg = fig.legend(lines, labels, bbox_to_anchor=(0.74, 0.96), ncol=4, framealpha=1)
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

# # 检查Nearest算法和Ma算法是否将服务全部都放置在Tier-1节点
# def check_nearest_and_Ma(res_dict):
#     pass

if __name__ == '__main__':
    file_list = dict()
    for e_ in etas:
        file_list[e_] = "min_avg/12-26_eta{}_new_conf".format(e_)

    # for e_ in [0.05, 0.10, 0.15]:
    #     file_list[e_] = "min_avg/1-19_eta{}_small_eta".format(e_)
    # for e_ in [0, 0.25, 0.5, 0.75]:
    #     file_list[e_] = "min_avg/12-26_eta{}_new_conf".format(e_)

    raw_result = process_data(file_list)

    # file_name = "min_avg/12-26_eta{}_new_conf"
    # raw_result = process_data(file_name)
    # print(raw_result)

    target_value_reduction_ratios = get_reduction_ratio(raw_result, "target_value")
    draw_reduction_ratio(target_value_reduction_ratios, "target_value")

    avg_delay_reduction_ratios = get_reduction_ratio(raw_result, "avg_delay")
    draw_reduction_ratio(avg_delay_reduction_ratios, "avg_delay")

    cost_reduction_ratios = get_reduction_ratio(raw_result, "cost")
    draw_reduction_ratio(cost_reduction_ratios, "cost")

    offloading_distribution = get_offloading_ratio(raw_result)
    print(offloading_distribution)

    # get_and_draw_all_reduction_ratios(raw_result)
