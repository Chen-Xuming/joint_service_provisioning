from analyse_min_max_single_eta import process_data as process_data_for_single_data
from analyse_min_max_single_eta import process_data_v2

import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

from collections import OrderedDict

fontsize = 20
linewidth = 3
markersize = 12
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 20
# color_list = ['#2878b5',  '#F28522', '#58B272', '#FF1F5B', '#991a4e', '#1f77b4', '#A6761D', '#009ADE', '#AF58BA']
# color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b']
# color_list = ['#002c53', '#9c403d', '#8983BF', '#58B272', '#f28522', '#009ade', '#ff1f5b']

# color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b', '#002c53', '#9c403d']
color_list = ['#ff1f5b', '#009ade', '#f28522', '#58B272', '#B22222', '#4B65AF']

marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

figure_size = (12, 9)
dpi = 60

# MEng
# figure_size = (13, 6)
# dpi = 60

shared_legend = False

x_label = "Number of Users"
y_label_f = r'$T_{max}+\eta H$'
y_label_delay = "Maximum Interaction Delay (ms)"
y_label_cost = "Average Cost"

# 黑白图
black_and_white_style = False
if black_and_white_style:
    color_list = ["#0d0d0d" for c in color_list]
    markersize = 13

# 中文
in_chinese = False
if in_chinese:
    from matplotlib import rcParams
    rcParams['font.family'] = 'SimSun'

    x_label = "用户数"
    y_label_f = r'$T_{max}+\eta H$'
    y_label_delay = "最大交互时延（毫秒）"
    y_label_cost = "平均开销"

plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42,
                     "mathtext.fontset" : "cm"})

# algorithm_list = ["Nearest", "M-Greedy", "M-Greedy-V2", "Min-Avg", "Max-First", "Ours"]
algorithm_list = ["Nearest", "M-Greedy", "M-Greedy-V2(Tx+Tp+Tq)", "Ours"]
algorithm_name_in_fig = ["Nearest-RA", "M-Greedy-RA", "M-Greedy-V2-RA", "Min-Max-RASP"]

# etas = [0.5, 0.75, 1.0]
etas = [0.25, 0.5, 0.75]

user_range = (40, 100)
user_step = 10

def process_data(file_base_name, need_error_bar=False):
    res = {}
    for eta_ in etas:
        file = file_base_name.format(eta_)
        if not need_error_bar:
            res[eta_] = process_data_for_single_data(file)
        else:
            res[eta_] = process_data_v2(file)
    return res

"""
    {
        eta1: [...],
        eta2: [...],
        ...
    }
"""
def get_reduction_ratio(res_dict, attribute: str, original_algorithm="M-Greedy-V2(Tx+Tp+Tq)", target_algorithm="Ours"):
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
        if eta == 0:
            continue
        plt.plot(x, ratios, label=r'$\eta={}$'.format(eta), color=color_list[idx], marker=marker_list[idx])
        idx += 1

    # leg = plt.legend(fontsize=fontsize_legend)
    leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 2}, loc='best')
    leg.set_draggable(state=True)

    plt.show()

def get_and_draw_all_reduction_ratios(res_dict):
    target_value_rr = get_reduction_ratio(raw_result, "target_value")
    max_delay_rr = get_reduction_ratio(raw_result, "max_delay")
    avg_cost_rr = get_reduction_ratio(raw_result, "cost")

    fig = plt.figure()
    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]

    # ------------------------- target value ---------------------------
    plt.subplot(1, 3, 1)
    plt.ylabel("Reduction Ratio of Weighted Sum of \nMaximum Latency and Cost (%)")
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

    # -------------------------- max delay --------------------------
    plt.subplot(1, 3, 2)
    plt.ylabel("Reduction Ratio of\nMaximum Interaction Latency (%)")
    plt.xlabel("Number of Users")
    plt.text(0.5, -0.25, "(b)", transform=plt.gca().transAxes, fontsize=20, va='center')
    plt.grid(linestyle='--')
    plt.xticks(ticks=x)
    idx = 0
    for eta, ratios in max_delay_rr.items():
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

def draw_merged_eta_for_some_attribution(res_dict, attribution: str, our_algo="Ours", compared_algo="M-Greedy-V2(Tx+Tp+Tq)"):
    fs = figure_size
    if shared_legend and attribution == "target_value":
        fs = (figure_size[0], figure_size[1] + 1.8)
    plt.figure(figsize=fs, dpi=dpi)

    y_label = ""
    if attribution == "target_value":
        y_label = y_label_f
    elif attribution == "max_delay":
        y_label = y_label_delay
    elif attribution == "cost":
        y_label = y_label_cost
    plt.ylabel(ylabel=y_label, fontsize=fontsize+10, labelpad=10)
    plt.xlabel(x_label, fontsize=fontsize+10, labelpad=10)
    plt.grid(linestyle='--')

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize + 8)
    plt.yticks(fontsize=fontsize + 8)

    if attribution == "target_value":
        y_ = [i for i in range(150, 325+25, 25)]
        plt.yticks(y_)
    if attribution == "cost":
        y_ = [i for i in range(70, 230, 10)]
        plt.yticks(y_)

    for idx, eta_ in enumerate(etas):
        eta_data = res_dict[eta_]

        plt.plot(x,
                 eta_data[our_algo][attribution],
                 label=algorithm_name_in_fig[algorithm_list.index(our_algo)] + " " + r'($\eta={}$)'.format(eta_),
                 color=color_list[idx],
                 marker=marker_list[idx])
        plt.plot(x,
                 eta_data[compared_algo][attribution],
                 label=algorithm_name_in_fig[algorithm_list.index(compared_algo)] + " " + r'($\eta={}$)'.format(eta_),
                 color=color_list[idx],
                 marker=marker_list[idx],
                 linestyle="--")

    # n_col = 1 if attribution != "max_delay" else 1
    # if in_chinese:
    #     leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 4}, loc='best', ncol=n_col)
    # else:
    #     leg = plt.legend(fontsize=fontsize_legend + 4, loc='best', ncol=n_col)
    # leg.set_draggable(state=True)
    # plt.show()

    if shared_legend and attribution in ["cost", "max_delay"]:
        plt.tight_layout()
        plt.show()
        return
    else:
        if shared_legend and attribution == "target_value":
            leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 6},
                             bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, mode="expand", ncol=2)
            frame = leg.get_frame()
            frame.set_linewidth(1.0)
            frame.set_edgecolor('black')
        else:
            if in_chinese:
                leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 5}, loc='best')
            else:
                leg = plt.legend(fontsize=fontsize_legend + 5, loc='best')
        leg.set_draggable(state=True)
        plt.tight_layout()
        plt.show()


# fixme
def draw_merged_eta_for_some_attribution_with_error_bar(res_dict, attribution: str, our_algo="Ours", compared_algo="M-Greedy-V2(Tx+Tp+Tq)"):
    plt.figure(figsize=figure_size, dpi=dpi)

    y_label = ""
    if attribution == "target_value":
        y_label = "Weighted Sum of Maximum\nLatency and Cost"
    elif attribution == "max_delay":
        y_label = "Maximum Interaction Latency (ms)"
    elif attribution == "cost":
        y_label = "Average Cost"
    plt.ylabel(ylabel=y_label, fontsize=fontsize+10, labelpad=10)
    plt.xlabel("Number of Users", fontsize=fontsize+10, labelpad=10)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize + 8)
    plt.yticks(fontsize=fontsize + 8)

    for idx, eta_ in enumerate(etas):
        eta_data = res_dict[eta_]

        y1 = [d[0] for d in eta_data[our_algo][attribution]]
        error_range = [[], []]
        for i_, range_ in enumerate(eta_data[our_algo][attribution]):
            error_range[0].append(y1[i_] - range_[1][0])
            error_range[1].append(range_[1][1] - y1[i_])
        plt.plot(x,
                 y1,
                 label=algorithm_name_in_fig[algorithm_list.index(our_algo)] + r'($\eta={}$)'.format(eta_),
                 color=color_list[idx],
                 marker=marker_list[idx])
        plt.errorbar(x, y1, yerr=error_range, elinewidth=2, capsize=4)

        y2 = [d[0] for d in eta_data[compared_algo][attribution]]
        error_range = [[], []]
        for i_, range_ in enumerate(eta_data[compared_algo][attribution]):
            error_range[0].append(y2[i_] - range_[1][0])
            error_range[1].append(range_[1][1] - y2[i_])
        plt.plot(x,
                 y2,
                 label=algorithm_name_in_fig[algorithm_list.index(compared_algo)] + r'($\eta={}$)'.format(eta_),
                 color=color_list[idx],
                 marker=marker_list[idx],
                 linestyle="--")
        plt.errorbar(x, y2, yerr=error_range, elinewidth=2, capsize=4)

    leg = plt.legend(fontsize=fontsize_legend + 2, loc='best')
    leg.set_draggable(state=True)
    plt.show()

def draw_histogram_with_error_bar(res_dict, attribution: str, our_algo="Ours", compared_algo="M-Greedy-V2(Tx+Tp+Tq)"):
    fig_size = (20, 12)
    plt.figure(figsize=fig_size, dpi=dpi)
    y_label = ""
    if attribution == "target_value":
        y_label = "Weighted Sum of Maximum\nLatency and Cost"
    elif attribution == "max_delay":
        y_label = "Maximum Interaction Latency (ms)"
    elif attribution == "cost":
        y_label = "Average Cost"
    plt.ylabel(ylabel=y_label, fontsize=fontsize + 10, labelpad=10)
    plt.xlabel("Number of Users", fontsize=fontsize + 10, labelpad=10)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x, fontsize=fontsize + 8)
    plt.yticks(fontsize=fontsize + 8)
    if attribution == "target_value":
        plt.ylim([150, 350])

    bar_width = 1.1
    gap_width = 0.3
    for idx_, x_ in enumerate(x):
        n_bar = len(etas) * 2
        # xs = [x_ + offs*bar_width + bar_width/2 for offs in range(-n_bar//2, n_bar//2+1, 1)]
        xs = [x_ - 3*bar_width - 2.5*gap_width + bar_width/2]
        for _ in range(1, n_bar):
            xs.append(xs[-1] + bar_width + gap_width)

        x_id = 0
        for eta_ in etas:
            eta_data = res_dict[eta_]

            y_ = eta_data[our_algo][attribution][idx_][0]
            error_range = [y_ - [eta_data[our_algo][attribution][idx_][1][0]], [eta_data[our_algo][attribution][idx_][1][1] - y_]]
            plt.bar(xs[x_id],
                    y_,
                    width=bar_width,
                    label=algorithm_name_in_fig[algorithm_list.index(our_algo)] + r'($\eta={}$)'.format(eta_),
                    yerr=error_range,
                    capsize=4,
                    color=color_list[x_id])
            x_id += 1

            y_ = eta_data[compared_algo][attribution][idx_][0]
            error_range = [y_ - [eta_data[compared_algo][attribution][idx_][1][0]], [eta_data[compared_algo][attribution][idx_][1][1] - y_]]
            plt.bar(x=xs[x_id],
                    height=y_,
                    width=bar_width,
                    label=algorithm_name_in_fig[algorithm_list.index(compared_algo)] + r'($\eta={}$)'.format(eta_),
                    yerr=error_range,
                    capsize=4,
                    color=color_list[x_id])
            x_id += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    leg = plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize_legend + 2, loc='best')
    leg.set_draggable(state=True)
    plt.show()


if __name__ == '__main__':
    file_name = "min_max/12-26_eta{}_new_conf"
    # raw_result = process_data(file_name)
    # print(raw_result)
    #
    # target_value_reduction_ratios = get_reduction_ratio(raw_result, "target_value")
    # draw_reduction_ratio(target_value_reduction_ratios, "target_value")
    #
    # max_delay_reduction_ratios = get_reduction_ratio(raw_result, "max_delay")
    # draw_reduction_ratio(max_delay_reduction_ratios, "max_delay")
    #
    # avg_cost_reduction_ratios = get_reduction_ratio(raw_result, "cost")
    # draw_reduction_ratio(avg_cost_reduction_ratios, "avg_cost")

    # get_and_draw_all_reduction_ratios(raw_result)
    #
    # offloading_distribution = get_offloading_ratio(raw_result)
    # print(offloading_distribution)

    raw_result = process_data(file_name)
    draw_merged_eta_for_some_attribution(raw_result, "target_value")
    draw_merged_eta_for_some_attribution(raw_result, "max_delay")
    draw_merged_eta_for_some_attribution(raw_result, "cost")

    offloading_distribution = get_offloading_ratio(raw_result)
    print(offloading_distribution)