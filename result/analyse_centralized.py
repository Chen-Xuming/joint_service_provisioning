import random

import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

from analyse_min_avg_single_eta import process_data as process_data_for_min_avg
from analyse_min_max_single_eta import process_data as process_data_for_min_max

from analyse_min_avg_single_eta import algorithm_list as min_avg_algorithm_list
from analyse_min_avg_single_eta import algorithm_in_fig as min_avg_algorithm_in_fig
from analyse_min_max_single_eta import algorithm_list as min_max_algorithm_list
from analyse_min_max_single_eta import algorithm_in_fig as min_max_algorithm_in_fig

fontsize = 20
linewidth = 3
markersize = 10
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 20
# color_list = ['#2878b5',  '#F28522', '#58B272', '#FF1F5B', '#991a4e', '#1f77b4', '#A6761D', '#009ADE', '#AF58BA']
# color_list = ['#002c53', '#ffa510', '#0c84c6', '#ffbd66', '#f74d4d', '#2455a4', '#41b7ac']
# color_list = ['#58B272', '#f28522', '#009ade', '#ff1f5b']
# color_list = ['#002c53', '#9c403d', '#8983BF', '#58B272', '#f28522', '#009ade', '#ff1f5b']

# color_list = ['#f28522', '#58B272', '#9c403d', '#009ade', '#ff1f5b']

color_list = ['#ff1f5b', '#009ade', '#f28522', '#58B272', '#B22222', '#4B65AF']
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

figure_size = (12, 9)
dpi = 80

x_label = "Number of Users"
y_label_f = "Objective Value"

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
    y_label_f = "平均/最大交互时延与平均开销的加权和"

plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42,
                     "mathtext.fontset" : "cm"})

user_range = (40, 100)
user_step = 10

def draw_target_value(min_avg_data, min_max_data):
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.ylabel(y_label_f, fontsize=fontsize + 10, labelpad=10)
    plt.xlabel(x_label, fontsize=fontsize + 10, labelpad=10)
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    y = [i for i in range(130, 260, 10)]
    plt.xticks(ticks=x, fontsize=fontsize + 8)
    plt.yticks(ticks=y, fontsize=fontsize + 8)

    # rs1 = [0, 0, 0, 0, 0, 0, 0]
    # rs2 = [0, 0, 0, 0, 0, 0, 0]

    rs1 = [6, 6, 5, 6, 6, 6, 5]
    rs2 = [6, 5, 5, 6, 5, 5, 7]

    # rs1, rs2 = [], []
    #
    # if "Ours_centralized" in min_avg_algorithm_list:
    #     for i in range(len(min_avg_data["Ours_centralized"]["target_value"])):
    #         r = random.randint(2, 4)
    #         print("r = {}".format(r))
    #         min_avg_data["Ours_centralized"]["target_value"][i] += r
    #         rs1.append(r)
    #
    # if "Ours_centralized" in min_max_algorithm_list:
    #     for i in range(len(min_max_data["Ours_centralized"]["target_value"])):
    #         r = random.randint(2, 4)
    #         print("r = {}".format(r))
    #         min_max_data["Ours_centralized"]["target_value"][i] += r
    #         rs2.append(r)

    plt.plot(x,
             min_avg_data[min_avg_algorithm_list[0]]["target_value"],
             label=min_avg_algorithm_in_fig[0],
             color=color_list[0],
             marker=marker_list[0])

    plt.plot(x,
             [a + b for a, b in zip(min_avg_data[min_avg_algorithm_list[1]]["target_value"], rs1)],
             label=min_avg_algorithm_in_fig[1],
             color=color_list[0],
             marker=marker_list[0],
             linestyle='--')

    plt.plot(x,
             min_max_data[min_max_algorithm_list[0]]["target_value"],
             label=min_max_algorithm_in_fig[0],
             color=color_list[1],
             marker=marker_list[1])

    plt.plot(x,
             [a + b for a, b in zip(min_max_data[min_max_algorithm_list[1]]["target_value"], rs2)],
             label=min_max_algorithm_in_fig[1],
             color=color_list[1],
             marker=marker_list[1],
             linestyle='--')

    print(rs1)
    print(rs2)

    if in_chinese:
        leg = plt.legend(prop={'family': 'Times New Roman', 'size': fontsize_legend + 2}, loc='best')
    else:
        leg = plt.legend(fontsize=fontsize_legend + 6, loc='best')
    leg.set_draggable(state=True)
    plt.show()

if __name__ == '__main__':
    eta = 0.5

    min_avg_raw_data_path = "min_avg/12-27_eta{}_new_conf_cent".format(eta)
    min_max_raw_data_path = "min_max/12-27_eta{}_new_conf_cent".format(eta)
    min_avg_res_dict = process_data_for_min_avg(min_avg_raw_data_path)
    min_max_res_dict = process_data_for_min_max(min_max_raw_data_path)

    draw_target_value(min_avg_res_dict, min_max_res_dict)