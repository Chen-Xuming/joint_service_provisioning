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
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

algorithm_list = ["Nearest", "M-Greedy", "SP-Max-First", "Ours"]

# 获取一组实验的json文件的路径
def get_json_file_list(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix == ".json":
            json_file = os.path.join(dir_path, file_name)
            files.append(json_file)
    return files

eta_range = (0.25, 1.0)
eta_step = 0.25
user_range = (40, 100)
user_step = 10
num_algorithms = len(algorithm_list)

def process_data(dir_path):
    json_file_list = get_json_file_list(dir_path)

    data = {}

    eta = eta_range[0]
    while eta <= eta_range[1]:
        data[eta] = {}

        for user_num in range(user_range[0], user_range[1] + user_step, user_step):
            data[eta][user_num] = {}
            for alg in algorithm_list:
                data[eta][user_num][alg] = {
                    "max_delay": [],
                    "avg_cost": [],
                    "target_value": [],
                    "running_time": [],
                    "local_count": [],
                    "common_count": [],
                }
        eta += eta_step

    """
        将所有文件的数据汇集到一起
    """
    for file in json_file_list:
        print(file)
        single_file_data = json.load(open(file))

        eta = eta_range[0]
        while eta <= eta_range[1]:
            data_each_eta = single_file_data[str(eta)]

            for user_num in range(user_range[0], user_range[1] + user_step, user_step):
                data_each_user_num = data_each_eta[str(user_num)]

                for alg in algorithm_list:
                    data_each_alg = data_each_user_num[alg]

                    data[eta][user_num][alg]["max_delay"].extend(data_each_alg["max_delay"])
                    data[eta][user_num][alg]["avg_cost"].extend(data_each_alg["avg_cost"])
                    data[eta][user_num][alg]["target_value"].extend(data_each_alg["target_value"])
                    data[eta][user_num][alg]["running_time"].extend(data_each_alg["running_time"])
                    data[eta][user_num][alg]["local_count"].extend(data_each_alg["local_count"])
                    data[eta][user_num][alg]["common_count"].extend(data_each_alg["common_count"])

            eta += eta_step

    """
        对每个子数组，求平均值
    """
    eta = eta_range[0]
    while eta <= eta_range[1]:

        for user_num in range(user_range[0], user_range[1] + user_step, user_step):
            data_each_user_num = data[eta][user_num]

            for alg in algorithm_list:

                data[eta][user_num][alg]["max_delay"] = np.mean(data[eta][user_num][alg]["max_delay"])
                data[eta][user_num][alg]["avg_cost"] = np.mean(data[eta][user_num][alg]["avg_cost"])
                data[eta][user_num][alg]["target_value"] = np.mean(data[eta][user_num][alg]["target_value"])
                data[eta][user_num][alg]["running_time"] = np.mean(data[eta][user_num][alg]["running_time"])
                data[eta][user_num][alg]["local_count"] = np.mean(data[eta][user_num][alg]["local_count"])
                data[eta][user_num][alg]["common_count"] = np.mean(data[eta][user_num][alg]["common_count"])

        eta += eta_step

    """
        每个算法，对每项指标不同的用户数的值汇集在一起
        {
            0.25:{
                "Nearest": {
                    "max_delay": [...],
                    "avg_cost": [...],
                    ...
                },
                ...
            },
            ...
        }
    """
    result = {}
    eta = eta_range[0]
    while eta <= eta_range[1]:
        result[eta] = {}
        for alg in algorithm_list:
            result[eta][alg] = {
                "max_delay": [],
                "avg_cost": [],
                "target_value": [],
                "running_time": [],
                "local_count": [],
                "common_count": []
            }

        for user_num, dict_ in data[eta].items():
            for alg, alg_res in dict_.items():
                for k, v in alg_res.items():
                    result[eta][alg][k].append(alg_res[k])

        eta += eta_step

    return result

"""
    {
        eta1: [...],
        eta2: [...],
        ...
    }
"""
def get_reduction_ratio(res_dict, attribute: str, original_algorithm="M-Greedy", target_algorithm="Ours"):
    reduction_ratios = {}
    for eta, dict_ in res_dict.items():
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
    file_dir = "min_max/11-05_eta0.25-1.0_minmax-v2"
    result_dict = process_data(file_dir)

    # target_value_reduction_ratios = get_reduction_ratio(result_dict, "target_value", original_algorithm="Nearest")
    target_value_reduction_ratios = get_reduction_ratio(result_dict, "target_value")
    draw_reduction_ratio(target_value_reduction_ratios, "target_value")

    # offloading_distribution = get_offloading_ratio(result_dict)
    # print(offloading_distribution)
