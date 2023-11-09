""" 网络拓扑配置 """
config = {
    "common_area_site_num": 20,      # common-area 拥有的节点数
    "edge_area_site_num": 2,         # 每个 edge-area 拥有的节点数
    "edge_area_num": 10,
    "global_radius": 20,
    "edge_radius": 2,
    "global_min_distance": 4,
    "tx_edge_common_range": (6, 10),
    "tx_edge_edge_range": (2, 5),
    "tx_user_edge_range": (1, 3),
    "show_graph": False,

    # "data_size": (3, 1, 6),          # 传输数据量        (3, 1, 6)
    "data_size": (2, 1, 3),

    # "minmax_arrival_rate": (2, 6),
    # "trigger_probability": 1.0,         # 无用
    # "minmax_service_rate_A_common": (50, 80),
    # "minmax_service_rate_R_common": (40, 60),
    # "minmax_price_A_common": (3, 6),                # (1, 2)
    # "minmax_price_R_common": (5, 8),                # (1, 2)
    # "minmax_service_rate_A_edge": (30, 50),
    # "minmax_service_rate_R_edge": (20, 35),
    # "minmax_price_A_edge": (5, 8),                  # (2, 4)
    # "minmax_price_R_edge": (9, 14),                 # (2, 4)

    "minmax_arrival_rate": (2, 6),
    "trigger_probability": 1.0,         # 无用
    "minmax_service_rate_A_common": (50, 80),
    "minmax_service_rate_R_common": (40, 60),
    "minmax_price_A_common": (8, 15),                # (1, 2)
    "minmax_price_R_common": (12, 20),                # (1, 2)
    "minmax_service_rate_A_edge": (30, 50),
    "minmax_service_rate_R_edge": (20, 35),
    "minmax_price_A_edge": (12, 18),                  # (2, 4)
    "minmax_price_R_edge": (15, 25),                 # (2, 4)

    # "minmax_arrival_rate": (2, 6),
    # "trigger_probability": 1.0,         # 无用
    # "minmax_service_rate_A_common": (50, 80),
    # "minmax_service_rate_R_common": (40, 60),
    # "minmax_price_A_common": (3, 10),                # (1, 2)
    # "minmax_price_R_common": (5, 15),                # (1, 2)
    # "minmax_service_rate_A_edge": (30, 50),
    # "minmax_service_rate_R_edge": (20, 35),
    # "minmax_price_A_edge": (6, 12),                  # (2, 4)
    # "minmax_price_R_edge": (10, 20),                 # (2, 4)

    # edge-common 价格范围相同
    # "minmax_arrival_rate": (2, 6),
    # "trigger_probability": 0.9,
    # "minmax_service_rate_A_common": (50, 80),
    # "minmax_service_rate_R_common": (40, 60),
    # "minmax_price_A_common": (3, 6),                # (1, 2)
    # "minmax_price_R_common": (5, 8),                # (1, 2)
    # "minmax_service_rate_A_edge": (30, 50),
    # "minmax_service_rate_R_edge": (20, 35),
    # "minmax_price_A_edge": (3, 6),                  # (2, 4)
    # "minmax_price_R_edge": (5, 8),                  # (2, 4)

    # edge-common 服务率范围相同
    # "minmax_arrival_rate": (2, 6),
    # "trigger_probability": 0.9,
    # "minmax_service_rate_A_common": (50, 80),
    # "minmax_service_rate_R_common": (40, 60),
    # "minmax_price_A_common": (3, 6),                # (1, 2)
    # "minmax_price_R_common": (5, 8),                # (1, 2)
    # "minmax_service_rate_A_edge": (50, 80),
    # "minmax_service_rate_R_edge": (40, 60),
    # "minmax_price_A_edge": (5, 8),                  # (2, 4)
    # "minmax_price_R_edge": (9, 14),                  # (2, 4)

    # 所有节点完全同构（价格和服务率相同）
    # "minmax_arrival_rate": (2, 6),
    # "trigger_probability": 0.9,
    # "minmax_service_rate_A_common": (60, 60),
    # "minmax_service_rate_R_common": (40, 40),
    # "minmax_price_A_common": (3, 3),                # (1, 2)
    # "minmax_price_R_common": (5, 5),                # (1, 2)
    # "minmax_service_rate_A_edge": (60, 60),
    # "minmax_service_rate_R_edge": (40, 40),
    # "minmax_price_A_edge": (3, 3),                  # (2, 4)
    # "minmax_price_R_edge": (5, 5),                  # (2, 4)

    "eta": 0.25,
}

""" alpha 初始参考值 """
alpha_initial_values = dict()
eta_list = [0, 0.25, 0.5, 0.75, 1.0]
user_num_list = [u for u in range(40, 110, 10)]
for i, eta in enumerate(eta_list):
    alpha_initial_values[eta] = {}
    for user_num in user_num_list:
        if 40 <= user_num <= 60:
            alpha = [3e-5, 3e-5, 1e-5, 5e-6, 5e-6]
            alpha_initial_values[eta][user_num] = alpha[i]

        elif 70 <= user_num <= 80:
            alpha = [2e-5, 2e-5, 5e-6, 3e-6, 3e-6]
            alpha_initial_values[eta][user_num] = alpha[i]

        elif user_num > 80:
            alpha = [1e-5, 1e-5, 3e-6, 1e-6, 1e-6]
            alpha_initial_values[eta][user_num] = alpha[i]

if __name__ == '__main__':
    for eta, eta_dict in alpha_initial_values.items():
        print("-----------------------------")
        print("eta = {}".format(eta))
        print("num_user\talpha")
        for nu, alpha in eta_dict.items():
            print("{}\t{}".format(nu, alpha))














