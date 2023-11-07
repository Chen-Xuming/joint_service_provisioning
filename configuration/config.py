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

    "minmax_arrival_rate": (2, 6),
    "trigger_probability": 1.0,         # 无用
    "minmax_service_rate_A_common": (50, 80),
    "minmax_service_rate_R_common": (40, 60),
    "minmax_price_A_common": (3, 6),                # (1, 2)
    "minmax_price_R_common": (5, 8),                # (1, 2)
    "minmax_service_rate_A_edge": (30, 50),
    "minmax_service_rate_R_edge": (20, 35),
    "minmax_price_A_edge": (5, 8),                  # (2, 4)
    "minmax_price_R_edge": (9, 14),                 # (2, 4)

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

    "eta": 0,
}