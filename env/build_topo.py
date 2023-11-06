from env.area import Area
from env.site_node import SiteNode
from configuration.config import topo_config

import networkx as nx
import matplotlib.pyplot as plt

def build_network_topo():
    common_site_num = topo_config["common_area_site_num"]
    edge_site_num = topo_config["edge_area_site_num"]
    edge_area_num = topo_config["edge_area_num"]

    net_topo = nx.Graph()   # 无向图，边的权重是传输时延

    """ 1. 生成 Common Area """