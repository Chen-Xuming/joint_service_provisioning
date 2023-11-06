from env.site_node import SiteNode

class Area:
    def __init__(self, area_id, pos):
        self.area_id = area_id      # 当 area_id = 0 时是common_area，其余是 edge_area
        self.site_nodes = []        # 当前区域内的节点
        self.pos = pos              # 中心坐标

