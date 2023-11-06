class SiteNode:
    """
    :param
    global_id: 节点的全局 id
    local_id:  节点在 area 内的 id
    area_id:    area id
    pos:        坐标
    """
    def __init__(self, global_id, local_id, area_id, pos):
        self.global_id = global_id
        self.local_id = local_id
        self.area_id = area_id
        self.pos = pos

        # 服务率和处理时延
        self.service_rate_a = None      # 个/秒
        self.service_rate_r = None
        self.tp_a = None                # 秒
        self.tp_r = None

        # 服务单价
        self.price_a = None
        self.price_r = None