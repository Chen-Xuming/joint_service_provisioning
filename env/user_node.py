from env.site_node import SiteNode
from env.service import Service

class UserNode:
    def __init__(self, user_id, area_id):
        self.user_id = user_id
        self.area_id = area_id      # 所属于的area
        self.arrival_rate = 0

        self.service_a = None       # type: Service
        self.service_r = None       # type: Service

        self.node_name = "u{}".format(user_id)

    def reset(self):
        self.service_a.reset()
        self.service_r.reset()