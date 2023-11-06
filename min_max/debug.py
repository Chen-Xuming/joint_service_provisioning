import math
def get_x_v1(lamb, miu):
    num_server = 1
    while num_server * miu <= lamb:
        num_server += 1
    return num_server

def get_x_v2(lamb, miu):
    num_server = math.ceil(lamb / miu)
    return num_server

def check(lamb, miu):
    x1 = get_x_v1(lamb, miu)
    x2 = get_x_v2(lamb, miu)
    print(x1, x2)
    assert x1 == x2

check(8, 2.5)