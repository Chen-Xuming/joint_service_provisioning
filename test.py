n = 0b111

print(bin(4))

def consider_what(x):
    flags = [bool(x & 4), bool(x & 2), bool(x & 1)]
    print(flags)

consider_what(0b100)
consider_what(0b110)
consider_what(0b111)
consider_what(0b101)
