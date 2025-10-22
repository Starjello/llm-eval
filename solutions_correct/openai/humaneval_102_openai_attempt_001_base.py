def choose_num(x, y):
    if x > y:
        return -1
    if y % 2 != 0:
        y -= 1
    if x % 2 != 0:
        x += 1
    if x > y:
        return -1
    return y