def choose_num(x, y):
    if x > y:
        return -1

    if y % 2 == 0:
        return y
    elif (y - 1) >= x and (y - 1) % 2 == 0:
        return y - 1
    else:
        return -1