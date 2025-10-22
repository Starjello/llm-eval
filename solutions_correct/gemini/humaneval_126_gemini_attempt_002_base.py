def is_sorted(lst):
    if not lst:
        return True

    counts = {}
    for x in lst:
        counts[x] = counts.get(x, 0) + 1
        if counts[x] > 2:
            return False

    for i in range(len(lst) - 1):
        if lst[i] > lst[i+1]:
            return False

    return True