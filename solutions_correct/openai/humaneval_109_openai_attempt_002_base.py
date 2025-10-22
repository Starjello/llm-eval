def move_one_ball(arr):
    if not arr:
        return True
    n = len(arr)
    count = 0
    for i in range(n):
        if arr[i] > arr[(i + 1) % n]:
            count += 1
    return count <= 1