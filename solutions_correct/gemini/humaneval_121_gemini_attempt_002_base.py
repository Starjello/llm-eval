def solution(lst):
    total_sum = 0
    for i in range(len(lst)):
        if i % 2 == 0:  # Check if the position (index) is even
            if lst[i] % 2 != 0:  # Check if the element at this even position is odd
                total_sum += lst[i]
    return total_sum