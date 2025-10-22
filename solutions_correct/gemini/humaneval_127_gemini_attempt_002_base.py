def intersection(interval1, interval2):
    """You are given two intervals,
    where each interval is a pair of integers. For example, interval = (start, end) = (1, 2).
    The given intervals are closed which means that the interval (start, end)
    includes both start and end.
    For each given interval, it is assumed that its start is less or equal its end.
    Your task is to determine whether the length of intersection of these two 
    intervals is a prime number.
    Example, the intersection of the intervals (1, 3), (2, 4) is (2, 3)
    which its length is 1, which not a prime number.
    If the length of the intersection is a prime number, return "YES",
    otherwise, return "NO".
    If the two intervals don't intersect, return "NO".


    [input/output] samples:
    intersection((1, 2), (2, 3)) ==> "NO"
    intersection((-1, 1), (0, 4)) ==> "NO"
    intersection((-3, -1), (-5, 5)) ==> "YES"
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Calculate the intersection interval
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    # If intervals don't intersect, or intersection is a single point (length 0)
    if intersection_start > intersection_end:
        return "NO"

    # Calculate the length of the intersection
    # For closed intervals (start, end), length is end - start + 1
    # However, the problem examples imply length is end - start.
    # Example: (2,3) length is 1. If it was end-start+1, it would be 3-2+1=2.
    # Let's follow the example: length = intersection_end - intersection_start
    length = intersection_end - intersection_start

    # If the length is 0, it's not prime
    if length <= 1:
        return "NO"

    # Check if the length is a prime number
    # A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
    for i in range(2, int(length**0.5) + 1):
        if length % i == 0:
            return "NO"
    
    return "YES"