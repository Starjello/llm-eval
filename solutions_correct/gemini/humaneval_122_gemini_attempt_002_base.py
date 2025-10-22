def add_elements(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return
    the sum of the elements with at most two digits from the first k elements of arr.

    Example:

        Input: arr = [111,21,3,4000,5,6,7,8,9], k = 4
        Output: 24 # sum of 21 + 3

    Constraints:
        1. 1 <= len(arr) <= 100
        2. 1 <= k <= len(arr)
    """
    total_sum = 0
    for i in range(min(k, len(arr))):  # Ensure we don't go out of bounds if k is larger than arr length
        element = arr[i]
        # Check if the element has at most two digits
        # This means the absolute value of the number is less than 100
        # For example, -99, -1, 0, 1, 99 are all at most two digits.
        if -99 <= element <= 99:
            total_sum += element
    return total_sum