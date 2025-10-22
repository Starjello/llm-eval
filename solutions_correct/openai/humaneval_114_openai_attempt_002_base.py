def minSubArraySum(nums):
    min_sum = float('inf')
    current_sum = 0
    
    for num in nums:
        current_sum += num
        if current_sum < min_sum:
            min_sum = current_sum
        if current_sum > 0:
            current_sum = 0
            
    return min_sum

# Test cases
print(minSubArraySum([2, 3, 4, 1, 2, 4]))  # Expected output: 1
print(minSubArraySum([-1, -2, -3]))          # Expected output: -6
print(minSubArraySum([1, -1, -2, 3, -4]))    # Expected output: -4
print(minSubArraySum([5, 4, 3, 2, 1]))       # Expected output: 1
print(minSubArraySum([-5, -4, -3, -2, -1]))  # Expected output: -15