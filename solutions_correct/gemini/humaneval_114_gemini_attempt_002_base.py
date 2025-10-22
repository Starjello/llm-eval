def minSubArraySum(nums):
    min_so_far = nums[0]
    current_min = nums[0]

    for i in range(1, len(nums)):
        current_min = min(nums[i], current_min + nums[i])
        min_so_far = min(min_so_far, current_min)

    return min_so_far