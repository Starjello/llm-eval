import heapq

def maximum(arr, k):
    if k == 0:
        return []
    
    # Use a min-heap to keep track of the k largest elements.
    # The smallest element among the k largest will be at the root.
    min_heap = []

    for num in arr:
        if len(min_heap) < k:
            heapq.heappush(min_heap, num)
        else:
            # If the current number is greater than the smallest among the k largest,
            # remove the smallest and add the current number.
            if num > min_heap[0]:
                heapq.heapreplace(min_heap, num)
    
    # The min_heap now contains the k largest elements.
    # Sort them to get the final result.
    min_heap.sort()
    return min_heap