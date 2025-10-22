import heapq

def minPath(grid, k):
    n = len(grid)
    
    # Store (value, row, col) for each cell to easily find starting points
    cell_info = []
    for r in range(n):
        for c in range(n):
            cell_info.append((grid[r][c], r, c))
    
    # Sort by value to prioritize smaller starting values
    cell_info.sort()

    # Priority queue to store (current_path_tuple, current_row, current_col, path_length)
    # The path_tuple is used for lexicographical comparison.
    # We use a tuple of values for the path, not a list, because tuples are hashable and comparable.
    pq = []

    # Initialize PQ with all possible starting cells
    for val, r, c in cell_info:
        if k == 1:
            return [val] # Special case for k=1, just return the smallest value
        heapq.heappush(pq, ((val,), r, c, 1)) # path_tuple, row, col, current_length

    min_path_found = None

    while pq:
        current_path_tuple, r, c, length = heapq.heappop(pq)

        if length == k:
            # If we found a path of length k, it's a candidate.
            # Since heapq pops the lexicographically smallest first,
            # the first path of length k we extract is the answer.
            return list(current_path_tuple)

        # Explore neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < n:
                next_val = grid[nr][nc]
                new_path_tuple = current_path_tuple + (next_val,)
                heapq.heappush(pq, (new_path_tuple, nr, nc, length + 1))

    return [] # Should not be reached based on problem constraints (answer is unique)