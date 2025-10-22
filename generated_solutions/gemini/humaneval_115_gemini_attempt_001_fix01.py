import math

def max_fill(grid, capacity):
    total_water_needed = 0
    for row in grid:
        total_water_needed += sum(row)
    
    if total_water_needed == 0:
        return 0
    
    return math.ceil(total_water_needed / capacity)