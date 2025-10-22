import math

def max_fill(grid, capacity):
    total_water_needed = 0
    for row in grid:
        total_water_needed += sum(row)
    
    if capacity <= 0:
        # This case should ideally be handled or prevented by problem constraints.
        # If capacity is 0, it's impossible to fill, so it's an error or infinite.
        # For this problem's context, assuming capacity > 0.
        return 0 

    if total_water_needed == 0:
        return 0
    
    return math.ceil(total_water_needed / capacity)