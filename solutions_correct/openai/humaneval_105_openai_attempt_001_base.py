def by_length(arr):
    if not arr:
        return []
    valid_numbers = [x for x in arr if 1 <= x <= 9]
    valid_numbers.sort()
    valid_numbers.reverse()
    number_names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return [number_names[x - 1] for x in valid_numbers]