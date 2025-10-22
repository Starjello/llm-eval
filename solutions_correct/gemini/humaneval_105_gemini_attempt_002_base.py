def by_length(arr):
    if not arr:
        return []

    names = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
    }

    filtered_and_sorted = sorted([x for x in arr if 1 <= x <= 9])

    reversed_arr = filtered_and_sorted[::-1]

    result = [names[x] for x in reversed_arr]

    return result