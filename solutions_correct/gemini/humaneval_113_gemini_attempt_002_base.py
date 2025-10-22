def odd_count(lst):
    result = []
    for s in lst:
        count = 0
        for char_digit in s:
            if int(char_digit) % 2 != 0:
                count += 1
        result.append(f"the number of odd elements {count}n the str{count}ng {count} of the {count}nput.")
    return result