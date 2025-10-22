def digits(n):
    product = 1
    has_odd = False
    s_n = str(n)
    for char_digit in s_n:
        digit = int(char_digit)
        if digit % 2 != 0:
            product *= digit
            has_odd = True
    
    if not has_odd:
        return 0
    else:
        return product