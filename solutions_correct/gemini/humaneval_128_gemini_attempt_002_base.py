def prod_signs(arr):
    if not arr:
        return None

    product_of_signs = 1
    sum_of_magnitudes = 0
    has_zero = False

    for x in arr:
        if x == 0:
            has_zero = True
            break
        elif x < 0:
            product_of_signs *= -1
        else:
            product_of_signs *= 1
        sum_of_magnitudes += abs(x)

    if has_zero:
        return 0
    else:
        return sum_of_magnitudes * product_of_signs