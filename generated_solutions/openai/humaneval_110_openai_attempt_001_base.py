def exchange(lst1, lst2):
    has_odd_in_lst1 = any(x % 2 != 0 for x in lst1)
    has_even_in_lst2 = any(x % 2 == 0 for x in lst2)
    return "YES" if not has_odd_in_lst1 or has_even_in_lst2 else "NO"