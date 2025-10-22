def exchange(lst1, lst2):
    odd_in_lst1 = any(x % 2 != 0 for x in lst1)
    even_in_lst2 = any(x % 2 == 0 for x in lst2)
    return "YES" if odd_in_lst1 and even_in_lst2 else "NO"