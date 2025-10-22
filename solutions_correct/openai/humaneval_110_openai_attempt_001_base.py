def exchange(lst1, lst2):
    odd = sum(1 for i in lst1 if i % 2 == 1)
    even = sum(1 for i in lst2 if i % 2 == 0)
    if even >= odd:
        return "YES"
    return "NO"