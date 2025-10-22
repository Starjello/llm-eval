def count_nums(arr):
    def sum_of_digits(n):
        return sum(int(d) if n >= 0 else -int(d) for d in str(abs(n)))
    
    return sum(1 for num in arr if sum_of_digits(num) > 0)