def f(n):
    result = []
    current_factorial = 1
    for i in range(1, n + 1):
        if i % 2 == 0:
            current_factorial *= i
            result.append(current_factorial)
        else:
            sum_up_to_i = (i * (i + 1)) // 2
            result.append(sum_up_to_i)
            if i > 1: # Reset current_factorial for the next even number
                current_factorial = sum_up_to_i # This is wrong, factorial should be calculated independently or reset
                                                # Let's re-think the factorial calculation.
                                                # It's better to calculate factorial from scratch or keep a separate variable.
                                                # The problem states "the factorial of i", not "the factorial of the previous even number times i".
                                                # So, if i is even, calculate i!
                                                # If i is odd, calculate sum(1..i)
                                                # The current_factorial variable should only be used for even numbers.
                                                # Let's restart the logic for factorial.
    
    # Corrected logic:
    result = []
    for i in range(1, n + 1):
        if i % 2 == 0:
            # Calculate factorial of i
            fact_i = 1
            for j in range(1, i + 1):
                fact_i *= j
            result.append(fact_i)
        else:
            # Calculate sum from 1 to i
            sum_i = (i * (i + 1)) // 2
            result.append(sum_i)
    return result