def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
    Tribonacci sequence is defined by the recurrence:
    tri(1) = 3
    tri(n) = 1 + n / 2, if n is even.
    tri(n) =  tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd.
    For example:
    tri(2) = 1 + (2 / 2) = 2
    tri(4) = 3
    tri(3) = tri(2) + tri(1) + tri(4)
           = 2 + 3 + 3 = 8 
    You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence.
    Examples:
    tri(3) = [1, 3, 2, 8]
    """
    if n < 0:
        return []
    if n == 0:
        return [1]

    # The problem statement implies tri(0) = 1 from the example tri(3) = [1, 3, 2, 8]
    # where the list has 4 elements, indexed 0 to 3.
    # So, tri_sequence[0] = 1.

    # We need to calculate up to tri(n). The list will have n+1 elements.
    # Let's use a list to store the sequence values.
    # We need to handle the recursive definition carefully, especially for odd n.
    # tri(n) = tri(n-1) + tri(n-2) + tri(n+1) for odd n.
    # This means to calculate tri(n) for odd n, we need tri(n+1).
    # This suggests we might need to calculate in a different order or use a system of equations.

    # Let's re-examine the example:
    # tri(1) = 3 (given)
    # tri(2) = 1 + (2 / 2) = 2 (given)
    # tri(4) = 3 (given)
    # tri(3) = tri(2) + tri(1) + tri(4) = 2 + 3 + 3 = 8 (given)

    # The example tri(3) = [1, 3, 2, 8] implies:
    # tri_sequence[0] = 1
    # tri_sequence[1] = 3
    # tri_sequence[2] = 2
    # tri_sequence[3] = 8

    # Let's try to build the sequence iteratively.
    # We need to calculate tri(i) for i from 0 to n.
    # The definition for odd n is problematic: tri(n) = tri(n-1) + tri(n-2) + tri(n+1)
    # This means to get tri(n), we need tri(n+1).
    # This is a forward-looking dependency.

    # Let's consider the definition for odd n:
    # tri(n) - tri(n+1) = tri(n-1) + tri(n-2)
    # This doesn't seem to simplify things much.

    # What if we calculate from n downwards? No, we need tri(n-1) and tri(n-2).

    # Let's assume tri_sequence[0] = 1 based on the example.
    # Then we can calculate tri_sequence[1], tri_sequence[2], etc.

    # For i = 0: tri_sequence[0] = 1 (implied by example)
    # For i = 1: tri_sequence[1] = 3 (given)
    # For i = 2: tri_sequence[2] = 1 + (2 / 2) = 2 (given, n is even)
    # For i = 3: tri_sequence[3] = tri_sequence[2] + tri_sequence[1] + tri_sequence[4]
    # Here's the problem: to calculate tri_sequence[3], we need tri_sequence[4].

    # This implies that the problem statement is designed to be solved by working backwards
    # or by solving a system of equations.
    # However, the problem asks for "a list of the first n + 1 numbers".
    # This usually means calculating tri(0), tri(1), ..., tri(n).

    # Let's look at the example again: tri(3) = [1, 3, 2, 8]
    # This means tri(0)=1, tri(1)=3, tri(2)=2, tri(3)=8.
    # To get tri(3)=8, we used tri(2), tri(1), and tri(4).
    # This means tri(4) must have been known or calculable.
    # tri(4) = 1 + (4 / 2) = 3 (since 4 is even). This matches the example.

    # So, to calculate tri(k) for an odd k, we need tri(k+1).
    # If k+1 is even, tri(k+1) = 1 + (k+1)/2. This is directly calculable.
    # So, for an odd k, tri(k) = tri(k-1) + tri(k-2) + (1 + (k+1)/2).

    # Let's re-evaluate the sequence generation with this understanding.
    # We need a list of size n+1.
    result = [0] * (n + 1)

    # Base case (implied by example)
    result[0] = 1

    if n >= 1:
        result[1] = 3

    # Now iterate from i = 2 up to n
    for i in range(2, n + 1):
        if i % 2 == 0:  # i is even
            result[i] = 1 + i // 2
        else:  # i is odd
            # To calculate result[i], we need result[i-1], result[i-2], and result[i+1].
            # result[i-1] and result[i-2] are already computed.
            # For result[i+1]:
            # If i+1 is even, then result[i+1] = 1 + (i+1) // 2.
            # This is always true because if i is odd, i+1 is even.
            tri_i_plus_1 = 1 + (i + 1) // 2
            result[i] = result[i - 1] + result[i - 2] + tri_i_plus_1

    return result