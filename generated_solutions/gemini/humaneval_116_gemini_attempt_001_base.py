def sort_array(arr):
    """
    In this Kata, you have to sort an array of non-negative integers according to
    number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    It must be implemented like this:
    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]
    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]
    >>> sort_array([1, 0, 2, 3, 4]) == [0, 1, 2, 3, 4]
    """
    if not arr:
        return []

    # The problem description and examples are contradictory for negative numbers.
    # The description says "non-negative integers", but the example shows negative numbers sorted numerically.
    # Given the example `sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]`,
    # it implies that if all numbers are negative, they should be sorted numerically.
    # If there's a mix or only non-negative, the bit count logic applies.
    # However, the core problem is about "number of ones in their binary representation",
    # which is typically defined for non-negative integers.
    #
    # Let's assume the "non-negative integers" in the description is the primary rule
    # for the bit-counting sort, and the negative example is a special case
    # where the bit-counting rule doesn't apply (or is undefined/trivial for negative numbers
    # in a way that defaults to numerical sort).
    #
    # If all numbers are negative, we just sort them numerically.
    if all(x < 0 for x in arr):
        return sorted(arr)

    # Otherwise, apply the bit-counting sort.
    # For non-negative numbers, count bits.
    # For negative numbers, if they are mixed with non-negative,
    # the problem doesn't specify. The most reasonable interpretation
    # for "non-negative integers" is that the bit-counting logic
    # only applies to non-negative numbers. If negative numbers are present
    # and mixed, their position relative to non-negative numbers is ambiguous
    # based on the description.
    #
    # Given the examples, it seems the intent is:
    # 1. If all elements are negative, sort numerically.
    # 2. Otherwise (mixed or all non-negative), apply the bit-count sort.
    #    For non-negative numbers, `bin(x).count('1')` works.
    #    For negative numbers, `bin(x).count('1')` would count ones in their
    #    two's complement representation, which is not usually what's meant
    #    by "number of ones" in this context unless specified.
    #    The simplest interpretation that aligns with "non-negative integers"
    #    is to only apply the bit-count logic to non-negative numbers.
    #
    # Let's stick to the "non-negative integers" part of the description
    # for the bit-counting logic.
    #
    # The example `sort_array([1, 0, 2, 3, 4]) == [0, 1, 2, 3, 4]`
    # shows 0 is handled. `bin(0)` is '0b0', `count('1')` is 0.
    # `bin(1)` is '0b1', `count('1')` is 1.
    # `bin(2)` is '0b10', `count('1')` is 1.
    # `bin(3)` is '0b11', `count('1')` is 2.
    # `bin(4)` is '0b100', `count('1')` is 1.
    #
    # Sorting by (bit_count, decimal_value):
    # 0: (0, 0)
    # 1: (1, 1)
    # 2: (1, 2)
    # 4: (1, 4)
    # 3: (2, 3)
    #
    # Result: [0, 1, 2, 4, 3]
    #
    # This contradicts `sort_array([1, 0, 2, 3, 4]) == [0, 1, 2, 3, 4]`.
    #
    # This implies that the problem statement "sort an array of non-negative integers according to number of ones"
    # and the example `[1, 0, 2, 3, 4] == [0, 1, 2, 3, 4]` are contradictory.
    #
    # Let's re-evaluate the example `[1, 5, 2, 3, 4] == [1, 2, 3, 4, 5]`.
    # 1: (1, 1)
    # 2: (1, 2)
    # 3: (2, 3)
    # 4: (1, 4)
    # 5: (2, 5)
    #
    # Sorted by (bit_count, decimal_value):
    # 1: (1, 1)
    # 2: (1, 2)
    # 4: (1, 4)
    # 3: (2, 3)
    # 5: (2, 5)
    #
    # Result: [1, 2, 4, 3, 5]
    # This also contradicts `[1, 5, 2, 3, 4] == [1, 2, 3, 4, 5]`.
    #
    # The examples provided `[1, 5, 2, 3, 4] == [1, 2, 3, 4, 5]` and `[1, 0, 2, 3, 4] == [0, 1, 2, 3, 4]`
    # simply show a numerical sort. This means the problem description is completely at odds with the examples.
    #
    # Given the "It must be implemented like this:" and the examples, the examples take precedence.
    # The examples show a simple numerical sort for both positive and negative numbers.
    #
    # Therefore, the most straightforward interpretation that satisfies ALL examples is a simple numerical sort.
    # The description about "number of ones in their binary representation" seems to be a red herring
    # or a misstatement given the provided examples.

    return sorted(arr)