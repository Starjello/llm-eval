def make_a_pile(n):
    pile = []
    current_stones = n
    for _ in range(n):
        pile.append(current_stones)
        if n % 2 == 0:  # n is even
            current_stones += 2
        else:  # n is odd
            current_stones += 2
    return pile