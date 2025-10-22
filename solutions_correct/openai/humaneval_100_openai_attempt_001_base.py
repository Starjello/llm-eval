def make_a_pile(n):
    pile = []
    for i in range(n):
        if i == 0:
            pile.append(n)
        else:
            if n % 2 == 0:
                pile.append(pile[-1] + 2)
            else:
                pile.append(pile[-1] + 2)
    return pile