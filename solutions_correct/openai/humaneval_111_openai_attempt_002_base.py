def histogram(test):
    if not test:
        return {}
    
    counts = {}
    for letter in test.split():
        counts[letter] = counts.get(letter, 0) + 1
    
    max_count = max(counts.values())
    return {letter: count for letter, count in counts.items() if count == max_count}