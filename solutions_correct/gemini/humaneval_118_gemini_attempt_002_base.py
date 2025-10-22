def get_closest_vowel(word):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    
    # Iterate from right to left, excluding the first and last characters
    for i in range(len(word) - 2, 0, -1):
        char = word[i]
        
        # Check if the current character is a vowel
        if char in vowels:
            # Check if it's surrounded by consonants
            if word[i-1] in consonants and word[i+1] in consonants:
                return char
                
    return ""