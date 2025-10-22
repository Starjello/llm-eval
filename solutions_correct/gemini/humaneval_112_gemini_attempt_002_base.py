def reverse_delete(s,c):
    """Task
    We are given two strings s and c, you have to deleted all the characters in s that are equal to any character in c
    then check if the result string is palindrome.
    A string is called palindrome if it reads the same backward as forward.
    You should return a tuple containing the result string and True/False for the check.
    Example
    For s = "abcde", c = "ae", the result should be ('bcd',False)
    For s = "abcdef", c = "b"  the result should be ('acdef',False)
    For s = "abcdedcba", c = "ab", the result should be ('cdedc',True)
    """
    chars_to_delete = set(c)
    result_chars = []
    for char_s in s:
        if char_s not in chars_to_delete:
            result_chars.append(char_s)
    
    result_string = "".join(result_chars)
    
    is_palindrome = result_string == result_string[::-1]
    
    return (result_string, is_palindrome)