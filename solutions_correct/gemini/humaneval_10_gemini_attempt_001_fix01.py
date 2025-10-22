def is_palindrome(text):
    return text == text[::-1]

def make_palindrome(string):
    """
    Find the longest palindromic suffix of a string and append the reverse of the prefix
    before that suffix to the end of the string to make the whole string a palindrome.
    For example:
    make_palindrome('cat') == 'catac'
    make_palindrome('cata') == 'catac'
    """
    if not string:
        return ""
    n = len(string)
    for i in range(n + 1):  # Iterate from 0 to n (inclusive)
        suffix = string[i:]
        if is_palindrome(suffix):
            prefix_to_reverse = string[:i]
            return string + prefix_to_reverse[::-1]
    return string