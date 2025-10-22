def is_palindrome(text):
    return text == text[::-1]

def make_palindrome(string):
    """
    Find the shortest palindrome that can be formed by adding characters to the end of the string.
    This is done by finding the longest palindromic suffix of the string, and then appending
    to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """
    if not string:
        return ""
    n = len(string)
    for i in range(n, -1, -1):
        if is_palindrome(string[i:]):
            prefix_to_reverse = string[:i]
            return string + prefix_to_reverse[::-1]
    return string + string[::-1]

print(make_palindrome('dog'))