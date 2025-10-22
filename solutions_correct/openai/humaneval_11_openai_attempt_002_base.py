def string_xor(a: str, b: str) -> str:
    result = []
    for bit_a, bit_b in zip(a, b):
        result.append('1' if bit_a != bit_b else '0')
    return ''.join(result)