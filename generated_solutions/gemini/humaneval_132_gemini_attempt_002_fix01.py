def is_nested(string):
    stack = []
    max_depth = 0
    current_depth = 0
    for char in string:
        if char == '[':
            stack.append(char)
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ']':
            if stack and stack[-1] == '[':
                stack.pop()
                current_depth -= 1
            else:
                # If a closing bracket appears without a matching opening bracket,
                # or if the stack is empty, it's an invalid nesting sequence
                # for the purpose of determining if it's "nested" in the sense
                # of having at least two levels of *valid* nesting.
                # Resetting current_depth and stack here effectively means
                # any subsequent valid nesting won't count towards the depth
                # of the *current* invalid sequence.
                # However, the problem definition implies we are looking for
                # *any* valid nesting of depth 2 or more.
                # The original code's logic for `else` on ']' was problematic.
                # If we encounter ']' and stack is empty or top is not '[',
                # it means this ']' is unmatched. This should invalidate the
                # current nesting sequence for depth calculation.
                # The problem is subtle: "is_nested" implies a well-formed
                # structure to some extent. If it's just about max depth
                # regardless of validity, then `[[[]]]` and `[[[]` would both
                # yield max_depth 3. But `][` would yield max_depth 0.
                # The original code's `current_depth = 0; stack = []` on an
                # unmatched ']' seems to be an attempt to handle invalidity.
                # Let's re-evaluate the definition of "nested".
                # If it means "contains a substring that is well-nested to depth 2+",
                # then the current logic is almost right.
                # If it means "the entire string is well-nested to depth 2+",
                # then we need to check `not stack` at the end.

                # The original test case `[[[]]]` passes, `[]` fails.
                # `[[` passes, `][` fails.
                # This suggests it's about finding *any* sequence that reaches
                # a depth of 2 or more, and that unmatched closing brackets
                # should reset the current depth count for *that* sequence.
                # The original `current_depth = 0` and `stack = []` on an
                # unmatched ']' is actually a reasonable way to handle this
                # for the given problem's apparent intent.
                # The issue might be in the final return condition.

                # Let's trace `[[[]]]`
                # [ : stack=['['], cd=1, md=1
                # [ : stack=['[','['], cd=2, md=2
                # [ : stack=['[','[','['], cd=3, md=3
                # ] : stack=['[','['], cd=2, md=3
                # ] : stack=['['], cd=1, md=3
                # ] : stack=[], cd=0, md=3
                # return 3 >= 2 -> True. Correct.

                # Let's trace `[[`
                # [ : stack=['['], cd=1, md=1
                # [ : stack=['[','['], cd=2, md=2
                # return 2 >= 2 -> True. Correct.

                # Let's trace `[]`
                # [ : stack=['['], cd=1, md=1
                # ] : stack=[], cd=0, md=1
                # return 1 >= 2 -> False. Correct.

                # Let's trace `][`
                # ] : stack=[], else branch: cd=0, stack=[] (no change), md=0
                # [ : stack=['['], cd=1, md=1
                # return 1 >= 2 -> False. Correct.

                # The problem description is "Fix the module to pass tests."
                # The current code seems to correctly calculate max_depth based
                # on the given logic. The `AssertionError` suggests a test case
                # where `max_depth >= 2` is not the correct final condition.

                # What if the problem implies that the *entire* string must be
                # well-formed to be considered "nested"?
                # E.g., `[[[]]]` is nested. `[[[]` is not (unmatched '[').
                # `[[[]]]]` is not (unmatched ']').
                # If this is the case, we need to check `not stack` at the end.

                # Let's test this hypothesis:
                # `[[[]]]` -> md=3, stack=[] -> True.
                # `[[[]` -> md=3, stack=['[','[','['] -> False (if `not stack` is added).
                # `[[[]]]]` -> md=3, stack=[] (after last ']') -> True (if `not stack` is added, but this is wrong).
                # The `else` branch for `char == ']'` needs to be careful.
                # If `stack` is empty or `stack[-1]` is not `[`, it means an unmatched closing bracket.
                # This should invalidate the *entire* sequence for being "nested" in a well-formed sense.
                # The original code's `current_depth = 0; stack = []` on an unmatched ']'
                # effectively resets the counter for *subsequent* valid nesting, but it doesn't
                # invalidate the *overall* result if `max_depth` was already reached.

                # Let's consider the definition of "nested".
                # "A string is nested if it contains at least two levels of brackets."
                # This usually implies well-formedness.
                # Example: `[[[]]]` -> True (depth 3)
                # Example: `[]` -> False (depth 1)
                # Example: `[[` -> True (depth 2, even if not closed)
                # Example: `][` -> False (max depth 1, not well-formed)
                # Example: `[[]]` -> True (depth 2)

                # The current code's behavior for `[[` is `max_depth = 2`, `return True`. This seems correct.
                # The current code's behavior for `][` is `max_depth = 1`, `return False`. This seems correct.

                # The only way the current code could fail is if `max_depth >= 2` is not the only condition.
                # What if the string must be *perfectly balanced* AND have depth >= 2?
                # E.g., `[[[]]]` -> True
                # `[[[]` -> False (unbalanced)
                # `[[[]]]]` -> False (unbalanced)

                # If this is the case, we need to add `and not stack` to the return.
                # Let's test this:
                # `[[[]]]` -> md=3, stack=[] -> True and True -> True. Correct.
                # `[[[]` -> md=3, stack=['[','[','['] -> True and False -> False. Correct.
                # `[[[]]]]` -> md=3, stack=[] (after last ']') -> True and True -> True. This is WRONG.
                # The `else` branch for `]` needs to mark the sequence as invalid if an unmatched `]` is found.

                # Let's refine the `else` for `char == ']':`
                # If `stack` is empty or `stack[-1]` is not `[`, it means we have an unmatched closing bracket.
                # In this case, the entire string cannot be considered "nested" in a well-formed sense.
                # We should probably set `max_depth` to 0 or a flag to indicate invalidity.
                # Or, more simply, if we encounter an unmatched `]`, the `current_depth` should not just reset,
                # but the `max_depth` should also be considered invalid for the purpose of the final check.

                # Let's try a different approach for the `else` branch of `char == ']':`
                # If an unmatched `]` is found, it means the structure is invalid.
                # We can set `max_depth` to 0 and break, or return False immediately if we want strict well-formedness.
                # But the problem implies finding *any* nested structure.

                # The original code's `current_depth = 0; stack = []` for an unmatched