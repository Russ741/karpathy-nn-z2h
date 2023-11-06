def expect_eq(name, actual, expected):
    if expected != actual:
        raise Exception(f"Expected {name} to be {expected}, was {actual}")
