def expect_eq(name, expected, actual):
    if expected != actual:
        raise Exception(f"Expected {name} to be {expected}, was {actual}")
