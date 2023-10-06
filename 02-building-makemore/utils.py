import torch

def expect_close(var_name, expected, actual):
    if not torch.isclose(expected, actual, rtol = 0.0, atol = 0.0001).all():
        err = f"Expected {var_name} for test case to be \n{expected},\n is \n{actual}"
        raise RuntimeError(err)