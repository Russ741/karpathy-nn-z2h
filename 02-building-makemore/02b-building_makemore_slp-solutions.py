# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false
# ### Preamble: Load data
#
# Objective: Load a list of words from the [names.txt](https://github.com/karpathy/makemore/blob/master/names.txt) file into a list variable named ```words```.

# %%
import requests

def load_words():
    words_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    words = requests.get(words_url).text.splitlines()
    return words

# %% deletable=false editable=false
def test_words():
    if not isinstance(loaded_words, list):
        print(f"Expected words to be a list")
        return
    if (len_words := len(loaded_words)) != (expected_words := 32033):
        print(f"Expected {expected_words} elements in words, found {len_words} elements")
        return
    if (zeroth_word := loaded_words[0]) != (expected_zeroth := "emma"):
        print(f"Expected zeroth word in words to be '{expected_zeroth}', was '{zeroth_word}'")
        return
    if (final_word := loaded_words[-1]) != (expected_final := "zzyzx"):
        print(f"Expected final word in words to be '{expected_final}', was '{final_word}'")
        return
    print("load_words looks good. Onwards!")
loaded_words = load_words()
test_words()

# %%

def generate_bigrams(words):
    bigrams = []
    for word in words:
        bigrams.append(('.', word[0]))
        for pos in range(len(word) - 1):
            bigrams.append((word[pos], word[pos + 1]))
        bigrams.append((word[-1], '.'))
    return bigrams

# %% deletable=false editable=false
def test_bigrams():
    if not isinstance(generated_bigrams, list):
        print(f"Expected bigrams to be a list")
        return
    if (start_m_ct := generated_bigrams.count(('.', 'm'))) != (expected_start_m_ct := 2538):
        print(f"Expected {expected_start_m_ct} ('a', 'b') bigrams, found {start_m_ct}")
        return
    if (ab_ct := generated_bigrams.count(('a', 'b'))) != (expected_ab_ct := 541):
        print(f"Expected {expected_ab_ct} ('a', 'b') bigrams, found {ab_ct}")
        return
    if (s_end_ct := generated_bigrams.count(('s', '.'))) != (expected_s_end_ct := 1169):
        print(f"Expected {expected_s_end_ct} ('s', '.') bigrams, found {s_end_ct}")
        return
    print("generate_bigrams looks good. Onwards!")
generated_bigrams = generate_bigrams(loaded_words)
test_bigrams()

# %%

def get_stoi(bigrams):
    chars = set()
    for bigram in bigrams:
        chars.add(bigram[0])
        chars.add(bigram[1])
    stoi = { v:k for (k, v) in enumerate(sorted(chars))}
    return stoi

# %% deletable=false editable=false
import string

def test_stoi():
    if not isinstance(stoi, dict):
        print(f"Expected stoi to be a dict")
        return
    for c in string.ascii_lowercase:
        if not c in stoi:
            print(f"Expected {c} to be in stoi")
            return
    print("get_stoi looks good. Onwards!")
stoi = get_stoi(generated_bigrams)
test_stoi()

# %%

def get_itos(stoi):
    itos = {stoi[c]:c for c in stoi}
    return itos

# %% deletable=false editable=false
def test_itos():
    if not isinstance(itos, dict):
        print(f"Expected stoi to be a dict")
        return
    for c in string.ascii_lowercase:
        c_i = stoi[c]
        if (expected_c := itos[c_i]) != c:
            print(f"Expected itos[{c_i}] to be {expected_c}, was {c}")
    print("get_itos looks good. Onwards!")
itos = get_itos(stoi)
test_itos()

# %%

import torch

def get_x_and_y(bigrams, stoi):
    x = torch.tensor(list(map(lambda bigram : stoi[bigram[0]], bigrams)))
    y = torch.tensor(list(map(lambda bigram : stoi[bigram[-1]], bigrams)))

    return x, y

# %% deletable=false editable=false
def test_x_and_y():
    if (x0 := x[0]) != (expected_x0 := 0):
        print(f"Expected x[0] to be {expected_x0}, was {x0}")
        return
    if (y0 := y[0]) != (expected_y0 := 5):
        print(f"Expected y[0] to be {expected_y0}, was {y0}")
        return
    if (x_sfe := x[-2]) != (expected_x_sfe := 26):
        print(f"Expected x[-2] to be {expected_x_sfe}, was {x_sfe}")
        return
    if (y_sfe := y[-2]) != (expected_y_sfe := 24):
        print(f"Expected y[-2] to be {expected_y_sfe}, was {y_sfe}")
    print("get_x_and_y looks good. Onwards!")
x, y = get_x_and_y(generated_bigrams, stoi)
test_x_and_y()

# %%

import torch
def initialize_w_b(stoi):
    stoi_n = len(stoi)
    W = torch.rand((stoi_n,stoi_n), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((1,stoi_n),dtype=torch.float64, requires_grad=True)

    return W, b

# %% deletable=false editable=false
def test_initialize_w_b():
    if (w_len := len(W)) != (expected_w_len := 27):
        print(f"Expected W to have {expected_w_len} rows, had {w_len}")
        return
    for row_idx in range(w_len):
        if (row_len := len(W[row_idx])) != (expected_row_len := 27):
            print(f"Expected W[{row_idx}] to have {expected_row_len} columns, had {row_len}")
            return
        for col_idx in range(row_len):
            if (val := W[row_idx][col_idx]) == 0.0:
                print(f"Expected W[{row_idx}][{col_idx}] to be non-zero.")
                return
    if not W.requires_grad:
        print("W must be marked with requires_grad so its grad property will be populated by backpropagation for use in gradient descent.")
        return
    if not b.requires_grad:
        print("b must be marked with requires_grad so its grad property will be populated by backpropagation for use in gradient descent.")
        return
    if (b_shape := b.shape) != (expected_b_shape := (1, 27)):
        print(f"Expected b to have shape {expected_b_shape}, had shape {b_shape}")
        return
    print("initialize_w_b looks good. Onwards!")
W, b = initialize_w_b(stoi)
test_initialize_w_b()

# %%

def forward_prop(x, W, b):
    one_hot = torch.nn.functional.one_hot(x).double()
    output = torch.matmul(one_hot, W) + b

    softmax = output.exp()
    softmax = softmax / softmax.sum(1, keepdim=True)

    return softmax

# %% deletable=false editable=false
def test_forward_prop():
    if (y_hat_len := len(y_hat)) != (expected_y_hat_len := len(y)):
        print(f"Expected y_hat to have {expected_y_hat_len} rows, had {y_hat_len}")
        return
    if (cols := len(y_hat[0])) != (expected_cols := len(stoi)):
        print(f"Expected y_hat to have {expected_cols} columns, had {cols}")
        return
    for row_idx in range(y_hat_len):
        if abs((sum := y_hat[row_idx].sum()) - 1.0) > 0.00001:
            print(f"Expected y_hat[{row_idx}] to sum to 1.0, summed to {sum}")
            return
    print("forward_prop looks good. Onwards!")
y_hat = forward_prop(x, W, b)
test_forward_prop()

# %%

def calculate_loss(y_hat, y):
    match_probabilities = y_hat[torch.arange(len(y)), y]
    neg_log_likelihood = -match_probabilities.log().mean()
    return neg_log_likelihood

# %% deletable=false editable=false
from math import exp

def test_calculate_loss():
    y = torch.tensor([2], dtype=torch.int64)
    y_hat = torch.tensor([
        [0.0, 0.0, 1.0, 0.0]
    ])
    if abs((loss := calculate_loss(y_hat, y))) > 0.00001:
        print(f"Expected loss for first example to be 0.0, was {loss}")
        return

    y = torch.tensor([2, 0], dtype=torch.int64)
    y_hat = torch.tensor([
        [0.09, 0.09, exp(-0.5), 0.09],
        [exp(-0.1), 0.01, 0.02, 0.03]
    ])
    if abs((loss := calculate_loss(y_hat, y)) - (expected_loss := 0.3)) > 0.00001:
        print(f"Expected loss for second example to be {expected_loss}, was {loss}")
        return
    print("calculate_loss looks good. Onwards!")
calculated_loss = calculate_loss(y_hat, y)
test_calculate_loss()

# %%

def descend_gradient(W, b, learning_rate):
    W.data -= learning_rate * W.grad
    b.data -= learning_rate * b.grad
    return W, b

# %% deletable=false editable=false

def test_descend_gradient():
    W = torch.tensor([
        [1.0, 2.0,],
        [3.0, -4.0],
        [-5.0, 6.0],
    ])
    W.grad = torch.tensor([
        [-2.0, 1.0],
        [0.0, -2.0],
        [4.0, 1.0]
    ])
    new_w, new_b = descend_gradient(W, b, 3.0)
    expected_new_w = torch.tensor([
        [7.0, -1.0],
        [3.0, 2.0],
        [-17.0, 3.0]
    ])
    if not new_w.equal(expected_new_w):
        print(f"Expected new W for test case to be \n{expected_new_w}\n, is \n{new_w}")
        return
    print("descend_gradient looks good. Onward!")
test_descend_gradient()

# %%

def main():
    words = load_words()
    bigrams = generate_bigrams(words)
    stoi = get_stoi(bigrams)
    itos = get_itos(stoi)
    x, y = get_x_and_y(bigrams, stoi)
    W, b = initialize_w_b(stoi)
    for i in range(1, 101, 1):
        y_hat = forward_prop(x,W,b)
        loss = calculate_loss(y_hat, y)
        if i % 10 == 0:
            print(f"Round {i} loss: {loss.item()}")
        W.grad = None
        b.grad = None
        loss.backward()
        W, b = descend_gradient(W, b, 10.0)


if __name__ == "__main__":
    main()

# %%
