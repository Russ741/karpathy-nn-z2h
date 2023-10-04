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

# %%

def forward_prop(x, W, b):
    one_hot = torch.nn.functional.one_hot(x).double()
    output = torch.matmul(one_hot, W) + b

    softmax = output.exp()
    softmax = softmax / softmax.sum(1, keepdim=True)

    return softmax

# %%

def calculate_loss(y_hat, y):
    match_probabilities = y_hat[torch.arange(len(y)), y]
    neg_log_likelihood = -match_probabilities.log().mean()
    return neg_log_likelihood

# %%

def descend_gradient(W, b, learning_rate):
    W.data -= learning_rate * W.grad
    b.data -= learning_rate * b.grad
    return W, b

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
