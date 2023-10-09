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
def test_generate_bigrams():
    bigrams = generate_bigrams(loaded_words)
    if not isinstance(bigrams, list):
        print(f"Expected bigrams to be a list")
        return
    if (start_m_ct := bigrams.count(('.', 'm'))) != (expected_start_m_ct := 2538):
        print(f"Expected {expected_start_m_ct} ('a', 'b') bigrams, found {start_m_ct}")
        return
    if (ab_ct := bigrams.count(('a', 'b'))) != (expected_ab_ct := 541):
        print(f"Expected {expected_ab_ct} ('a', 'b') bigrams, found {ab_ct}")
        return
    if (s_end_ct := bigrams.count(('s', '.'))) != (expected_s_end_ct := 1169):
        print(f"Expected {expected_s_end_ct} ('s', '.') bigrams, found {s_end_ct}")
        return
    print("generate_bigrams looks good. Onwards!")
test_generate_bigrams()

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

def test_get_stoi():
    bigrams = [
        ('.', 'h'),
        ('h', 'i'),
        ('i', '.'),
        ('.', 'b'),
        ('b', 'y'),
        ('y', 'e'),
        ('e', '.'),
    ]
    expected_s = sorted(['.', 'h', 'i', 'b', 'y', 'e'])
    stoi = get_stoi(bigrams)
    if not isinstance(stoi, dict):
        print(f"Expected stoi to be a dict")
        return
    s = sorted(stoi.keys())
    if s != expected_s:
        print(f"Expected stoi keys to be {expected_s} when sorted, were {s}")
        return
    expected_i = list(range(len(s)))
    i = sorted(stoi.values())
    if i != expected_i:
        print(f"Expected stoi values to be {expected_i} when sorted, were {i}")
        return
    print("get_stoi looks good. Onwards!")
test_get_stoi()

# %%
def get_itos(stoi):
    itos = {stoi[c]:c for c in stoi}
    return itos

# %% deletable=false editable=false
import string

def test_get_itos():
    stoi = {elem:idx for idx, elem in enumerate(string.ascii_lowercase + ".")}
    itos = get_itos(stoi)
    if not isinstance(itos, dict):
        print(f"Expected stoi to be a dict")
        return
    for c in string.ascii_lowercase + ".":
        c_i = stoi[c]
        if (expected_c := itos[c_i]) != c:
            print(f"Expected itos[{c_i}] to be {expected_c}, was {c}")
    print("get_itos looks good. Onwards!")
test_get_itos()

# %%
import torch

def get_x_and_y(bigrams, stoi):
    x = torch.tensor(list(map(lambda bigram : stoi[bigram[0]], bigrams)))
    y = torch.tensor(list(map(lambda bigram : stoi[bigram[-1]], bigrams)))

    return x, y

# %% deletable=false editable=false
def test_get_x_and_y():
    bigrams = [
        ('.', 'h'),
        ('h', 'i'),
        ('i', '.'),
        ('.', 'b'),
        ('b', 'y'),
        ('y', 'e'),
        ('e', '.'),
    ]
    stoi = {
        '.': 0,
        'h': 1,
        'i': 2,
        'b': 3,
        'y': 4,
        'e': 5,
    }
    x, y = get_x_and_y(bigrams, stoi)
    if (x0 := x[0]) != (expected_x0 := 0):
        print(f"Expected x[0] to be {expected_x0}, was {x0}")
        return
    if (y0 := y[0]) != (expected_y0 := 1):
        print(f"Expected y[0] to be {expected_y0}, was {y0}")
        return
    if (x_sfe := x[-2]) != (expected_x_sfe := 4):
        print(f"Expected x[-2] to be {expected_x_sfe}, was {x_sfe}")
        return
    if (y_sfe := y[-2]) != (expected_y_sfe := 5):
        print(f"Expected y[-2] to be {expected_y_sfe}, was {y_sfe}")
    print("get_x_and_y looks good. Onwards!")
test_get_x_and_y()

# %%
import torch
def initialize_w_b(stoi):
    stoi_n = len(stoi)
    W = torch.rand((stoi_n,stoi_n), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((1,stoi_n),dtype=torch.float64, requires_grad=True)

    return W, b

# %% deletable=false editable=false
def test_initialize_w_b():
    stoi = {'q': 0, 'w': 1, 'e': 2, 'r': 3}
    expected_s_ct = 4
    W, b = initialize_w_b(stoi)
    if (w_len := len(W)) != expected_s_ct:
        print(f"Expected W to have {expected_s_ct} rows, had {w_len}")
        return
    for row_idx in range(w_len):
        if (row_len := len(W[row_idx])) != expected_s_ct:
            print(f"Expected W[{row_idx}] to have {expected_s_ct} columns, had {row_len}")
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
    if (b_shape := b.shape) != (expected_b_shape := (1, expected_s_ct)):
        print(f"Expected b to have shape {expected_b_shape}, had shape {b_shape}")
        return
    print("initialize_w_b looks good. Onwards!")
test_initialize_w_b()

# %%
def forward_prop(x, W, b):
    one_hot = torch.nn.functional.one_hot(x, len(W)).double()
    output = torch.matmul(one_hot, W) + b

    softmax = output.exp()
    softmax = softmax / softmax.sum(1, keepdim=True)

    return softmax

# %% deletable=false editable=false
def test_forward_prop():
    x = torch.tensor([
        1,
        0,
    ])

    W = torch.tensor([
        [0.1, 0.9, 0.2, 0.01],
        [0.04, 0.2, 1.6, 0.25],
        [0.02, 0.03, 0.7, 0.01],
    ], dtype=torch.float64)

    b = torch.tensor([
        0.01, 0.02, 0.03, 0.04
    ], dtype=torch.float64)

    expected_y_hat = torch.tensor([
        [0.1203, 0.1426, 0.5841, 0.1530],
        [0.1881, 0.4228, 0.2120, 0.1771],
    ], dtype=torch.float64)

    y_hat = forward_prop(x, W, b)

    if not torch.isclose(expected_y_hat, y_hat, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected y_hat for test case to be \n{expected_y_hat}\n, was \n{y_hat}")
        return
    print("forward_prop looks good. Onwards!")
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
    b = torch.tensor([
        1.0,
        2.0,
    ])
    b.grad = torch.tensor([
        -1.0,
        0.5,
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
    expected_new_b = torch.tensor([
        4.0,
        0.5,
    ])
    if not new_b.equal(expected_new_b):
        print(f"Expected new b for test case to be \n{expected_new_b}\n, is \n{new_b}")
        return
    print("descend_gradient looks good. Onward!")
test_descend_gradient()

# %%
def train_model(x, y, W, b, learning_rate):
    y_hat = forward_prop(x,W,b)
    loss = calculate_loss(y_hat, y)
    W.grad = None
    b.grad = None
    loss.backward()
    W, b = descend_gradient(W, b, learning_rate)
    return loss.item()

# %% deletable=false editable=false
def test_train_model():
    x = torch.tensor([
        0,
        1,
        2,
    ])
    y = torch.tensor([
        1,
        2,
        0,
    ])
    W = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([
        0.1,
        0.2,
        0.3,
    ], dtype=torch.float64, requires_grad=True)

    loss = train_model(x, y, W, b, 2.0)

    expected_W = torch.tensor([
        [0.7996, 1.4452, 0.7552],
        [0.7996, 0.7785, 1.4219],
        [1.4663, 0.7785, 0.7552]
    ], dtype=torch.float64)
    if not torch.isclose(expected_W, W, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected W for test case to be \n{expected_W}\n, was \n{W}")
        return

    expected_b = torch.tensor([
        0.1654,
        0.2022,
        0.2323
    ], dtype=torch.float64)
    if not torch.isclose(expected_b, b, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected b for test case to be \n{expected_b}\n, was \n{b}")
        return
    print("train_model looks good. Onward!")
test_train_model()

# %%

# Write code that takes:
#   a model (W and b)
#   stoi and itos
#
# And returns
#  a string (the word generated by the model)

def generate_word(W, b, stoi, itos, gen):
    chr = '.'
    word = ""
    while True:
        x = torch.tensor([stoi[chr]])
        probability_distribution = forward_prop(x, W, b)
        sample = torch.multinomial(probability_distribution, 1, generator=gen).item()
        chr = itos[sample]
        if chr == '.':
            break
        word += chr
    return word

# %% deletable=false editable=false
def test_generate_word():
    stoi = {
        '.': 0,
        'o': 1,
        'n': 2,
        'w': 3,
        'a': 4,
        'r': 5,
        'd': 6,
    }
    stoi_n = len(stoi)
    itos = {v:k for k,v in stoi.items()}

    W = torch.zeros((stoi_n, stoi_n), dtype=torch.float64)
    b = torch.zeros((1, stoi_n), dtype=torch.float64)
    for i in range(stoi_n - 1):
        W[i][i+1] = 1.0
    W[stoi_n - 1][0] = 1.0

    gen = torch.Generator()
    gen.manual_seed(2147476727)
    if (word := generate_word(W, b, stoi, itos, gen)) != (expected_word := "onward"):
        print(f"Expected word for test case to be {expected_word}, was {word}")
        return
    print(f"generate_word looks good. Onward!")
test_generate_word()

# %% [markdown] deletable=false editable=false
# ### Finale: Put it all together
#
# Objective: Write (and call) a function that:
# * generates the bigrams and character maps
# * repeatedly trains the model until its loss is acceptably small
#   * For reference, the "perfect" loss of the probability table approach is approximately 2.4241
# * uses the model to generate some made-up names

# %%
def train_model_and_generate_words():
    bigrams = generate_bigrams(loaded_words)
    stoi = get_stoi(bigrams)
    itos = get_itos(stoi)
    x, y = get_x_and_y(bigrams, stoi)
    W, b = initialize_w_b(stoi)
    for i in range(1, 101, 1):
        loss = train_model(x, y, W, b, 10.0)
    print(f"Final loss is {loss}")
    gen = torch.Generator()
    for i in range(10):
        print(generate_word(W, b, stoi, itos, gen))
train_model_and_generate_words()

# %%
