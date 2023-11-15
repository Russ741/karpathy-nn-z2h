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
#
# # Worksheet 2b - Single-Layer Perceptron
#
# This is the third in a series of companion worksheets for for Andrej Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) videos.
#
# It corresponds to roughly the second half of the second video in the series, named "[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)".
#
# The rest of the worksheets are listed in the README [here](https://github.com/Russ741/karpathy-nn-z2h/).
#
# The overall objective of this worksheet is to write code that generates a word that is similar to a set of example words it is trained on.
# It does so using a single-layer neural network.

# %% [markdown] deletable=false editable=false
# ### Prerequisite: Load worksheet utilities and download word list
#
# The following cell imports [utility functions](https://github.com/Russ741/karpathy-nn-z2h/blob/main/worksheets/worksheet_utils.py) that this worksheet depends on.
# If the file isn't already locally available (e.g. for Colab), it downloads it from GitHub.
#
# Similarly, if this directory does not already contain names.txt, it downloads it from
# [the makemore GitHub repository](https://github.com/karpathy/makemore/blob/master/names.txt).

# %% deletable=false editable=false
import os
import urllib
import shutil

try:
    from worksheet_utils import *
    print("worksheet_utils found.")
except ModuleNotFoundError:
    utils_local_filename = "worksheet_utils.py"
    print(f"Downloading worksheet_utils.")
    with urllib.request.urlopen("https://raw.githubusercontent.com/Russ741/karpathy-nn-z2h/main/worksheets/worksheet_utils.py") as response:
        with open(utils_local_filename, mode="xb") as utils_file:
            shutil.copyfileobj(response, utils_file)
    from worksheet_utils import *

WORDS_PATH = "names.txt"
if os.path.isfile(WORDS_PATH):
    print("word file found.")
else:
    print("word file not found, downloading.")
    with urllib.request.urlopen("https://github.com/karpathy/makemore/raw/master/names.txt") as response:
        with open(WORDS_PATH, mode="xb") as words_file:
            shutil.copyfileobj(response, words_file)

# %% [markdown] deletable=false editable=false
# ### Preamble: Load data
#
# Objective: Write a function that:
#  * Returns a list of strings
#    * Each string should be equal to the word from the corresponding line of the word file (at ```WORDS_PATH```)
#    * The strings should not include line-break characters
#
# Note: In practice, the order of the strings in the returned list does not matter, but for the
# test to pass, they should be in the same order in the list as in the word file.
#
# Video: [0:03:03](https://youtu.be/PaCmpygFfXo?t=183)

# %%
def load_words():
# Solution code
    words = open(WORDS_PATH).read().splitlines()
    return words
# End solution code

# %% deletable=false editable=false
def test_words():
    expect_type("loaded_words", loaded_words, list)
    expect_eq("len(loaded_words)", len(loaded_words), 32033)
    expect_eq("loaded_words[0]", loaded_words[0], "emma")
    expect_eq("loaded_words[-1]", loaded_words[-1], "zzyzx")
    print("load_words looks good. Onwards!")
loaded_words = load_words()
test_words()

# %% [markdown] deletable=false editable=false
# ### Step 1: Generate bigrams
#
# Objective: Populate the variable ```bigrams``` with a list of bigrams (2-element tuples) of adjacent characters in ```words```.
#
# Treat the start and end of each word as the character '.'
#
# Video: [0:06:24](https://youtu.be/PaCmpygFfXo?t=384) and [0:21:55](https://youtu.be/PaCmpygFfXo?t=1315)

# %%
def generate_bigrams(words):
# Solution code
    bigrams = []
    for word in words:
        bigrams.append(('.', word[0]))
        for pos in range(len(word) - 1):
            bigrams.append((word[pos], word[pos + 1]))
        bigrams.append((word[-1], '.'))
    return bigrams
# End solution code

# %% deletable=false editable=false
def test_generate_bigrams():
    bigrams = generate_bigrams(loaded_words)
    expect_type("bigrams", bigrams, list)
    expect_eq("count of ('.', 'm') bigrams", bigrams.count(('.', 'm')), 2538)
    expect_eq("count of ('a', 'b') bigrams", bigrams.count(('a', 'b')), 541)
    expect_eq("count of ('s', '.') bigrams", bigrams.count(('s', '.')), 1169)
    print("generate_bigrams looks good. Onwards!")
test_generate_bigrams()

# %% [markdown] deletable=false editable=false
# ### Step 2: Map characters to indices
#
# Objective: Write a function that takes the following arguments:
# * a list of char, char tuples representing all of the bigrams in a word list
#
# And returns:
# * a dict (```stoi```) where
#   * the key is a character from ```words``` (including '.' for start/end),
#   * the value is a unique integer, and
#   * all the values are in the range from 0 to ```len(stoi) - 1``` (no gaps)
#
# We'll use these unique integers as an index to represent the characters in a Tensor in later steps
#
# Note that for this list of words, the same value of ```stoi``` could be generated without looking at the words at all,
# but simply by using all the lowercase letters and a period. This approach would be more efficient for this exercise,
# but will not generalize well conceptually to more complex models in future exercises.
#
# Video: [0:15:40](https://youtu.be/PaCmpygFfXo?t=940)

# %%
def get_stoi(bigrams):
# Solution code
    chars = set()
    for bigram in bigrams:
        chars.add(bigram[0])
        chars.add(bigram[1])
    stoi = { v:k for (k, v) in enumerate(sorted(chars))}
    return stoi
# End solution code

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
    expect_type("stoi", stoi, dict)
    s = sorted(stoi.keys())
    expect_eq("stoi keys when sorted", s, expected_s)
    expected_i = list(range(len(s)))
    i = sorted(stoi.values())
    expect_eq("stoi values when sorted", i, expected_i)
    print("get_stoi looks good. Onwards!")
test_get_stoi()

# %% [markdown] deletable=false editable=false
# ### Step 3: Map indices to characters
#
# Objective: Write a function that takes the following arguments:
# * a dict (```stoi```) as defined in step 2
#
# And returns:
# * a dict (```itos```) where ```itos``` contains the same key-value pairs as ```stoi``` but with keys and values swapped.
#
# E.g. if ```stoi == {'.' : 0, 'b' : 1, 'z', 2}```, then ```itos == {0 : '.', 1 : 'b', 2 : 'z'}```
#
# Video: [0:18:49](https://youtu.be/PaCmpygFfXo?t=1129)

# %%
def get_itos(stoi):
# Solution code
    itos = {stoi[c]:c for c in stoi}
    return itos
# End solution code

# %% deletable=false editable=false
import string

def test_get_itos():
    stoi = {elem:idx for idx, elem in enumerate(string.ascii_lowercase + ".")}
    itos = get_itos(stoi)
    expect_type("itos", itos, dict)
    for c in string.ascii_lowercase + ".":
        c_i = stoi[c]
        expect_eq(f"itos.get({c_i})", itos.get(c_i), c)
    print("get_itos looks good. Onwards!")
test_get_itos()

# %% [markdown] deletable=false editable=false
# ### Step 4: Split bigrams into inputs and outputs
#
# Objective: Write a function that takes the following arguments:
# * a list ```bigrams``` as defined in step 1, and
# * a dict ```stoi``` as defined in step 2
#
# And returns:
# * a one-dimensional torch.Tensor ```x``` with all of the first characters in the tuples in ```bigrams```
# * a one-dimensional torch.Tensor ```y``` with all of the second characters in the tuples in ```bigrams```
# * Note: Both output tensors should be the same length as ```bigrams```
#
# Video: [1:05:25](https://youtu.be/PaCmpygFfXo?t=3925)

# %%
import torch

def get_x_and_y(bigrams, stoi):
# Solution code
    x = torch.tensor(list(map(lambda bigram : stoi[bigram[0]], bigrams)))
    y = torch.tensor(list(map(lambda bigram : stoi[bigram[-1]], bigrams)))

    return x, y
# End solution code

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
    expect_eq("x[0]", x[0], 0)
    expect_eq("y[0]", y[0], 1)
    expect_eq("x[-2]", x[-2], 4)
    expect_eq("y[-2]", y[-2], 5)
    print("get_x_and_y looks good. Onwards!")
test_get_x_and_y()

# %% [markdown] deletable=false editable=false
# ### Step 5: Provide initial values for the model parameters
#
# Objective: Write a function that takes the following arguments:
# * a dict ```stoi``` as defined in step 2
#   * the length of ```stoi``` will be referred to as ```stoi_n```
# * a ```torch.Generator``` (```gen```) to provide (pseudo)random initial values for the parameters
#
# And returns:
# * a pytorch.Tensor ```W``` of shape (```stoi_n```, ```stoi_n```) where each element is randomly generated
# * a pytorch.Tensor ```b``` of shape (1, ```stoi_n```)
#   * The elements of ```b``` can be zero
#
# Video: [1:14:03](https://youtu.be/PaCmpygFfXo?t=4433)

# %%
import torch

def initialize_w_b(stoi, gen):
# Solution code
    stoi_n = len(stoi)
    W = torch.rand((stoi_n,stoi_n), generator=gen, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((1,stoi_n), dtype=torch.float64, requires_grad=True)

    return W, b
# End solution code

# %% deletable=false editable=false
def test_initialize_w_b():
    stoi = {'q': 0, 'w': 1, 'e': 2, 'r': 3}
    expected_s_ct = 4
    gen = torch.Generator()
    gen.manual_seed(12345)
    W, b = initialize_w_b(stoi, gen)
    expect_eq("len(W)", len(W), expected_s_ct)
    for row_idx in range(len(W)):
        expect_eq(f"len(W[{row_idx}])", len(W[row_idx]), expected_s_ct)
        for col_idx in range(len(W[row_idx])):
            if (val := W[row_idx][col_idx]) == 0.0:
                raise Exception(f"Expected W[{row_idx}][{col_idx}] to be non-zero.")
    expect_eq("W.requires_grad", W.requires_grad, True)
    expect_eq("b.requires_grad", b.requires_grad, True)
    expect_eq("b.shape", b.shape, (1, expected_s_ct))
    print("initialize_w_b looks good. Onwards!")
test_initialize_w_b()

# %% [markdown] deletable=false editable=false
# ### Step 6: Forward propagation
#
# Objective: Write a function that takes the following arguments:
# * a pytorch.Tensor ```x``` of training or testing inputs
# * pytorch.Tensors ```W``` and ```b``` representing the parameters of the model
#
# And returns:
# * a pytorch.Tensor ```y_hat``` of the model's predicted outputs for each input in x
#   * The predicted outputs for a given sample should sum to 1.0
#   * The shape of ```y_hat``` should be (```len(x)```, ```len(W)```)
#     * Note that ```len(W)``` represents the number of different characters in the word list
#
# Video: [1:15:12](https://youtu.be/PaCmpygFfXo?t=4512)

# %%
def forward_prop(x, W, b):
# Solution code
    one_hot = torch.nn.functional.one_hot(x, len(W)).double()
    output = torch.matmul(one_hot, W) + b

    softmax = output.exp()
    softmax = softmax / softmax.sum(1, keepdim=True)

    return softmax
# End solution code

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

    expect_tensor_close("y_hat for test case", y_hat, expected_y_hat)
    print("forward_prop looks good. Onwards!")
test_forward_prop()

# %% [markdown] deletable=false editable=false
# ### Step 7: Loss calculation
# Objective: Write a function that takes the following arguments:
# * a pytorch.Tensor ```y_hat``` of predicted outputs for a particular set of inputs
# * a pytorch.Tensor ```y``` of actual outputs for the same set of inputs
#
# And returns:
# * a floating-point value representing the model's negative log likelihood loss for that set of inputs
#
# Video: [1:35:49](https://youtu.be/PaCmpygFfXo&t=5749)
# %%
def calculate_loss(y_hat, y):
# Solution code
    match_probabilities = y_hat[torch.arange(len(y)), y]
    neg_log_likelihood = -match_probabilities.log().mean()
    return neg_log_likelihood
# End solution code

# %% deletable=false editable=false
from math import exp

def test_calculate_loss():
    y = torch.tensor([2], dtype=torch.int64)
    y_hat = torch.tensor([
        [0.0, 0.0, 1.0, 0.0]
    ])
    expect_close("loss for first example", calculate_loss(y_hat, y), 0.0)

    y = torch.tensor([2, 0], dtype=torch.int64)
    y_hat = torch.tensor([
        [0.09, 0.09, exp(-0.5), 0.09],
        [exp(-0.1), 0.01, 0.02, 0.03]
    ])
    expect_close("loss for second example", calculate_loss(y_hat, y), 0.3)
    print("calculate_loss looks good. Onwards!")
test_calculate_loss()

# %% [markdown] deletable=false editable=false
# ### Step 8: Gradient descent
# Objective: Write a function that takes the following arguments:
# * pytorch.Tensors ```W``` and ```b``` representing the parameters of the model
# * a floating-point value ```learning_rate``` representing the overall size of adjustment to make to the parameters
#
# And returns:
# * the updated pytorch.Tensors ```W``` and ```b```
#   * Note: Updating the parameters in-place is desirable, but for ease of testing, please return them regardless.
#
# Video: [1:41:26](https://youtu.be/PaCmpygFfXo?t=6086)

# %%
def descend_gradient(W, b, learning_rate):
# Solution code
    W.data -= learning_rate * W.grad
    b.data -= learning_rate * b.grad
    return W, b
# End solution code

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
    expect_tensor_close("new W for test case", new_w, expected_new_w)

    expected_new_b = torch.tensor([
        4.0,
        0.5,
    ])
    expect_tensor_close("new b for test case", new_b, expected_new_b)
    print("descend_gradient looks good. Onward!")
test_descend_gradient()

# %% [markdown] deletable=false editable=false
# ### Step 9: Train model
# Objective: Write a function that takes the following arguments:
# * pytorch.Tensors ```x``` and ```y``` as described in Step 4
# * pytorch.Tensors ```W``` and ```b``` as described in Step 5
# * a floating-point value ```learning_rate``` representing the overall size of adjustment to make to the parameters
#
# Updates the values of W and b to fit the data slightly better
#
# And returns:
# * the loss as defined in Step 6
#
# Implementation note: this function should make use of several of the functions you've previously implemented.
#
# Video: [1:42:55](https://youtu.be/PaCmpygFfXo?t=6175)

# %%
def train_model(x, y, W, b, learning_rate):
# Solution code
    y_hat = forward_prop(x,W,b)
    loss = calculate_loss(y_hat, y)
    W.grad = None
    b.grad = None
    loss.backward()
    W, b = descend_gradient(W, b, learning_rate)
    return loss.item()
# End solution code

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
    expect_tensor_close("W for test case", W, expected_W)

    expected_b = torch.tensor([
        0.1654,
        0.2022,
        0.2323
    ], dtype=torch.float64)
    expect_tensor_close("b for test case", b, expected_b)
    print("train_model looks good. Onward!")
test_train_model()

# %% [markdown] deletable=false editable=false
# ### Step 10: Generate words
# Objective: Write a function that takes the following arguments:
# * pytorch.Tensors ```W``` and ```b``` as described in Step 5
# * a dict ```stoi``` as described in Step 2
# * a dict ```itos``` as described in Step 3
# * a torch.Generator to use for pseudorandom selection of elements
#
# Repeatedly generates a probability distribution for the next letter to select given the last letter
#
# And returns
# * a string representing a word generated by repeatedly sampling the probability distribution
#
# Video: [1:54:31](https://youtu.be/PaCmpygFfXo?t=6871)

# %%
def generate_word(W, b, stoi, itos, gen):
# Solution code
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
# End solution code

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
    # These weights result in a probability distribution where the desired next letter is roughly
    # 1000x as likely as the others.
    for i in range(stoi_n - 1):
        W[i][i+1] = 10.0
    W[stoi_n - 1][0] = 10.0

    gen = torch.Generator()
    gen.manual_seed(2147476727)
    expect_eq("generate_word result for test case", generate_word(W, b, stoi, itos, gen), "onward")
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
# Solution code
def train_model_and_generate_words():
    bigrams = generate_bigrams(loaded_words)
    stoi = get_stoi(bigrams)
    itos = get_itos(stoi)
    x, y = get_x_and_y(bigrams, stoi)
    gen = torch.Generator()
    W, b = initialize_w_b(stoi, gen)
    for i in range(1, 101, 1):
        loss = train_model(x, y, W, b, 10.0)
    print(f"Final loss is {loss}")
    for i in range(10):
        print(generate_word(W, b, stoi, itos, gen))
train_model_and_generate_words()
# End solution code

# %%
