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
# # Worksheet 3 - Multi-Layer Perceptron
#
# This is the fourth in a series of companion worksheets for for Andrej Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) videos.
#
# It corresponds to the third video in the series, named "[Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)".
#
# The rest of the worksheets are listed in the README [here](https://github.com/Russ741/karpathy-nn-z2h/).
#
# The overall objective of this worksheet is to write code that generates a word that is similar to a set of example words it is trained on.
# It does so using a multi-layer neural network.

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
# Video: [0:09:10](https://youtu.be/TCH_1BHY58I?t=550)

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
# ### Step 1: Map characters to indices
#
# Write a function that takes the following arguments:
# * ```words``` (list of strings)
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
# Video: [0:09:22](https://youtu.be/TCH_1BHY58I?t=562)

# %%
def get_stoi(words):
# Solution code
    chars = set()
    for word in words:
        for char in word:
            chars.add(char)
    chars.add('.')
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

    stoi = get_stoi(bigrams)

    expect_type("stoi", stoi, dict)
    s = sorted(stoi.keys())
    expected_s = sorted(['.', 'h', 'i', 'b', 'y', 'e'])
    expect_eq("stoi keys when sorted", s, expected_s)
    i = sorted(stoi.values())
    expected_i = list(range(len(s)))
    expect_eq("stoi values when sorted", i, expected_i)
    print("get_stoi looks good. Onwards!")
test_get_stoi()

# %% [markdown] deletable=false editable=false
# ### Step 2: Map indices to characters
#
# Objective: Write a function that takes the following arguments:
# * a dict (```stoi```) as defined in step 2
#
# And returns:
# * a dict (```itos```) where ```itos``` contains the same key-value pairs as ```stoi``` but with keys and values swapped.
#
# E.g. if ```stoi == {'.' : 0, 'b' : 1, 'z', 2}```, then ```itos == {0 : '.', 1 : 'b', 2 : 'z'}```
#
# Video: [0:09:22](https://youtu.be/TCH_1BHY58I?t=562)

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
        expect_eq(f"itos[{c_i}]", itos[c_i], c)
    print("get_itos looks good. Onwards!")
test_get_itos()


# %% [markdown] deletable=false editable=false
# ### Step 3: Generate inputs ```X``` and outputs ```Y```
#
# Write a function that takes the following arguments:
# * a list of strings (```words``` from the preamble)
# * a dict of characters to integers (```stoi``` from step 2)
# * an integer (```block_size```) that specifies how many characters to take into account when predicting the next one
#
# And returns:
# * a two-dimensional torch.Tensor ```X``` with each sequence of characters of length block_size from the words in ```words```
# * a one-dimensional torch.Tensor ```Y``` with the character that follows each sequence in ```x```
#
# Video: [0:09:35](https://youtu.be/TCH_1BHY58I?t=575)

# %%
import torch

def get_X_and_Y(words, stoi, block_size):
    X = []
    Y = []
# Solution code
    for word in words:
        word = '.' * block_size + word + '.'
        for idx in range(len(word) - block_size):
            end = idx + block_size
            chars = word[idx : end]
            X.append([stoi[i] for i in chars])
            Y.append(stoi[word[end]])
    return torch.tensor(X), torch.tensor(Y)
# End solution code

# %% deletable=false editable=false
def test_get_X_and_Y():
    words = [
        "hi",
        "bye",
    ]
    stoi = {
        '.': 0,
        'h': 1,
        'i': 2,
        'b': 3,
        'y': 4,
        'e': 5,
    }
    block_size = 3

    (X, Y) = get_X_and_Y(words, stoi, block_size)

    expected_X = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 3],
        [0, 3, 4],
        [3, 4, 5],
    ])
    expected_Y = torch.tensor([
        1,
        2,
        0,
        3,
        4,
        5,
        0,
    ])
    expect_tensor_close("X for test case", X, expected_X)
    expect_tensor_close("Y for test case", Y, expected_Y)
    print("get_x_and_y looks good. Onwards!")
test_get_X_and_Y()

# %% [markdown] deletable=false editable=false
# ### Step 4: Initialize vector embedding lookup table ```C```
#
# Write a function that takes the following arguments:
# * An integer (```indices```) representing the number of indices in ```stoi``` to embed
# * An integer (```embed_dimensions```) representing the number of dimensions the embedded vectors will have
# * A ```torch.Generator``` (```gen```) to provide (pseudo)random initial values for the parameters
#
# And returns:
# * a ```torch.Tensor``` of ```float64``` (```C```) representing the random initial vector for each index.
#
# Video: [0:12:19](https://youtu.be/TCH_1BHY58I?t=739)

# %%
import torch

def get_C(indices, embed_dimensions, gen):
# Solution code
    return torch.rand((indices, embed_dimensions), generator=gen, dtype=torch.float64, requires_grad=True)
# End solution code

# %% deletable=false editable=false
def test_get_C():
    indices = 7
    embed_dimensions = 4
    gen = torch.Generator()
    gen.manual_seed(12345)
    C = get_C(indices, embed_dimensions, gen)
    expect_type("C", C, torch.Tensor)
    expect_eq("C.dtype", C.dtype, torch.float64)
    expect_eq("C.shape", C.shape, (indices, embed_dimensions))
    for i in range(len(C)):
        for j in range(len(C)):
            if i == j:
                continue
            if C[i].equal(C[j]):
                raise Exception(f"Rows {i} and {j} of C are too similar.\n{C[i]=}\n{C[j]=}")
    print("get_C looks good. Onwards!")
test_get_C()

# %% [markdown] deletable=false editable=false
# ### Step 5: Generate vector embeddings of X
#
# Write a function that takes the following arguments:
# * a two-dimensional torch.Tensor ```X``` as defined in step 3
# * a two-dimensional torch.Tensor ```C``` as defined in step 4
#
# And returns:
# * a **two**-dimensional torch.Tensor ```emb``` where each row is the concatenated vector embeddings of the indices of the corresponding row in X
#   * Note the slight difference from the video, where emb is *three*-dimensional
#
# Note that the vector embeddings in a row in C theoretically do not need to match the order of the indices in the row in X;
# they only need to be consistent with the other rows in C.
# For this worksheet, though, if the order does differ, the test case will fail.
#
# Video: [0:13:07](https://youtu.be/TCH_1BHY58I?t=787) and [0:19:10](https://youtu.be/TCH_1BHY58I?t=1150)

# %%
def get_emb(X, C):
# Solution code
    return C[X].reshape(len(X), -1)
# End solution code

# %% deletable=false editable=false
def test_get_vector_embedding():
    X = torch.tensor([
        [1, 2],
        [2, 1],
        [0, 1],
    ])
    ZERO = [0.1, 0.2, 0.3]
    ONE = [0.4, 0.5, 0.6]
    TWO = [0.7, 0.8, 0.9]
    C = torch.tensor([
        ZERO,
        ONE,
        TWO,
    ])

    emb = get_emb(X, C)

    expected_emb = torch.tensor([
        ONE + TWO,
        TWO + ONE,
        ZERO + ONE,
    ])
    expect_tensor_close("emb", emb, expected_emb)
    print("get_vector_embedding looks good. Onwards!")
test_get_vector_embedding()

# %% [markdown] deletable=false editable=false
# ### Step 6: Initialize weight and bias coefficients
#
# Write a function that takes the following arguments:
# * the number of inputs (```input_ct```) to each neuron in the current layer
#   * For the hidden layer, this is equal to the number of cells in each row of emb
#   * For the output layer, this is equal to the number of neurons in the previous (hidden) layer
# * the number of neurons (```neuron_ct```) to include in the current layer
#   * Karpathy chooses to have 100 neurons for the hidden layer
#   * The output layer should have one neuron for each possible result
# * A ```torch.Generator``` (```gen```) to provide (pseudo)random initial values for the parameters
#
# And returns:
# * a two-dimensional ```torch.Tensor``` ```W``` of shape (```input_ct```, ```neuron_ct```) of type ```torch.float64```
#   * each element of ```W``` should be randomly generated
# * a one-dimensional pytorch.Tensor ```b``` of length ```neuron_ct```
#   * the elements of ```b``` can be zero
#
# Video: [0:29:17](https://youtu.be/TCH_1BHY58I?t=1757)

# %%
import torch

def initialize_W_b(input_ct, neuron_ct, gen):
# Solution code
    W = torch.rand((input_ct, neuron_ct), generator=gen, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(neuron_ct, dtype=torch.float64, requires_grad=True)

    return W, b
# End solution code

# %% deletable=false editable=false
def test_initialize_W_b():
    input_ct = 3
    neuron_ct = 5
    gen = torch.Generator()
    gen.manual_seed(12345)
    W, b = initialize_W_b(input_ct, neuron_ct, gen)
    expect_type("W", W, torch.Tensor)
    expect_type("b", b, torch.Tensor)
    expect_eq("W.dtype", W.dtype, torch.float64)
    expect_eq("b.dtype", b.dtype, torch.float64)
    expect_eq("W.shape", W.shape, (input_ct, neuron_ct))
    # The comma is required to make expected_b_shape into a single-element tuple
    expect_eq("b.shape", b.shape, (neuron_ct,))
    print("W and b look good. Onwards!")
test_initialize_W_b()

# %% [markdown] deletable=false editable=false
# ### Step 7: Forward propagate through hidden layer
#
# Write a function that takes the following arguments:
# * a two-dimensional ```torch.Tensor``` ```emb``` as defined in step 5
#   * This is the input to the hidden layer
# * a two-dimensional ```torch.Tensor``` ```W1``` as defined in step 6
#   * This is the hidden layer's weights
# * a one-dimensional ```torch.Tensor``` ```b1``` as defined in step 6
#   * This is the hidden layer's biases
#
# And returns:
# * a one-dimensional ```torch.Tensor``` ```h```
#   * This is the output of the hidden layer after applying a tanh activation function
#
# Video: [0:19:14](https://youtu.be/TCH_1BHY58I?t=1155) and [0:27:57](https://youtu.be/TCH_1BHY58I?t=1677)

# %%
def get_h(emb, W, b):
# Solution code
    return torch.tanh(emb @ W + b)
# End solution code

# %% deletable=false editable=false
def test_get_h():
    emb = torch.tensor([
        [0.1, 0.2],
        [-.3, 0.4],
        [.05, -.06],
    ], dtype=torch.float64)
    W = torch.tensor([
        [0.7, 0.8, -0.9, -0.1],
        [0.6, 0.5, 0.4, 0.3],
    ], dtype=torch.float64)
    b = torch.tensor([
        .09, -.01, .011, -.012
    ], dtype=torch.float64)

    h = get_h(emb, W, b)

    expected_h = torch.tensor([
        [ 2.7291e-01,  1.6838e-01,  1.0000e-03,  3.7982e-02],
        [ 1.1943e-01, -4.9958e-02,  4.1447e-01,  1.3713e-01],
        [ 8.8766e-02,  8.6736e-18, -5.7935e-02, -3.4986e-02],
    ], dtype=torch.float64)
    expect_tensor_close("h for test case", h, expected_h)
    print("get_h looks good. Onwards!")
test_get_h()

# %% [markdown] deletable=false editable=false
# ### Step 8: Calculate output layer outputs before activation
#
# Write a function that takes the following arguments:
# * a two-dimensional ```torch.Tensor``` ```W2``` as defined in step 6
#   * This is the output layer's weights
# * a one-dimensional ```torch.Tensor``` ```b2``` as defined in step 6
#   * This is the output layer's biases
#
# And returns:
# * a one-dimensional ```torch.Tensor``` ```logits```
#   * This is the output of the output layer before applying an activation function
#
# Video: [0:29:15](https://youtu.be/TCH_1BHY58I?t=1755)

# %%
def get_logits(h, W2, b2):
# Solution code
    return h @ W2 + b2
# End solution code

# %% deletable=false editable=false
def test_get_logits():
    h = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    W2 = torch.tensor([
        [10.0, 10.1, 11.0, 19.9],
        [100.0, -101.1, 0.0, 98.7],
    ])
    b2 = torch.tensor([
        3.0, 5.0, 11.0, 13.0,
    ])

    logits = get_logits(h, W2, b2)

    expected_logits = torch.tensor([
        [ 213.0, -187.1,   22.0,  230.3],
        [ 433.0, -369.1,   44.0,  467.5],
        [ 653.0, -551.1,   66.0,  704.7],
    ])
    expect_tensor_close("logits", logits, expected_logits)
    print("get_logits looks good. Onward!")
test_get_logits()

# %% [markdown] deletable=false editable=false
# ### Step 9: Calculate output softmax activation

# %%
def get_prob(logits):
# Solution code
    counts = torch.exp(logits)
    return counts / counts.sum(dim=1, keepdim=True)
# End solution code

# %% deletable=false editable=false
from math import log

def test_get_prob():
    logits = torch.tensor([
        [log(1), log(2), log(3), log(4)],
        [ 0.123,  0.123,  0.123,  0.123],
    ])

    prob = get_prob(logits)

    expected_prob = torch.tensor([
        [ 0.1,  0.2,  0.3,  0.4],
        [0.25, 0.25, 0.25, 0.25]
    ])
    expect_tensor_close("prob for test case", prob, expected_prob)
    print("get_prob looks good. Onward!")
test_get_prob()

# %% [markdown] deletable=false editable=false
# ### Step 10: Forward propagate from vector embeddings

# %%
def forward_prop(emb, W1, b1, W2, b2):
    h = get_h(emb, W1, b1)
    logits = get_logits(h, W2, b2)
    y_hat = get_prob(logits)

    return y_hat

# %% deletable=false editable=false
def test_forward_prop():
    emb = torch.tensor([
        [ 1.2,  2.1],
        [-0.5, -0.7],
        [ 0.0,  0.5247],
        [-4.0, 3.1]
    ])
    W1 = torch.tensor([
        [2.3, 0.9, 0.7],
        [-3.2, 0.8, 1.3],
    ])
    b1 = torch.tensor([
        0.2, 1.1, -0.1
    ])
    W2 = torch.tensor([
        [ 3.4,  4.5],
        [ 0.6,  0.5],
        [-1.2,  2.2]
    ])
    b2 = torch.tensor([
        0.3, -0.4,
    ])

    y_hat = forward_prop(emb, W1, b1, W2, b2)

    expected_y_hat = torch.tensor([
        [0.1832, 0.8168],
        [0.9396, 0.0604],
        [0.5000, 0.5000],
        [0.2770, 0.7230],
    ])
    expect_tensor_close("y_hat", y_hat, expected_y_hat)
    print("forward_prop looks good. Onward!")
test_forward_prop()

# %% [markdown] deletable=false editable=false
# ### Step 11: Loss calculation

# %%
def get_loss(Y_hat, Y):
# Solution code
    match_probabilities = Y_hat[torch.arange(len(Y_hat)), Y]
    log_likelihoods = match_probabilities.log()
    neg_log_likelihood = -log_likelihoods.mean()
    return neg_log_likelihood
# End solution code

# %% deletable=false editable=false
def test_get_loss():
    Y_hat = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.4512, 0.44933, 0.0, 0.0],
        [0.05, 0.05, 0.81873, 0.08127],
    ])
    Y = torch.tensor([
        0,
        3,
        1,
        2,
    ])
    neg_log_likelihood = get_loss(Y_hat, Y)
    expect_close("negative loss likelihood", neg_log_likelihood, 0.25)
test_get_loss()

# %% [markdown] deletable=false editable=false
# ### Step 12: Gradient descent

# %%
def descend_gradient(t, learning_rate):
# Solution code
    t.data -= learning_rate * t.grad
    return t
# End solution code

# %% deletable=false editable=false
def test_descend_gradient():
    pass
test_descend_gradient()

# %% [markdown] deletable=false editable=false
# ### Step 13: Train model once

# %%
def train_model(X, Y, C, W1, b1, W2, b2, learning_rate):
# Solution code
    emb = get_emb(X, C)
    Y_hat = forward_prop(emb, W1, b1, W2, b2)
    loss = get_loss(Y_hat, Y)
    parameters = (C, W1, b1, W2, b2)
    for parameter in parameters:
        parameter.grad = None
    loss.backward()
    for parameter in parameters:
        descend_gradient(parameter, learning_rate)
    return loss.item()
# End solution code

# %% deletable=false editable=false
def test_train_model():
    pass
test_train_model()

# %% [markdown] deletable=false editable=false
# ### Step 14: Generate a word

# %%
def generate_word(C, block_size, W1, b1, W2, b2, stoi, itos, gen):
    chr = '.'
    word = ""
    while True:
        block = ('.' * block_size + word)[-3:]
        idxes = [stoi[c] for c in block]
        x = torch.tensor([idxes])
        emb = get_emb(x, C)
        probability_distribution = forward_prop(emb, W1, b1, W2, b2)
        sample = torch.multinomial(probability_distribution, 1, generator=gen).item()
        chr = itos[sample]
        if chr == '.':
            break
        word += chr
    return word

# %% deletable=false editable=false
def test_generate_word():
    pass
test_generate_word()

# %% [markdown] deletable=false editable=false
# ### Step 15: Train the model repeatedly

# %%
# Solution code
stoi = get_stoi(loaded_words)
idx_ct = len(stoi)
itos = get_itos(stoi)
block_size = 3
X, Y = get_X_and_Y(loaded_words, stoi, block_size)
embeddings = 2
gen = torch.Generator()
C = get_C(idx_ct, embeddings, gen)
hidden_neuron_ct = 100
W1, b1 = initialize_W_b(block_size * embeddings, hidden_neuron_ct, gen)
W2, b2 = initialize_W_b(hidden_neuron_ct, idx_ct, gen)
learning_rate = .5

for i in range(1, 301, 1):
    loss = train_model(X, Y, C, W1, b1, W2, b2, learning_rate)
    if i == 1 or i % 10 == 0:
        print(f"{i}: {loss}")

print(f"Final loss is {loss}")
# End solution code

# %% [markdown] deletable=false editable=false
# ### Step 15: Generate words

# %%
# Solution code
for i in range(10):
    print(generate_word(C, block_size, W1, b1, W2, b2, stoi, itos, gen))
# End solution code

# %% [markdown] deletable=false editable=false
# ### Bonus: Calculate probability of an empty word

# %%
def get_empty_word_prob(C, W1, b1, W2, b2, stoi):
# Solution code
    emb = get_emb(torch.tensor([[0,0,0]]), C)
    probs = forward_prop(emb, W1, b1, W2, b2)[0]
    prob_empty = probs[stoi['.']]

    # Strictly optional: print the probability map
    prob_map = {letter : probs[idx].item() for letter, idx in stoi.items()}
    prob_map = sorted(prob_map.items(), key = lambda kv: (kv[1], kv[0]))
    for k, v in prob_map:
        print(f"{k}: {v:.5f}")

    return prob_empty
# End solution code

# %% deletable=false editable=false
prob_empty = get_empty_word_prob(C, W1, b1, W2, b2, stoi)
print(f"The probability of this model generating an empty word is {prob_empty}.")

# %%
