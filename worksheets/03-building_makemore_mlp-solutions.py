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
# * An integer (```indices```) representing the number of indices in ```stoi``` to provide embeddings for
# * An integer (```embedding_size```) representing the length of each embedding vector
# * A ```torch.Generator``` (```gen```) to provide (pseudo)random initial values for the parameters
#
# And returns:
# * a ```torch.Tensor``` of ```float64``` (```C```) representing the initial (random) embedding vectors for each index.
#
# Video: [0:03:01](https://youtu.be/TCH_1BHY58I?t=181), [0:12:19](https://youtu.be/TCH_1BHY58I?t=739), and [0:38:49](https://youtu.be/TCH_1BHY58I?t=2329)

# %%
import torch

def get_C(indices, embedding_size, gen):
# Solution code
    return torch.rand((indices, embedding_size), generator=gen, dtype=torch.float64, requires_grad=True)
# End solution code

# %% deletable=false editable=false
def test_get_C():
    indices = 7
    embedding_size = 4
    gen = torch.Generator()
    gen.manual_seed(12345)
    C = get_C(indices, embedding_size, gen)
    expect_type("C", C, torch.Tensor)
    expect_eq("C.dtype", C.dtype, torch.float64)
    expect_eq("C.shape", C.shape, (indices, embedding_size))
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
# Video: [0:05:55](https://youtu.be/TCH_1BHY58I?t=355), [0:13:07](https://youtu.be/TCH_1BHY58I?t=787) and [0:19:10](https://youtu.be/TCH_1BHY58I?t=1150)

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
# Video: [0:18:37](https://youtu.be/TCH_1BHY58I?t=1117), [0:29:17](https://youtu.be/TCH_1BHY58I?t=1757), and [0:38:49](https://youtu.be/TCH_1BHY58I?t=2329)

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
# ### Step 7: Initialize model
#
# Write a function that takes the following arguments:
# * An integer (```idx_ct```) representing the number of indices to provide embeddings for
# * An integer (```block_size```) that specifies how many characters to take into account when predicting the next one
# * An integer (```embedding_size```) representing the length of each embedding vector
# * An integer (```hidden_layer_size```) that specifies the number of neurons in the hidden layer
# * A ```torch.Generator``` (```gen```) to provide (pseudo)random initial values for the parameters
#
# And returns:
# * A Model [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) (defined below) with the following fields:
#   * A ```torch.tensor``` (```C```) representing the embedding vector for each index
#     * See Step 4
#   * A two-dimensional ```torch.Tensor``` (```W1```) representing the weights of the hidden layer
#     * See Step 6
#   * A one-dimensional ```torch.Tensor``` (```b1```) representing the biases of the hidden layer
#     * See Step 6
#   * A two-dimensional ```torch.Tensor``` (```W2```) representing the weights of the output layer
#     * See Step 6
#   * A one-dimensional ```torch.Tensor``` (```b2```) representing the biases of the output layer
#     * See Step 6
#
# Note: Karpathy does not use a namedtuple for these fields.
#
# Video: [0:32:27](https://youtu.be/TCH_1BHY58I?t=1947)

# %%
from collections import namedtuple
Model = namedtuple('Model', ['C', 'W1', 'b1', 'W2', 'b2'])

def initialize_model(idx_ct, block_size, embedding_size, hidden_layer_size, gen):
# Solution code
    C = get_C(idx_ct, embedding_size, gen)
    W1, b1 = initialize_W_b(block_size * embedding_size, hidden_layer_size, gen)
    W2, b2 = initialize_W_b(hidden_layer_size, idx_ct, gen)
    return Model(C, W1, b1, W2, b2)
# End solution code

# %% deletable=false editable=false
import torch

def test_initialize_model():
    idx_ct = 5
    block_size = 4
    embedding_size = 3
    hidden_layer_size = 7
    gen = torch.Generator()

    C, W1, b1, W2, b2 = initialize_model(idx_ct, block_size, embedding_size, hidden_layer_size, gen)

    expect_eq("C.shape", C.shape, (idx_ct, embedding_size))
    expect_eq("W1.shape", W1.shape, (block_size * embedding_size, hidden_layer_size))
    expect_eq("b1.shape", b1.shape, (hidden_layer_size,))
    expect_eq("W2.shape", W2.shape, (hidden_layer_size, idx_ct))
    expect_eq("b2.shape", b2.shape, (idx_ct,))
    print("initialize_model looks good. Onward!")
test_initialize_model()

# %% [markdown] deletable=false editable=false
# ### Step 8: Forward propagate through hidden layer
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
def get_h(emb, W1, b1):
# Solution code
    output = emb @ W1 + b1
    activation = torch.tanh(output)
    return activation
# End solution code

# %% deletable=false editable=false
def test_get_h():
    emb = torch.tensor([
        [0.1, 0.2],
        [-.3, 0.4],
        [.05, -.06],
    ], dtype=torch.float64)
    W1 = torch.tensor([
        [0.7, 0.8, -0.9, -0.1],
        [0.6, 0.5, 0.4, 0.3],
    ], dtype=torch.float64)
    b1 = torch.tensor([
        .09, -.01, .011, -.012
    ], dtype=torch.float64)

    h = get_h(emb, W1, b1)

    expected_h = torch.tensor([
        [ 2.7291e-01,  1.6838e-01,  1.0000e-03,  3.7982e-02],
        [ 1.1943e-01, -4.9958e-02,  4.1447e-01,  1.3713e-01],
        [ 8.8766e-02,  8.6736e-18, -5.7935e-02, -3.4986e-02],
    ], dtype=torch.float64)
    expect_tensor_close("h for test case", h, expected_h)
    print("get_h looks good. Onwards!")
test_get_h()

# %% [markdown] deletable=false editable=false
# ### Step 9: Calculate output layer outputs before activation
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
# ### Step 10: Forward propagate from vector embeddings
#
# Video: [0:32:37](https://youtu.be/TCH_1BHY58I?t=1957)

# %%
def forward_prop(X, model):
    emb = get_emb(X, model.C)
    h = get_h(emb, model.W1, model.b1)
    logits = get_logits(h, model.W2, model.b2)

    return logits

# %% deletable=false editable=false
def test_forward_prop():
    X = torch.tensor([
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ])
    C = torch.tensor([
        [1.2],
        [-0.5],
        [0.0],
        [-4.0],
        [2.1],
        [-0.7],
        [0.5247],
        [3.1],
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

    model = Model(C, W1, b1, W2, b2)
    logits = forward_prop(X, model)

    expected_logits = torch.tensor([
        [-3.6945, -2.1998],
        [ 4.3266,  1.5829],
        [-2.8482, -2.8482],
        [-4.0852, -3.1258],
    ])
    expect_tensor_close("logits", logits, expected_logits)
    print("forward_prop looks good. Onward!")
test_forward_prop()

# %% [markdown] deletable=false editable=false
# ### Step 11: Gradient descent
#
# Video: [0:38:23](https://youtu.be/TCH_1BHY58I?t=2303)

# %%
def descend_gradient(t, learning_rate):
# Solution code
    t.data -= learning_rate * t.grad
    return t
# End solution code

# %% deletable=false editable=false
def test_descend_gradient():
    t = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    t.grad = torch.tensor([
        [0.5, 0.3, 0.1],
        [-0.2, -0.4, -0.6]
    ])
    learning_rate = 2.0

    descend_gradient(t, learning_rate)

    expected_t = torch.tensor([
        [0.0, 1.4, 2.8],
        [4.4, 5.8, 7.2]
    ])
    expect_tensor_close("t", t, expected_t)
    print("descend_gradient looks good. Onward!")
test_descend_gradient()

# %% [markdown] deletable=false editable=false
# ### Step 12: Train model once
#
# Video: [0:37:57](https://youtu.be/TCH_1BHY58I?t=2277)

# %%
def train_once(X, Y, model, learning_rate):
# Solution code
    logits = forward_prop(X, model)
    loss = torch.nn.functional.cross_entropy(logits, Y)
    for parameter in model:
        parameter.grad = None
    loss.backward()
    for parameter in model:
        descend_gradient(parameter, learning_rate)
    return loss.item()
# End solution code

# %% deletable=false editable=false
def test_train_once():
    X = torch.tensor([
        [2, 1, 0, 1],
        [0, 0, 1, 2],
    ])
    Y = torch.tensor([
        0,
        1,
    ])
    C = torch.tensor([
        [1.0],
        [-1.0],
        [0.5],
    ], requires_grad=True)
    W1 = torch.tensor([
        [1.0],
        [1.1],
        [2.1],
        [-2.9]
    ], requires_grad=True)
    b1 = torch.tensor([
        [0.1],
    ], requires_grad=True)
    W2 = torch.tensor([
        [1.0, 2.0, 3.0]
    ], requires_grad=True)
    b2 = torch.tensor([
        1.0, 0.9, 0.8
    ], requires_grad=True)
    model = Model(C, W1, b1, W2, b2)
    learning_rate = 1.0
    loss = train_once(X, Y, model, learning_rate)
    expect_close("loss", loss, 1.8224)
    print("train_once looks good. Onward!")
test_train_once()

# %% [markdown] deletable=false editable=false
# ### Step 13: Initialize indices and model
#
# Video: [0:37:56](https://youtu.be/TCH_1BHY58I?t=2276)

# %%
# Solution code
stoi = get_stoi(loaded_words)
itos = get_itos(stoi)

idx_ct = len(stoi)
block_size = 3
embedding_size = 2
hidden_layer_size = 100
gen = torch.Generator()
model = initialize_model(idx_ct, block_size, embedding_size, hidden_layer_size, gen)
# End solution code

# %% [markdown] deletable=false editable=false
# ### Step 14: Initialize examples and labels
#
# Video: [0:53:20](https://youtu.be/TCH_1BHY58I?t=3200)

# %%
import random

# Solution code
random.shuffle(loaded_words)
train = int(0.8 * len(loaded_words))
dev = int(0.9 * len(loaded_words))
Xtr, Ytr = get_X_and_Y(loaded_words[0:train], stoi, block_size)
Xdev, Ydev = get_X_and_Y(loaded_words[train:dev], stoi, block_size)
Xtest, Ytest = get_X_and_Y(loaded_words[dev:], stoi, block_size)
# End solution code

# %% [markdown] deletable=false editable=false
# ### Step 15: Train the model repeatedly in minibatches
#
# Video: [0:38:38](https://youtu.be/TCH_1BHY58I?t=2318) and [0:42:22](https://youtu.be/TCH_1BHY58I?t=2542)

# %%
# Solution code
learning_rate = .5
minibatch_size = 32

for i in range(1, 30000, 1):
    ix = torch.randint(0, Xtr.shape[0], (minibatch_size,), generator=gen)
    X_mini = Xtr[ix]
    Y_mini = Ytr[ix]
    loss = train_once(X_mini, Y_mini, model, learning_rate)
    if i == 1 or i % 2000 == 0:
        print(f"{i}: {loss}")

logits = forward_prop(Xdev, model)
loss = torch.nn.functional.cross_entropy(logits, Ydev)
print(f"Dev loss is {loss}")
# End solution code

# %% [markdown] deletable=false editable=false
# ### Step 16: Measure the model's testing loss
#

# %%
# Solution code
logits = forward_prop(Xtest, model)
loss = torch.nn.functional.cross_entropy(logits, Ytest)
print(f"Testing loss is {loss}")
# End solution code

# %% [markdown] deletable=false editable=false
# ### Step 17: Get inputs to find probabilities for
#
# Video: [1:13:31](https://youtu.be/TCH_1BHY58I?t=4411)

# %%
def get_sampling_inputs(block_size, stoi, word):
# Solution code
    padded_word = block_size * '.' + word  # Add start-of-word characters in case len(word) < block_size
    block_letters = padded_word[-block_size:]
    block_idxes = [stoi[c] for c in block_letters]
    inputs = torch.tensor([block_idxes])
    return inputs
# End solution code

# %% deletable=false editable=false
def test_get_sampling_inputs():
    block_size = 4
    stoi = {
        '.': 0,
        'h': 1,
        'i': 2,
    }
    word = "hi"

    inputs = get_sampling_inputs(block_size, stoi, word)

    expect_tensor_close("inputs", inputs, torch.tensor([[0, 0, 1, 2]]))
    print("get_sampling_inputs looks good. Onward!")
test_get_sampling_inputs()

# %% [markdown] deletable=false editable=false
# ### Step 18: Sample probability distribution
#
# Video: [1:14:18](https://youtu.be/TCH_1BHY58I?t=4458)

# %%
def sample_distribution(probability_distribution, gen):
# Solution code
    num_samples = 1  # we only need one letter
    sample_idx = torch.multinomial(probability_distribution, num_samples, generator=gen).item()
    return sample_idx
# End solution code

# %% deletable=false editable=false
def test_sample_distribution():
    gen = torch.Generator()
    gen.manual_seed(12345)
    probability_distribution = torch.tensor([0.6, 0.1, 0.3])
    count = 10000

    samples = torch.zeros(3)
    for _ in range(count):
        samples[sample_distribution(probability_distribution, gen)] += 1

    expected_samples = probability_distribution * count
    expect_tensor_close("samples", samples, expected_samples, atol = 200)
    print("sample_distribution looks good. Onward!")
test_sample_distribution()

# %% [markdown] deletable=false editable=false
# ### Step 19: Generate a word by sampling
#
# Video: [1:13:24](https://youtu.be/TCH_1BHY58I?t=4404)

# %%
def generate_word(model, block_size, stoi, itos, sample_distribution_func, gen):
# Solution code
    word = ""
    while True:
        inputs = get_sampling_inputs(block_size, stoi, word)
        logits = forward_prop(inputs, model)
        probability_distribution = torch.nn.functional.softmax(logits, 1)
        sample_idx = sample_distribution_func(probability_distribution, gen)
        sample = itos[sample_idx]
        if sample == '.':
            break
        word += sample
    return word
# End solution code

# %% deletable=false editable=false
def test_generate_word():
    stoi = {
        '.': 0,
        'a': 1,
        'd': 2,
        'n': 3,
        'o': 4,
        'r': 5,
        'w': 6
    }
    itos = {v:k for k,v in stoi.items()}
    block_size = 3
    C = torch.tensor([
        [ 1.0,  0.1],
        [-0.9,  0.3],
        [ 0.2,  0.5],
        [-0.3,  0.6],
        [ 0.6, -0.4],
        [-0.7, -0.8],
        [-0.1,  0.9],
    ])
    W1 = torch.tensor([
        [ 1.3,  0.9],
        [ 0.7, -0.3],
        [-0.5,  1.4],
        [-3.2,  0.8],
        [ 1.3, -0.6],
        [ 1.4, -0.2],
    ])
    b1 = torch.tensor([
        0.2, 1.1
    ])
    W2 = torch.tensor([
        [ 3.4,  4.5, -1.8,  0.7,  0.4,  0.1, -1.1],
        [ 0.6,  0.5,  2.1, -0.9,  1.1, -0.3,  0.8],
    ])
    b2 = torch.tensor([
        0.3, -0.4, 0.2, -0.8, 0.0, -0.1, 1.1
    ])

    model = Model(C, W1, b1, W2, b2)

    target_pos = 0
    target_seq = [0.98, 0.895, 0.99, 0.18, 0.74, 0.25, 0.0]
    def mock_sample(probability_distribution, _):
        nonlocal target_pos
        target_sum = target_seq[target_pos]
        prob_sum = 0.0
        dist_pos = 0
        while True:
            prob_sum += probability_distribution[0][dist_pos]
            if prob_sum >= target_sum:
                break
            dist_pos += 1
        target_pos += 1
        return dist_pos

    word = generate_word(model, block_size, stoi, itos, mock_sample, None)

    expect_eq("word", word, "onward")
    print("generate_word looks good. Onward!")
test_generate_word()

# %% [markdown] deletable=false editable=false
# ### Step 20: Generate words
#
# Video: [1:13:24](https://youtu.be/TCH_1BHY58I?t=4404)

# %%
# Solution code
for i in range(10):
    print(generate_word(model, block_size, stoi, itos, sample_distribution, gen))
# End solution code

# %% [markdown] deletable=false editable=false
# ### Bonus: Calculate probability of an empty word

# %%
def get_empty_word_prob(model, stoi):
# Solution code
    logits = forward_prop(torch.tensor([[0,0,0]]), model)
    print(logits)
    probs = torch.nn.functional.softmax(logits, 1)[0]
    prob_empty = probs[stoi['.']]

    # Strictly optional: print the probability map
    prob_map = {letter : probs[idx].item() for letter, idx in stoi.items()}
    prob_map = sorted(prob_map.items(), key = lambda kv: (kv[1], kv[0]))
    for k, v in prob_map:
        print(f"{k}: {v:.5f}")

    return prob_empty
# End solution code

# %% deletable=false editable=false
prob_empty = get_empty_word_prob(model, stoi)
print(f"The probability of this model generating an empty word is {prob_empty}.")

# %%
