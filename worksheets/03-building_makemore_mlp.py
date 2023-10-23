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
# ### Preamble: Load data
#
# Write a function that:
# * Loads the remotely-hosted [names.txt](https://github.com/karpathy/makemore/blob/master/names.txt) file
# ([raw link](https://github.com/karpathy/makemore/raw/master/names.txt))
#
# And returns:
# * a list of strings (```words```)
#   * Each string should be equal to the word from the corresponding line of names.txt
#   * The strings should not include line-break characters
#
# Notes:
# * You can reuse your work from the previous worksheet for this if you like.
# * The test_words block below will save the loaded words as loaded_words for you to reuse later
#
# Video: [0:09:10](https://youtu.be/TCH_1BHY58I?t=550)

# %%
# The sample solution uses this library; if your code doesn't, feel free to remove it.
import requests

def load_words():
# TODO: Implement solution here

# %% deletable=false editable=false
def test_words():
    if not isinstance(loaded_words, list):
        print(f"Expected words to be a list")
        return
    if (len_words := len(loaded_words)) != (expected_words := 32033):
        print(f"Expected {expected_words} elements in words, found {len_words} elements")
        return
    sorted_words = sorted(loaded_words)
    if (zeroth_word := sorted_words[0]) != (expected_zeroth := "aaban"):
        print(f"Expected zeroth word in words to be '{expected_zeroth}', was '{zeroth_word}'")
        return
    if (final_word := sorted_words[-1]) != (expected_final := "zzyzx"):
        print(f"Expected final word in words to be '{expected_final}', was '{final_word}'")
        return
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
# TODO: Implement solution here

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
# TODO: Implement solution here

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
# TODO: Implement solution here

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

    if not torch.is_tensor(X):
        print(f"Expected X to be a tensor, was {type(X)}")
        return
    if not torch.is_tensor(Y):
        print(f"Expected Y to be a tensor, was {type(Y)}")
        return
    expected_X = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 3],
        [0, 3, 4],
        [3, 4, 5],
        [4, 5, 0],
        [5, 0, 0],
    ])
    expected_Y = torch.tensor([
        1,
        2,
        0,
        0,
        0,
        3,
        4,
        5,
        0,
        0,
        0,
    ])
    if (shape_X := X.shape) != (expected_shape_X := expected_X.shape):
        print(f"Expected shape of X for test case to be {expected_shape_X}, was {shape_X}")
        return
    if not X.equal(expected_X):
        print(f"Expected X for test case to be {expected_X}, was {X}")
        return
    if not Y.equal(expected_Y):
        print(f"Expected Y for test case to be {expected_Y}, was {Y}")
        return
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
# TODO: Implement solution here

# %% deletable=false editable=false
def test_get_C():
    indices = 7
    embed_dimensions = 4
    gen = torch.Generator()
    gen.manual_seed(12345)
    C = get_C(indices, embed_dimensions, gen)
    if not torch.is_tensor(C):
        print(f"Expected C to be a tensor, was {type(C)}")
        return
    if not torch.is_floating_point(C):
        print(f"Expected C to be a tensor of floating point.")
        return
    if (shape_C := C.shape) != (expected_shape_C := (indices, embed_dimensions)):
        print(f"Expected shape of C for test case to be {expected_shape_C}, was {shape_C}")
        return
    for i in range(len(C)):
        for j in range(len(C)):
            if i == j:
                continue
            if C[i].equal(C[j]):
                print(f"Rows {i} and {j} of C are too similar.")
                print(f"{C[i]=}")
                print(f"{C[j]=}")
                return
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
# TODO: Implement solution here

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
    if not emb.equal(expected_emb):
        print(f"Expected emb to be \n{expected_emb}\n, was \n{emb}")
        return
    print("get_vector_embedding looks good. Onwards!")
test_get_vector_embedding()

# %% [markdown] deletable=false editable=false
# ### Step : Initialize hidden layer coefficients

# %% [markdown] deletable=false editable=false
# ### Step : Forward propagate through hidden layer

# %% [markdown] deletable=false editable=false
# ### Step : Initialize output layer coefficients

# %% [markdown] deletable=false editable=false
# ### Step : Forward propagate through output layer

# %% [markdown] deletable=false editable=false
# ### Step : Calculate loss

# %% [markdown] deletable=false editable=false
# ### Step : Gradient descent

# %% [markdown] deletable=false editable=false
# ### Step : Train model

# %% [markdown] deletable=false editable=false
# ### Step : Generate words


