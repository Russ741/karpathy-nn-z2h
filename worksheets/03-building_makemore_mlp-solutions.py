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
# Solution code
    words_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    words = requests.get(words_url).text.splitlines()
    return words
# End solution code

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
# Solution code
    itos = {stoi[c]:c for c in stoi}
    return itos
# End solution code

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
# Solution code
    for word in words:
        word = '.' * block_size + word + '.' * block_size
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
# Solution code
    return torch.rand((indices, embed_dimensions), dtype=torch.float64, generator=gen, requires_grad=True)
# End solution code

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
    if not emb.equal(expected_emb):
        print(f"Expected emb to be \n{expected_emb}\n, was \n{emb}")
        return
    print("get_vector_embedding looks good. Onwards!")
test_get_vector_embedding()

# %% [markdown] deletable=false editable=false
# ### Step 6: Initialize hidden layer coefficients
#
# Write a function that takes the following arguments:
# * the number of inputs (```input_ct```) to each neuron in the hidden layer
#   * Equal to the number of cells in each row of emb
#   * In the video, this is given the value 6
# * the number of neurons (```neuron_ct```) to include in the hidden layer
#   * In the video, this is given the value 100
#
# And returns:
# * a two-dimensional ```torch.Tensor``` ```W``` of shape (```input_ct```, ```neuron_ct```) of type ```torch.float64```
#   * each element of ```W``` should be randomly generated
# * a one-dimensional pytorch.Tensor ```b``` of length ```neuron_ct```
#   * the elements of ```b``` can be zero

# %%
import torch

def initialize_W_b(input_ct, neuron_ct):
# Solution code
    W = torch.rand((input_ct, neuron_ct), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(neuron_ct, dtype=torch.float64, requires_grad=True)

    return W, b
# End solution code

# %% deletable=false editable=false
def test_initialize_W_b():
    input_ct = 3
    neuron_ct = 5
    W, b = initialize_W_b(input_ct, neuron_ct)
    if not torch.is_tensor(W):
        print("Expected W to be a tensor")
        return
    if not torch.is_tensor(b):
        print("Expected B to be a tensor")
        return
    if not W.is_floating_point():
        print("Expected W to be a tensor of floating point numbers")
        return
    if not b.is_floating_point():
        print("Expected b to be a tensor of floating point numbers")
        return
    if (W_shape := W.shape) != (expected_W_shape := (input_ct, neuron_ct)):
        print(f"Expected W shape to be {expected_W_shape}, was {W_shape}")
        return
    # The comma is required to make expected_b_shape into a single-element tuple
    if (b_shape := b.shape) != (expected_b_shape := (neuron_ct,)):
        print(f"Expected b shape to be {expected_b_shape}, was {b_shape}")
        return
    print("W and b look good. Onwards!")
test_initialize_W_b()

# %% [markdown] deletable=false editable=false
# ### Step 7: Forward propagate through hidden layer
#
# Write a function that takes the following arguments:
# * a two-dimensional ```torch.Tensor``` ```emb``` as defined in step 5
#   * This is the input to the hidden layer
# * a two-dimensional ```torch.Tensor``` ```W``` as defined in step 6
#   * This is the hidden layer's weights
# * a one-dimensional ```torch.Tensor``` ```b``` as defined in step 6
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
    if not torch.isclose(expected_h, h, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected h for test case to be \n{expected_h}\n, was \n{h}")
        return
    print("get_h looks good. Onwards!")
test_get_h()

# %% [markdown] deletable=false editable=false
# ### Step : Forward propagate through output layer

# %%
def get_logits(h, W2, b2):
    return h @ W2 + b2

def get_prob(logits):
    counts = torch.exp(logits)
    return counts / counts.sum(dim=1, keepdim=True)

def forward_prop(emb, W1, b1, W2, b2):
    h = get_h(emb, W1, b1)
    logits = get_logits(h, W2, b2)
    y_hat = get_prob(logits)

    return y_hat

def get_loss(Y_hat, Y):
    match_probabilities = Y_hat[torch.arange(len(Y)), Y]
    neg_log_likelihood = -match_probabilities.log().mean()
    return neg_log_likelihood

def descend_gradient(W, b, learning_rate):
    W.data -= learning_rate * W.grad
    b.data -= learning_rate * b.grad
    return W, b

def train_model(emb, Y, C, W1, b1, W2, b2, learning_rate):
    Y_hat = forward_prop(emb, W1, b1, W2, b2)
    loss = get_loss(Y_hat, Y)
    C.grad = None
    W1.grad = None
    b1.grad = None
    W2.grad = None
    b2.grad = None
    loss.backward()
    C.data -= learning_rate * C.grad
    W2, b2 = descend_gradient(W2, b2, learning_rate)
    W1, b1 = descend_gradient(W1, b1, learning_rate)
    return loss.item()

def generate_word(C, block_size, W1, b1, W2, b2, stoi, itos, gen):
    chr = '.'
    word = ""
    while True:
        block = ('.' * block_size + word)[-3:]
        print(f"{block=}")
        idxes = [stoi[c] for c in block]
        x = torch.tensor([idxes])
        emb = get_emb(x, C)
        probability_distribution = forward_prop(emb, W1, b1, W2, b2)
        print(f"{probability_distribution=}")
        sample = torch.multinomial(probability_distribution, 1, generator=gen).item()
        chr = itos[sample]
        print(f"{chr=}")
        if chr == '.':
            break
        word += chr
    return word

# %%

stoi = get_stoi(loaded_words)
idx_ct = len(stoi)
itos = get_itos(stoi)
block_size = 3
X, Y = get_X_and_Y(loaded_words, stoi, block_size)
embeddings = 2
gen = torch.Generator()
C = get_C(idx_ct, embeddings, gen)
emb = get_emb(X, C)
hidden_neuron_ct = 100
W1, b1 = initialize_W_b(len(emb[0]), hidden_neuron_ct)
W2, b2 = initialize_W_b(hidden_neuron_ct, idx_ct)
learning_rate = 2.5

for i in range(1, 101, 1):
    loss = train_model(emb, Y, C, W1, b1, W2, b2, learning_rate)

print(f"Final loss is {loss}")

# %%
for i in range(10):
    print(generate_word(C, block_size, W1, b1, W2, b2, stoi, itos, gen))

# %%

print(f"{b2=}")

# %% [markdown] deletable=false editable=false
# ### Step : Calculate loss

# %% [markdown] deletable=false editable=false
# ### Step : Gradient descent

# %% [markdown] deletable=false editable=false
# ### Step : Train model

# %% [markdown] deletable=false editable=false
# ### Step : Generate words


