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
# # Worksheet 2a - Probability Table
#
# This is the second in a series of companion worksheets for for Andrej Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) videos.
#
# It corresponds to roughly the first half of the second video in the series, named "[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)".
#
# The first worksheet in the series is provided by Andrej, and can be found [here](https://colab.research.google.com/drive/1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN?usp=sharing).
#
# The rest of the worksheets are listed in the README [here](https://github.com/Russ741/karpathy-nn-z2h/).
#
# The overall objective of this worksheet is to write code that generates a word that is similar to a set of example words it is trained on.
#
# Note that this worksheet uses a probability table, *not* neural networks like subsequent neural networks.

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
# Objective: Load a list of words from the word file at ```WORDS_PATH``` into a list variable named ```words```.
#
# Note: In practice, the order of the strings in the returned list does not matter, but for the
# test to pass, they should be in the same order in the list as in the word file.
#
# Video: [0:03:03](https://youtu.be/PaCmpygFfXo?t=183)

# %%
# Solution code
words = open(WORDS_PATH).read().splitlines()
# End solution code

# %% deletable=false editable=false
def test_words():
    expect_type("words", words, list)
    expect_eq("len(words)", len(words), 32033)
    expect_eq("words[0]", words[0], "emma")
    expect_eq("words[-1]", words[-1], "zzyzx")
    print("words looks good. Onwards!")
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
bigrams = []
# Solution code
for word in words:
    bigrams.append(('.', word[0]))
    for pos in range(len(word) - 1):
        bigrams.append((word[pos], word[pos + 1]))
    bigrams.append((word[-1], '.'))
# End solution code

# %% deletable=false editable=false
def test_bigrams():
    expect_type("bigrams", bigrams, list)
    expect_eq("count of ('.', 'm') bigrams", bigrams.count(('.', 'm')), 2538)
    expect_eq("count of ('a', 'b') bigrams", bigrams.count(('a', 'b')), 541)
    expect_eq("count of ('s', '.') bigrams", bigrams.count(('s', '.')), 1169)
    print("bigrams looks good. Onwards!")
test_bigrams()

# %% [markdown] deletable=false editable=false
# ### Step 2: Map characters to indices
#
# Objective: Build a dict ```stoi``` where the key is a character from ```words``` (including '.' for start/end) and the value is a unique integer.
#
# (We'll use these unique integers as an index to represent the characters in a Tensor in later steps)
#
# Video: [0:15:40](https://youtu.be/PaCmpygFfXo?t=940)

# %%
stoi = {}
# Solution code
chars = set()
for bigram in bigrams:
    chars.add(bigram[0])
    chars.add(bigram[1])
stoi = { v:k for (k, v) in enumerate(sorted(chars))}
# End solution code

# %% deletable=false editable=false
import string

def test_stoi():
    expect_type("stoi", stoi, dict)
    for c in string.ascii_lowercase:
        if not c in stoi:
            raise Exception(f"Expected {c} to be in stoi")
    print("stoi looks good. Onwards!")
test_stoi()

# %% [markdown] deletable=false editable=false
# ### Step 3: Map indices to characters
#
# Objective: Build a dict ```itos``` that has the same key-value pairs as ```stoi```, but with each pair's key and value swapped.
#
# Video: [0:18:49](https://youtu.be/PaCmpygFfXo?t=1129)

# %%
itos = {}
# Solution code
itos = {stoi[c]:c for c in stoi}
# End solution code

# %% deletable=false editable=false
def test_itos():
    expect_type("itos", itos, dict)
    expect_eq("len(itos)", len(itos), len(stoi))
    for k,v in stoi.items():
        expect_eq(f"itos.get({v})", itos.get(v), k)
    print("itos looks good. Onwards!")
test_itos()

# %% [markdown] deletable=false editable=false
# ### Step 4: Count occurrences of each bigram
#
# Objective: Build a torch Tensor ```N``` where:
# * the row is the index of the first character in the bigram
# * the column is the index of the second character in the bigram
# * the value is the number of times that bigram occurs (represented as an integer)
#
# Video: [0:12:45](https://www.youtube.com/watch?v=PaCmpygFfXo&t=1315s)

# %%
import torch

# Solution code
N = torch.zeros(len(stoi), len(stoi), dtype=torch.int32)
for bigram in bigrams:
    i0 = stoi[bigram[0]]
    i1 = stoi[bigram[1]]
    N[i0][i1] += 1
# End solution code

# %% deletable=false editable=false
def test_N():
    expect_type("N", N, torch.Tensor)
    expect_eq("N.dtype", N.dtype, torch.int32)
    expect_eq("N.shape", N.shape, (27, 27))
    expect_eq("N.sum()", N.sum(), 228146)
    expect_eq("N for ('.', 'm')", N[stoi['.']][stoi['m']], 2538)
    expect_eq("N for ('m', '.')", N[stoi['m']][stoi['.']], 516)
    print("N looks good. Onwards!")
test_N()

# %% [markdown]
# ### Step 5: Build probability distribution of bigrams
#
# Objective: Build a torch Tensor ```P``` where:
# * the row is the index of the first character in a bigram
# * the column is the index of the second character in a bigram
# * the value is the probability (as torch.float64) of a bigram in ```bigrams``` ending with the second character if it starts with the first character
#
# Video: [0:25:35](https://youtu.be/PaCmpygFfXo?t=1535) and [0:36:17](https://youtu.be/PaCmpygFfXo?t=2177)

# %%
# Solution code
P = torch.zeros(len(stoi), len(stoi), dtype=torch.float64)
N_sum = N.sum(1, keepdim=True)
P = N / N_sum
# End solution code

# %% deletable=false editable=false
def test_P():
    for row_idx in itos:
        row_sum = P[row_idx].sum().item()
        expect_close(f"sum of P[{row_idx}] (for character '{itos[row_idx]}')", row_sum, 1.0)
    print("P looks good. Onwards!")
test_P()


# %% [markdown] deletable=false editable=false
# ### Step 6: Write a bigram probability calculation function
#
# This is slightly different from the steps that the Karpathy video follows, but will make it easier for this worksheet to verify your code.
#
# Write a ```bigram_probability``` function that takes the following arguments:
# * a 2-element tuple of characters
#
# And returns:
# * a floating-point number from 0.0 to 1.0 that represents the probability of the second character following the first character

# %%
def bigram_probability(bigram):
# Solution code
    return P[stoi[bigram[0]]][stoi[bigram[1]]]
# End solution code

# %% deletable=false editable=false
def test_bigram_probability():
    expect_eq("bigram_probability(('.', '.'))", bigram_probability(('.', '.')), 0.0)
    expect_close("bigram_probability(('m', 'a'))", bigram_probability(('m', 'a')), 0.3899)
    print("bigram_probability looks good. Onwards!")
test_bigram_probability()

# %% [markdown] deletable=false editable=false
# ### Step 7: Write a negative log likelihood loss function
#
# Write a ```calculate_loss``` function that takes the following arguments:
# * the name of the bigram probability function written in step 6
# * a list of bigrams (2-element tuples)
#
# And returns:
# * a floating-point number representing the negative log likelihood of all of the bigrams in the list argument
#     * Note that Karpathy defines this to be the negative of the *mean* of each tuple's log likelihood, not their sum
#
# Video: [0:50:47](https://youtu.be/PaCmpygFfXo?t=3047)

# %%
# The sample solution uses this library; if your code doesn't, feel free to remove it.
import math

def calculate_loss(probability_func, bigram_list):
# Solution code
    probabilities = list(map(probability_func, bigram_list))
    log_probabilities = list(map(math.log, probabilities))
    negative_log_likelihood = - sum(log_probabilities) / len(log_probabilities)
    return negative_log_likelihood
# End solution code

# %% deletable=false editable=false
def test_calculate_loss():
    bigrams = [('.', 'a'), ('a', 'b'), ('b', '.')]
    expect_close("calculate_loss for a probability_func that always returns 1.0", calculate_loss(lambda _ : 1.0, bigrams), 0.0)
    # TODO: Handle zero-probability tuples somehow.
    # if (all_zeroes := calculate_loss(lambda _ : 0.0, bigrams)) != (expected_all_zeroes := math.inf):
    #    print(f"Using a probability_func that always returns 0.0 resulted in {all_zeroes}, expected {expected_all_zeroes} ")
    #    return
    expect_close("calculate_loss for bigram_probability", calculate_loss(bigram_probability, bigrams), 3.0881)
    print("calculate_loss looks good. Onwards!")
test_calculate_loss()

# %% [markdown] deletable=false editable=false
# ### Step 8: Calculate ```bigram_probability```'s loss for the bigrams in ```words```
#
# Use the function from step #7 to calculate the bigram probability function's loss when given all of the bigrams in ```words```

# %%
loss_for_words = 0.0
# Solution code
loss_for_words = calculate_loss(bigram_probability, bigrams)
# End solution code

# %% deletable=false editable=false
def test_loss_for_words():
    expect_close("loss_for_words", loss_for_words, 2.4540)
    print("loss_for_words looks good. Congratulations!")
test_loss_for_words()


# %% [markdown] deletable=false editable=false
# ### Step 9: Pick characters based on the probabilities
#
# Objective: Write a function that takes the following arguments:
# * a probability distribution like the one you built in step 5
# * a ```torch.Generator``` to make the selection process deterministic
#
# And returns:
# * a word (string) generated by repeatedly sampling the probability distribution to determine the next character in the string
#
# Video: [0:26:28](https://youtu.be/PaCmpygFfXo?t=1588)

# %%
def generate_word(probabilities, generator):
# Solution code
    current_letter_index = stoi['.']
    word = ""
    while True:
        current_letter_index = torch.multinomial(probabilities[current_letter_index], 1, generator=generator).item()
        current_letter_index
        if current_letter_index == stoi['.']:
            break
        word += itos[current_letter_index]
    return word
# End solution code

# %%
def test_generate_word():
    generator = torch.Generator()
    generator.manual_seed(2147483645)
    # Colab with pytorch 2 produces different samples with the same seed than local Jupyter Lab/Notebook
    # Check for both
    expected_words = ("machina", "drlen")
    if not (word := generate_word(P, generator)) in expected_words:
        raise Exception(f"Generated word was {word}, expected one of {expected_words}")
    print("generate_word looks good. Onwards!")
test_generate_word()

# %%
