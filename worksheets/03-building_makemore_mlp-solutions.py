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
# ### Step : Generate X and Y

# %% [markdown] deletable=false editable=false
# ### Step : Initialize vector embedding lookup table

# %% [markdown] deletable=false editable=false
# ### Step : Convert X to tensor vector

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


