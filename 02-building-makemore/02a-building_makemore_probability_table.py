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
# ### Preamble: Load data
#
# Objective: Load a list of words from the remotely-hosted [names.txt](https://github.com/karpathy/makemore/blob/master/names.txt) file
# ([raw link](https://github.com/karpathy/makemore/raw/master/names.txt)) into a list variable named ```words```.

# %%
# TODO: Implement solution here

# %% deletable=false editable=false
def test_words():
    if not isinstance(words, list):
        print(f"Expected words to be a list")
        return
    if (len_words := len(words)) != (expected_words := 32033):
        print(f"Expected {expected_words} elements in words, found {len_words} elements")
        return
    if (zeroth_word := words[0]) != (expected_zeroth := "emma"):
        print(f"Expected zeroth word in words to be '{expected_zeroth}', was '{zeroth_word}'")
        return
    if (final_word := words[-1]) != (expected_final := "zzyzx"):
        print(f"Expected final word in words to be '{expected_final}', was '{final_word}'")
        return
    print("words looks good. Onwards!")
test_words()

# %% [markdown] deletable=false editable=false
# ### Step 1: Generate bigrams
#
# Objective: Populate the variable ```bigrams``` with a list of bigrams (2-element tuples) of adjacent characters in ```words```.
#
# Treat the start and end of each word as the character '.'

# %%
# TODO: Implement solution here

# %% deletable=false editable=false
def test_bigrams():
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
    print("bigrams looks good. Onwards!")
test_bigrams()

# %% [markdown] deletable=false editable=false
# ### Step 2: Map characters to indices
#
# Objective: Build a dict ```stoi``` where the key is a character from ```words``` (including '.' for start/end) and the value is a unique integer.
#
# (We'll use these unique integers as an index to represent the characters in a Tensor in later steps)

# %%
stoi = {}
# TODO: Implement solution here

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
    print("stoi looks good. Onwards!")
test_stoi()

# %% [markdown] deletable=false editable=false
# ### Step 3: Map indices to characters
#
# Objective: Build a dict ```itos``` that has the same key-value pairs as ```stoi```, but with each pair's key and value swapped.

# %%
itos = {}
# TODO: Implement solution here

# %% deletable=false editable=false
def test_itos():
    if not isinstance(itos, dict):
        print(f"Expected itos to be a dict")
        return
    if (len_itos := len(itos)) != (expected_len := len(stoi)):
        print(f"Expected length to be {expected_len}, was {len_itos}")
        return
    for k,v in stoi.items():
        if v not in itos:
            print(f"Expected {v} to be a key in itos")
            return
        if (itos_v := itos[v]) != (expected_itos_v := k):
            print(f"Expected itos[{v}] to be {expected_itos_v}, was {itos_v}")
            return
    print("itos looks good. Onwards!")
test_itos()

# %% [markdown] deletable=false editable=false
# ### Step 4: Count occurrences of each bigram
#
# Objective: Build a torch Tensor ```N``` where:
# * the row is the index of the first character in the bigram
# * the column is the index of the second character in the bigram
# * the value is the number of times that bigram occurs (represented as an integer)

# %%
import torch

# TODO: Implement solution here

# %% deletable=false editable=false
def test_N():
    if torch.is_floating_point(N):
        print(f"Expected N to be a tensor of integral type, was of floating point type.")
        return
    if (N_shape := N.shape) != (expected_N_shape := (27, 27)):
        print(f"Expected shape of N to be {expected_N_shape}, was {N_shape}")
        return
    if (N_sum := N.sum()) != (expected_N_sum := 228146):
        print(f"Expected the sum of elements in N to be {expected_N_sum}, was {N_sum}")
        return
    if (N_start_m := N[stoi['.']][stoi['m']]) != (expected_N_start_m := 2538):
        print(f"Expected N for ('.', 'm') to be {expected_N_start_m}, was {N_start_m}")
        return
    if (N_m_end := N[stoi['m']][stoi['.']]) != (expected_N_m_end := 516):
        print(f"Expected N for ('m', '.') to be {expected_N_m_end}, was {N_m_end}")
        return
    print("N looks good. Onwards!")
test_N()

# %% [markdown]
# ### Step 5: Build probability distribution of bigrams
#
# Objective: Build a torch Tensor ```P``` where:
# * the row is the index of the first character in a bigram
# * the column is the index of the second character in a bigram
# * the value is the probability (as torch.float64) of a bigram in ```bigrams``` ending with the second character if it starts with the first character

# %%
# TODO: Implement solution here

# %% deletable=false editable=false
def test_P():
    for row_idx in itos:
        if abs(1.0 - (row_sum := P[row_idx].sum().item())) > 0.00001:
            row_c = itos[row_idx]
            print(f"Expected the sum of row {row_idx} ({row_c}) to be 1.0, was {row_sum}")
            return
    print("P looks good. Onwards!")
test_P()


# %% [markdown] deletable=false editable=false
# ### Step #6: Write a bigram probability calculation function
#
# This is slightly different from the steps that the Karpathy video follows, but will make it easier for this worksheet to verify your code.
#
# Write a ```bigram_probability``` function that takes the following arguments:
# * a 2-element tuple of characters
#
# And returns:
# * a floating-point number from 0.0 to 1.0 that represents the probability of the second character following the first character

# %%
# TODO: Implement solution here

# %% deletable=false editable=false
def test_bigram_probability():
    if (prob_start_end := bigram_probability(('.', '.'))) != (expected_start_end := 0.0):
        print(f"Calculated probability of ('.', '.') is {prob_start_end}, expected {expected_start_end}")
        return
    if abs((prob_m_a := bigram_probability(('m', 'a'))) - (expected_m_a := 0.3899)) > 0.001:
        print(f"Calculated probability of ('m', 'a') is {prob_m_a}, expected {expected_m_a}")
        return
    print("bigram_probability looks good. Onwards!")
test_bigram_probability()

# %% [markdown] deletable=false editable=false
# ### Step #7: Write a negative log likelihood loss function
#
# Write a ```calculate_loss``` function that takes the following arguments:
# * the name of the bigram probability function written in step 6
# * a list of bigrams (2-element tuples)
#
# And returns:
# * a floating-point number representing the negative log likelihood of all of the bigrams in the list argument
#     * Note that Karpathy defines this to be the negative of the *mean* of each tuple's log likelihood, not their sum

# %%
# TODO: Implement solution here

# %% deletable=false editable=false
def test_calculate_loss():
    bigrams = [('.', 'a'), ('a', 'b'), ('b', '.')]
    if abs((all_ones := calculate_loss(lambda _ : 1.0, bigrams)) - (expected_all_ones := 0.0)) > 0.0001:
        print(f"Using a probability_func that always returns 1.0 resulted in {all_ones}, expected {expected_all_ones}")
        return
    # TODO: Handle zero-probability tuples somehow.
    # if (all_zeroes := calculate_loss(lambda _ : 0.0, bigrams)) != (expected_all_zeroes := math.inf):
    #    print(f"Using a probability_func that always returns 0.0 resulted in {all_zeroes}, expected {expected_all_zeroes} ")
    #    return
    if abs((using_bp := calculate_loss(bigram_probability, bigrams)) - (expected_using_bp := 3.0881)) > 0.0001:
        print(f"Using your bigram_probability function resulted in {using_bp}, expected {expected_using_bp}")
        return
    print("calculate_loss looks good. Onwards!")
test_calculate_loss()

# %% [markdown] deletable=false editable=false
# ### Step 8: Calculate ```bigram_probability```'s loss for the bigrams in ```words```
#
# Use the function from step #7 to calculate the bigram probability function's loss when given all of the bigrams in ```words```

# %%
loss_for_words = 0.0
# TODO: Implement solution here

# %% deletable=false editable=false
def test_loss_for_words():
    if abs(loss_for_words - (expected_loss := 2.4540)) > 0.0001:
        print(f"loss_for_words is {loss_for_words}, expected {expected_loss}")
        return
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

# %%
# TODO: Implement solution here

# %%
def test_generate_word():
    generator = torch.Generator()
    generator.manual_seed(2147483645)
    if (word := generate_word(P, generator)) != (expected_word := "machina"):
        print(f"Generated word was {word}, expected {expected_word}")
        return
    print("generate_word looks good. Onwards!")
test_generate_word()

# %%
