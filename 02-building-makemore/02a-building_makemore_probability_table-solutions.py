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
# Exercises for Andrej Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) videos.
#
# This notebook is for Part 2: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)

# %% [markdown] deletable=false editable=false
# ### Preamble: Load data
#
# Objective: Load a list of words from the [names.txt](https://github.com/karpathy/makemore/blob/master/names.txt) file into a list variable named ```words```.

# %%
# To load names.txt from the makemore GitHub page:
import requests

words_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
words = requests.get(words_url).text.splitlines()

# To load names.txt from a local file:
# # curl https://raw.githubusercontent.com/karpathy/makemore/master/names.txt > names.txt
#
# # read() gets the file as one long string with line breaks in it
# # splitlines() divides the whole-file string into a list of strings and removes the line breaks
# words = open("names.txt").read().splitlines()

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
bigrams = []
for word in words:
    bigrams.append(('.', word[0]))
    for pos in range(len(word) - 1):
        bigrams.append((word[pos], word[pos + 1]))
    bigrams.append((word[-1], '.'))

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
chars = set()
for bigram in bigrams:
    chars.add(bigram[0])
    chars.add(bigram[1])
stoi = { v:k for (k, v) in enumerate(sorted(chars))}

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
itos = {stoi[c]:c for c in stoi}

# %% [markdown] deletable=false editable=false
# ### Step 4: Count occurrences of each bigram
#
# Objective: Build a torch Tensor ```N``` where:
# * the row is the index of the first character in the bigram
# * the column is the index of the second character in the bigram
# * the value is the number of times that bigram occurs (represented as an integer)

# %%
import torch

N = torch.zeros(len(stoi), len(stoi), dtype=torch.int32)
for bigram in bigrams:
    i0 = stoi[bigram[0]]
    i1 = stoi[bigram[1]]
    N[i0][i1] += 1

# %% [markdown]
# ### Step 5: Build probability distribution of bigrams
#
# Objective: Build a torch Tensor ```P``` where:
# * the row is the index of the first character in a bigram
# * the column is the index of the second character in a bigram
# * the value is the probability (as torch.float64) of a bigram in ```bigrams``` ending with the second character if it starts with the first character

# %%
P = torch.zeros(len(stoi), len(stoi), dtype=torch.float64)
# TODO: Fill in P with probability distributions for bigrams.
N_sum = N.sum(1, keepdim=True)
P = N / N_sum

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
def bigram_probability(bigram):
    return P[stoi[bigram[0]]][stoi[bigram[1]]]

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
import math

def calculate_loss(probability_func, bigram_list):
    probabilities = list(map(probability_func, bigram_list))
    log_probabilities = list(map(math.log, probabilities))
    negative_log_likelihood = - sum(log_probabilities) / len(log_probabilities)
    return negative_log_likelihood

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
loss_for_words = calculate_loss(bigram_probability, bigrams)

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
def generate_word(probabilities, generator):
    current_letter_index = stoi['.']
    word = ""
    while True:
        current_letter_index = torch.multinomial(probabilities[current_letter_index], 1, generator=generator).item()
        current_letter_index
        if current_letter_index == stoi['.']:
            break
        word += itos[current_letter_index]
    return word


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
