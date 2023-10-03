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

# %%

def load_words():
    words = open("../names.txt").read().splitlines()
    return words

# %%

def generate_bigrams(words):
    bigrams = []
    for word in words:
        bigrams.append(('.', word[0]))
        for pos in range(len(word) - 1):
            bigrams.append((word[pos], word[pos + 1]))
        bigrams.append((word[-1], '.'))
    return bigrams

# %%

def get_stoi(bigrams):
    chars = set()
    for bigram in bigrams:
        chars.add(bigram[0])
        chars.add(bigram[1])
    stoi = { v:k for (k, v) in enumerate(sorted(chars))}
    return stoi

# %%

def get_itos(stoi):
    itos = {stoi[c]:c for c in stoi}
    return itos

# %%

def get_x_and_y(bigrams, stoi):
    x = torch.tensor(list(map(lambda bigram : stoi[bigram[0]], bigrams)))
    y = torch.tensor(list(map(lambda bigram : stoi[bigram[-1]], bigrams)))

    return x, y

# %%

import torch
def initialize_w_b(stoi):
    stoi_n = len(stoi)
    W = torch.rand((stoi_n,stoi_n), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((1,stoi_n),dtype=torch.float64, requires_grad=True)

    return W, b

# %%

def forward_prop(x, W, b):
    one_hot = torch.nn.functional.one_hot(x).double()
    output = torch.matmul(one_hot, W) + b

    softmax = output.exp()
    softmax = softmax / softmax.sum(1, keepdim=True)

    return softmax

# %%

def calculate_loss(y_hat, y):
    match_probabilities = y_hat[torch.arange(len(y)), y]
    neg_log_likelihood = -match_probabilities.log().mean()
    return neg_log_likelihood

# %%

def descend_gradient(W, b, learning_rate):
    W.data -= learning_rate * W.grad
    b.data -= learning_rate * b.grad
    return W, b

# %%

def main():
    words = load_words()
    bigrams = generate_bigrams(words)
    stoi = get_stoi(bigrams)
    itos = get_itos(stoi)
    x, y = get_x_and_y(bigrams, stoi)
    W, b = initialize_w_b(stoi)
    for i in range(1, 1001, 1):
        y_hat = forward_prop(x,W,b)
        loss = calculate_loss(y_hat, y)
        if i % 10 == 0:
            print(f"Round {i} loss: {loss.item()}")
        W.grad = None
        b.grad = None
        loss.backward()
        W, b = descend_gradient(W, b, 10.0)


if __name__ == "__main__":
    main()
