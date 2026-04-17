import torch
import numpy as np
import sys
import time
sys.path.insert(0, 'python')
import bobr
from mltoolkit.testing_tools import timing
torch.set_float32_matmul_precision('high')

def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32).reshape(-1, 784) / 255.0
    x_test  = x_test.astype(np.float32).reshape(-1, 784) / 255.0

    def one_hot(y, c=10):
        oh = np.zeros((len(y), c), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    return x_train, one_hot(y_train), x_test, one_hot(y_test)

def torch_load_mnist():
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = torch.from_numpy(x_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()
    x_test = torch.from_numpy(x_test).cuda()
    y_test = torch.from_numpy(y_test).cuda()
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()
x_torch_train, y_torch_train, x_torch_test, y_torch_test = torch_load_mnist()


# @title Constants
BATCH_SIZE      = 64 # @param 
NUM_EPOCHS      = 50 # @param 
LR              = 1e-5 # @param 
WEIGTH_DECAY    = 0 # @param
NUM_BATCHES = len(x_train) // BATCH_SIZE


# @title Bobr model definition
model = bobr.Sequential()
model.add(bobr.LinearRelu(784, 128))
model.add(bobr.LinearRelu(128, 64))
model.add(bobr.Linear(64, 10))

x = bobr.from_numpy(x_train[:BATCH_SIZE]); x.label = "X"
y = bobr.from_numpy(y_train[:BATCH_SIZE]); y.label = "Y"

logits = model.forward(x); logits.label = "logits"
loss   = bobr.cross_entropy(logits, y); loss.label = "loss"

g = bobr.make_model(loss)
g.build_backward()

optim = bobr.AdamW(params=model.parameters(), lr=LR, weight_decay=WEIGTH_DECAY)

def train_bobr():
    # @title Bobr train
    for epoch in range(NUM_EPOCHS):

        for batch_idx in range(NUM_BATCHES):
            s = batch_idx * BATCH_SIZE
            e = s + BATCH_SIZE

            x.data_from_numpy(x_train[s:e])
            y.data_from_numpy(y_train[s:e])

            g.fwd_bwd_pass()
            bobr.sync()

            optim.step()
            bobr.sync()

def check_acc_bobr():
    def accuracy(logits_np, labels_np):
        preds  = np.argmax(logits_np, axis=-1)
        truths = np.argmax(labels_np, axis=-1)
        return np.mean(preds == truths)

    test_correct = 0
    test_total   = 0
    n_test_batches = len(x_test) // BATCH_SIZE
    for tb in range(n_test_batches):
        s = tb * BATCH_SIZE
        e = s + BATCH_SIZE
        x.data_from_numpy(x_test[s:e])
        y.data_from_numpy(y_test[s:e])
        bobr.sync()
        g.fwd()
        bobr.sync()
        test_correct += accuracy(np.array(logits.data), y_test[s:e]) * BATCH_SIZE
        test_total   += BATCH_SIZE

    print(f"Bobr Test acc: {test_correct/test_total*100:.2f}% | ")

with timing("Training bobr"):
    train_bobr()
check_acc_bobr()

# @title Torch model definition
torch_model = torch.nn.Sequential(*[
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
]).cuda()
torch_loss = torch.nn.CrossEntropyLoss()
torch_optim = torch.optim.AdamW(torch_model.parameters(), lr=LR, weight_decay=WEIGTH_DECAY)


def train_torch():
    # @title Torch train
    for epoch in range(NUM_EPOCHS):

        for batch_idx in range(NUM_BATCHES):
            s = batch_idx * BATCH_SIZE
            e = s + BATCH_SIZE

            X = x_torch_train[s:e]
            Y = y_torch_train[s:e]

            torch_optim.zero_grad()

            logits = torch_model(X)
            loss = torch_loss(logits, Y)
            loss.backward()
            torch_optim.step()

def check_acc_torch():
    def accuracy(logits_np, labels_np):
        preds  = np.argmax(logits_np, axis=-1)
        truths = np.argmax(labels_np, axis=-1)
        return np.mean(preds == truths)

    test_correct = 0
    test_total   = 0
    n_test_batches = len(x_test) // BATCH_SIZE
    for tb in range(n_test_batches):
        s = tb * BATCH_SIZE
        e = s + BATCH_SIZE

        X = x_torch_train[s:e]
        Y = y_torch_train[s:e]
        logits = torch_model(X)
        loss = torch_loss(logits, Y)
        
        test_correct += accuracy(logits.detach().cpu().numpy(), y_torch_train[s:e].detach().cpu().numpy()) * BATCH_SIZE
        test_total   += BATCH_SIZE

    print(f"Torch Test acc: {test_correct/test_total*100:.2f}% | ")

with timing("Training torch (no compile)"):
    train_torch()
check_acc_torch()


# compiled torch
torch_model = torch.nn.Sequential(*[
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
]).cuda()
torch_loss = torch.nn.CrossEntropyLoss()
torch_optim = torch.optim.AdamW(torch_model.parameters(), lr=LR, weight_decay=WEIGTH_DECAY)
torch_model = torch.compile(torch_model)

with timing("Training torch (compiled)"):
    train_torch()
check_acc_torch()


