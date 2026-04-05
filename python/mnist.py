import numpy as np
import bobr
import torch
import torch.nn.functional as F
from keras.datasets import mnist

# ============ Load MNIST ============
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# flatten and normalize to [0, 1]
train_X = train_X.reshape(-1, 784).astype(np.float32) / 255.0
test_X = test_X.reshape(-1, 784).astype(np.float32) / 255.0

# one-hot encode labels
def one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

train_y_oh = one_hot(train_y)
test_y_oh = one_hot(test_y)

# ============ Config ============
batch_size = 64
lr = 0.01
epochs = 3
in_dim = 784
hidden = 128
out_dim = 10

# ============ Bobr Training ============
print("=" * 50)
print("BOBR TRAINING")
print("=" * 50)

# init weights (small random values via numpy)
np.random.seed(42)
W1_np = (np.random.randn(in_dim, hidden) * 0.01).astype(np.float32)
b1_np = np.zeros((1, hidden), dtype=np.float32)
W2_np = (np.random.randn(hidden, out_dim) * 0.01).astype(np.float32)
b2_np = np.zeros((1, out_dim), dtype=np.float32)

W1 = bobr.Tensor([in_dim, hidden], True); W1.from_numpy(W1_np)
b1 = bobr.Tensor([1, hidden], True); b1.fill(0.0)
W2 = bobr.Tensor([hidden, out_dim], True); W2.from_numpy(W2_np)
b2 = bobr.Tensor([1, out_dim], True); b2.fill(0.0)

params = [W1, b1, W2, b2]

n_batches = train_X.shape[0] // batch_size

for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i in range(n_batches):
        # zero grad
        for p in params:
            p.zero_grad()

        # get batch
        start = i * batch_size
        end = start + batch_size
        x_np = train_X[start:end]
        y_np = train_y_oh[start:end]
        y_labels = train_y[start:end]

        # load batch into tensors
        x = bobr.Tensor([batch_size, in_dim], False)
        x.from_numpy(x_np)
        y = bobr.Tensor([batch_size, out_dim], False)
        y.from_numpy(y_np)

        # forward: h = relu(x @ W1 + b1), out = h @ W2 + b2
        h = (x @ W1 + b1).relu()
        out = h @ W2 + b2

        # MSE loss: loss = sum((out - y)^2) / (batch_size * out_dim)
        # since you don't have sub, use: diff = out + (y * (-1))
        neg_one = bobr.Tensor([1, 1], False); neg_one.fill(-1.0)
        neg_y = y * neg_one
        diff = out + neg_y            # [batch, 10]
        sq = diff * diff              # [batch, 10]
        loss = sq.sum(0).sum(0)       # scalar-ish

        # backward
        loss.build_backward()
        loss.eval()
        loss.geval()

        # get loss value
        loss_val = loss.data.item() / (batch_size * out_dim)
        epoch_loss += loss_val

        # compute accuracy
        out_np = out.data
        preds = np.argmax(out_np, axis=1)
        correct += np.sum(preds == y_labels)
        total += batch_size

        # SGD update
        for p in params:
            g = p.grad
            p_data = p.data
            p_data -= lr * g / (batch_size * out_dim)
            p.from_numpy(p_data)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Acc: {correct/total*100:.1f}%")

# ============ Bobr Test ============
test_batches = test_X.shape[0] // batch_size
correct = 0
total = 0
for i in range(test_batches):
    start = i * batch_size
    end = start + batch_size
    x = bobr.Tensor([batch_size, in_dim], False)
    x.from_numpy(test_X[start:end])

    h = (x @ W1 + b1).relu()
    out = h @ W2 + b2
    out.eval()

    preds = np.argmax(out.data, axis=1)
    correct += np.sum(preds == test_y[start:end])
    total += batch_size

print(f"Bobr Test Accuracy: {correct/total*100:.1f}%")

print()
print("=" * 50)
print("PYTORCH TRAINING")
print("=" * 50)

# ============ PyTorch Training (same architecture, same init) ============
device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(42)
W1_pt = torch.tensor(np.random.randn(in_dim, hidden).astype(np.float32) * 0.01,
                      device=device, requires_grad=True)
b1_pt = torch.zeros(1, hidden, device=device, requires_grad=True)
W2_pt = torch.tensor(np.random.randn(hidden, out_dim).astype(np.float32) * 0.01,
                      device=device, requires_grad=True)
b2_pt = torch.zeros(1, out_dim, device=device, requires_grad=True)

params_pt = [W1_pt, b1_pt, W2_pt, b2_pt]

train_X_t = torch.tensor(train_X, device=device)
train_y_oh_t = torch.tensor(train_y_oh, device=device)
train_y_t = torch.tensor(train_y, device=device)

for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i in range(n_batches):
        for p in params_pt:
            if p.grad is not None:
                p.grad.zero_()

        start = i * batch_size
        end = start + batch_size
        x_t = train_X_t[start:end]
        y_t = train_y_oh_t[start:end]

        h_t = (x_t @ W1_pt + b1_pt).relu()
        out_t = h_t @ W2_pt + b2_pt
        diff_t = out_t - y_t
        loss_t = (diff_t * diff_t).sum()

        loss_t.backward()

        loss_val = loss_t.item() / (batch_size * out_dim)
        epoch_loss += loss_val

        preds = out_t.argmax(dim=1)
        correct += (preds == train_y_t[start:end]).sum().item()
        total += batch_size

        with torch.no_grad():
            for p in params_pt:
                p -= lr * p.grad / (batch_size * out_dim)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Acc: {correct/total*100:.1f}%")

# ============ PyTorch Test ============
test_X_t = torch.tensor(test_X, device=device)
with torch.no_grad():
    h_t = (test_X_t @ W1_pt + b1_pt).relu()
    out_t = h_t @ W2_pt + b2_pt
    preds = out_t.argmax(dim=1)
    correct = (preds == torch.tensor(test_y, device=device)).sum().item()
    print(f"PyTorch Test Accuracy: {correct/test_X.shape[0]*100:.1f}%")