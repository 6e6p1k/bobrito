import numpy as np
import bobr
from keras.datasets import mnist
import gc

# ============ Load MNIST ============
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(-1, 784).astype(np.float32) / 255.0
test_X = test_X.reshape(-1, 784).astype(np.float32) / 255.0

def one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

train_y_oh = one_hot(train_y)

# ============ Config ============
batch_size = 64
epochs = 3
in_dim = 784
hidden = 128
out_dim = 10

# ============ Init weights ============
np.random.seed(42)

W1 = bobr.Tensor(np.random.randn(in_dim, hidden).astype(np.float32) * 0.01, True)
b1 = bobr.zeros([1, hidden], True)
W2 = bobr.Tensor(np.random.randn(hidden, out_dim).astype(np.float32) * 0.01, True)
b2 = bobr.zeros([1, out_dim], True)

# for idx, param in enumerate([W1, b1, W2, b2]):
#     print(f"Parameter {idx}: requires_grad={param.requires_grad}")

opt = bobr.SGD([W1, b1, W2, b2], lr=0.01)

n_batches = train_X.shape[0] // batch_size

# ============ Train ============
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i in range(n_batches):
        opt.zero_grad()

        start = i * batch_size
        end = start + batch_size
        x = bobr.Tensor(train_X[start:end])
        y = bobr.Tensor(train_y_oh[start:end])

        # forward
        h = (x @ W1 + b1).relu()
        out = h @ W2 + b2

        # MSE loss
        neg_one = bobr.full([1, 1], -1.0)
        diff = out + y * neg_one
        sq = diff * diff
        scale = bobr.full([1, 1], 1.0 / (batch_size * out_dim))
        loss = sq.sum(0).sum(0) * scale

        # backward
        loss.build_backward()
        loss.eval()
        loss.geval()

        # update
        opt.step()

        # logging
        loss_val = loss.data.item()
        epoch_loss += loss_val

        out_np = out.data
        preds = np.argmax(out_np, axis=1)
        correct += np.sum(preds == train_y[start:end])
        total += batch_size
        
        # bobr.clear_grads(loss)
        if i % 100 == 0:
            gc.collect()
            print(f"batch {i}: {bobr.memdata()}")
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Acc: {correct/total*100:.1f}%")
    


# ============ Test ============
test_batches = test_X.shape[0] // batch_size
correct = 0
total = 0
for i in range(test_batches):
    start = i * batch_size
    end = start + batch_size
    x = bobr.Tensor(test_X[start:end])

    h = (x @ W1 + b1).relu()
    out = h @ W2 + b2
    out.eval()

    preds = np.argmax(out.data, axis=1)
    correct += np.sum(preds == test_y[start:end])
    total += batch_size

print(f"Test Accuracy: {correct/total*100:.1f}%")