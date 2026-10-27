import numpy as np
import bobrito as bobr
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

W1 = bobr.Tensor(np.random.randn(in_dim, hidden).astype(np.float32) * 0.01, True);  W1.label = "W1"
b1 = bobr.zeros([1, hidden], True);                                                 b1.label = "b1"
W2 = bobr.Tensor(np.random.randn(hidden, out_dim).astype(np.float32) * 0.01, True); W2.label = "W2"
b2 = bobr.zeros([1, out_dim], True);                                                b2.label = "b2"

# for idx, param in enumerate([W1, b1, W2, b2]):
#     print(f"Parameter {idx}: requires_grad={param.requires_grad}")

opt = bobr.SGD([W1, b1, W2, b2], lr=0.01)

n_batches = train_X.shape[0] // batch_size


# model definition --------------------------------

x = bobr.full([batch_size, in_dim], 0.5)
y = bobr.full([batch_size, out_dim], 0.23)

# forward
h = (x @ W1 + b1).relu()
out = h @ W2 + b2 # [64, 10]

# softmax
maxes = out.max(-1, keepdims=True)  # [64, 1]
shifted = out - maxes # [64, 10]
exp_out = shifted.exp() # [64, 10]
sm_out = exp_out * (exp_out.sum(-1, keepdims=True) ** -1) # [64, 10] * [64, 1]

# MSE loss
diff = sm_out - y
sq = diff ** 2
scale = bobr.full([1, 1], 1.0 / (batch_size * out_dim)); scale.label = "scale"
loss = sq.sum(0).sum(0) * scale; loss.label = "loss"
loss.build_backward()

# ============ Train ============
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i in range(n_batches):
        opt.zero_grad()


        start = i * batch_size
        end = start + batch_size
        x = bobr.Tensor(train_X[start:end]);        x.label = "X"
        y = bobr.Tensor(train_y_oh[start:end]);     y.label = "Y"

        # bobr.draw_graph(loss)
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
        break
    break
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

