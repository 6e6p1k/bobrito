"""
MNIST training using the Module API.
Compares the old manual parameter style with the new Linear/Sequential.
"""
import sys
sys.path.insert(0, 'python')
import bobrito as bobr
import numpy as np

# ── Dataset ───────────────────────────────────────────────────

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


def accuracy(logits_np, labels_np):
    preds  = np.argmax(logits_np, axis=-1)
    truths = np.argmax(labels_np, axis=-1)
    return np.mean(preds == truths)


# ── Training ──────────────────────────────────────────────────

x_train, y_train, x_test, y_test = load_mnist()
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

batch_size  = 64
num_epochs  = 5
lr          = 1e-3
num_batches = len(x_train) // batch_size

# Build model using Module API
l1 = bobr.Linear(784, 128)
l2 = bobr.Linear(128, 64)
l3 = bobr.Linear(64, 10)

# Placeholder leaf nodes for data
x = bobr.from_numpy(x_train[:batch_size]); x.label = "X"
y = bobr.from_numpy(y_train[:batch_size]); y.label = "Y"

# Build computation graph (static, built once)
h1     = bobr.relu(l1.forward(x))
h2     = bobr.relu(l2.forward(h1))
logits = l3.forward(h2); logits.label = "logits"
loss   = bobr.cross_entropy(logits, y); loss.label = "loss"

# Create persistent graph (disposable=False)
g = bobr.make_model(loss)
g.build_backward()

for epoch in range(num_epochs):
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx in range(1, num_batches):
        s = batch_idx * batch_size
        e = s + batch_size

        x.data_from_numpy(x_train[s:e])
        y.data_from_numpy(y_train[s:e])

        bobr.sync()
        g.fwd_bwd_pass()
        bobr.sync()

        total_loss += float(loss.data.squeeze())
        correct += accuracy(np.array(logits.data), y_train[s:e]) * batch_size
        total   += batch_size

        g.sgd_step(lr)
        # g.clear_graph_data()

    # Test set eval
    test_correct = 0
    test_total   = 0
    n_test_batches = len(x_test) // batch_size
    for tb in range(n_test_batches):
        s = tb * batch_size
        e = s + batch_size
        x.data_from_numpy(x_test[s:e])
        y.data_from_numpy(y_test[s:e])
        bobr.sync()
        g.fwd_bwd_pass()  # fwd only would be nicer, but works
        bobr.sync()
        test_correct += accuracy(np.array(logits.data), y_test[s:e]) * batch_size
        test_total   += batch_size
        # g.clear_graph_data()

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train acc: {correct/total*100:.2f}% | "
          f"Test acc: {test_correct/test_total*100:.2f}% | "
          f"Loss: {total_loss/num_batches:.4f}")
