import sys
sys.path.insert(0, 'python')
import bobr
import numpy as np
from torch.nn import functional as F

# Load MNIST dataset
def load_dataset():
    print("Loading MNIST dataset...")
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Flatten images from (28, 28) to (784,)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # One-hot encode labels
    def one_hot(y, num_classes=10):
        n = y.shape[0]
        one_hot_y = np.zeros((n, num_classes), dtype=np.float32)
        one_hot_y[np.arange(n), y] = 1.0
        return one_hot_y

    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    print(f"✓ Loaded MNIST training data: {x_train.shape}, labels: {y_train_oh.shape}")
    print(f"✓ Loaded MNIST test data: {x_test.shape}, labels: {y_test_oh.shape}")
    for arr in [x_train, x_test, y_train, y_test]:
        arr = arr.astype(np.float32)
    return x_train, x_test, y_train_oh, y_test_oh

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def linear(x, W, b):
    return bobr.add(bobr.matmul(x, W), b)

def make_parameter(shape, name):
    param = bobr.uniform(shape)
    param.requires_grad = True
    param.set_label(name)
    return param

def make_W(shape, name):
    size = np.prod(shape)
    std = np.sqrt(2.0 / shape[0])  # fan_in only
    W = bobr.from_numpy(np.random.normal(0, std, size).astype(np.float32).reshape(shape), True)
    W.label = name
    assert W.requires_grad == True
    return W

def make_b(shape, name):
    b = bobr.from_numpy(np.zeros((shape), dtype=np.float32), True)
    b.label = name
    assert b.requires_grad == True
    return b

x_train, x_test, y_train, y_test = load_dataset()

# Hyperparameters
batch_size = 128
num_epochs = 5
learning_rate = 1e-3
num_batches = len(x_train) // batch_size  # Just 10 batches for quick demo

try:
    # Forward pass: 784 -> 128 -> 64 -> 10
    batch_idx = 1
    x = bobr.from_numpy(x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]); x.label = "X"
    y = bobr.from_numpy(y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]);  y.label = "Y"

    W_1_shape = [784, 128]
    b_1_shape = [1, 128]

    W_2_shape = [128, 64]
    b_2_shape = [1, 64]

    W_3_shape = [64, 10]
    b_3_shape = [1, 10]

    parameters = [
        make_W(W_1_shape, "W_1"),
        make_b(b_1_shape, "b_1"),
        make_W(W_2_shape, "W_2"),
        make_b(b_2_shape, "b_2"),
        make_W(W_3_shape, "W_3"),
        make_b(b_3_shape, "b_3"),
    ]

    h1 = bobr.relu(x @ parameters[0] + parameters[1])
    h2 = bobr.relu(h1 @ parameters[2] + parameters[3])
    logits = (h2 @ parameters[4] + parameters[5])
    logits.label = "logits"

    # Compute loss (cross_entropy already divides by batch size)
    loss = bobr.cross_entropy(logits, y)
    loss.label = "loss"

    g = bobr.make_model(loss)
    g.build_backward()
    bobr.export_graph(loss)

    right = 0
    total = 0
    total_loss = 0
    for epoch in range(num_epochs):
        right = 0
        total = 0
        total_loss = 0
        for batch_idx in range(1, num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(x_train))
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            x.data_from_numpy(batch_x.astype(np.float32))
            y.data_from_numpy(batch_y.astype(np.float32))

            bobr.sync()
            g.fwd_bwd_pass()
            bobr.sync()

            # print(f"batch loss: {loss.data}")
            # print(f"batch loss: {float(loss.data.squeeze()):.4f}")
            # print(f"batch loss/64: {float(loss.data.squeeze()) / 64:.4f}")
            
            total_loss += float(loss.data.squeeze())
            np_logits = np.array(logits.data)
            preds = np.argmax(softmax(np_logits), axis=-1)
            labels = np.argmax(batch_y, axis=-1)
            right += np.sum(preds == labels)
            total += batch_size

            bobr.sync()
            # check if gradients are non-zero
            # W1_grad = np.array(parameters[0].grad)
            # print(f"W1 grad norm: {np.linalg.norm(W1_grad):.6f}")

            g.sgd_step(1e-3)

            # g.clear_graph_data()
            # W1_after = np.array(parameters[0].data)
            # print(f"W1 data norm after step: {np.linalg.norm(W1_after):.6f}")
            # break
        # break
            # print(f"Batch {batch_idx+1:3d}/{num_batches} | "
            #     f"Accuracy = {right / total * 100:.5}% | "
            #     f"Loss: {total_loss / (num_epochs * (epoch + 1))}")

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
                f"Accuracy = {right / total * 100:.5}% | "
                f"Loss: {total_loss / batch_idx}")

        
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)