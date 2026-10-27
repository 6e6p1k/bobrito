"""
MNIST training using the Module API.
Compares the old manual parameter style with the new Linear/Sequential.
"""
import sys
sys.path.insert(0, 'python')
import bobrito as bobr
import numpy as np
import keras

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
model = keras.models.Sequential([
    keras.layers.Dense(128, input_shape=(784, ), activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(lr),
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
)

