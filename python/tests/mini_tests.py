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
    return x_train, x_test, y_train, y_test

def log_sm(logits):
    maxes = bobr.max(logits, -1, True);           
    shifted = bobr.sub(logits, maxes);     
    exp_shifted = bobr.exp(shifted);        
    sum_exp = bobr.sum(exp_shifted, -1, True);     
    log_sum = bobr.log(sum_exp);          
    log_softmax = bobr.sub(shifted, log_sum); 
    log_softmax.label = "sm_out"
    return log_softmax

def cross_entropy(log_softmax, targets):
    prod = bobr.mul(targets, log_softmax)      
    sum_classes = bobr.sum(prod, -1, False)
    sum_all = bobr.sum(sum_classes, 0, False)
    neg_sum_all = bobr.neg(sum_all)
    loss = bobr.div(neg_sum_all, bobr.full([1], batch_size)); 
    loss.label = "loss"
    return loss

def define_model(x, y):
    # Forward pass: 784 -> 128 -> 64 -> 10
    h1 = bobr.linear(x, 128)
    h1.label = "h1_linear"
    h1 = bobr.relu(h1)
    h1.label = "h1_activated"
    
    h2 = bobr.linear(h1, 64)
    h2.label = "h2_linear"
    h2 = bobr.relu(h2)
    h2.label = "h2_activated"
    
    logits = bobr.linear(h2, 10)
    logits.label = "logits"
    
    # Compute loss
    loss = bobr.cross_entropy(logits, y)
    loss.label = "loss"
    

    g = bobr.make_model(loss)
    return logits, loss, g



x_train, x_test, y_train_oh, y_test_oh = load_dataset()

# Hyperparameters
batch_size = 64
num_epochs = 15
learning_rate = 1e-3
num_batches = min(10, len(x_train) // batch_size)  # Just 10 batches for quick demo



try:
    x = bobr.normal([batch_size, 784]); x.label = "X"
    y = bobr.normal([batch_size, 1]);  y.label = "Y"
    logits, loss, g = define_model(x, y)
    g.build_backward()
    # print(loss)
    # print(g.weights)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(x_train))
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = np.expand_dims(y_train_oh[start_idx:end_idx], -1)
            batch_y = batch_y.astype(np.float32)
            batch_sz = batch_x.shape[0]
            
            # Create input nodes from numpy arrays
            # Convert to nested lists for BOBR
            
            # print(f"{batch_x.shape = }")
            # print(f"{batch_y.shape = }")
            # print(f"{x.shape = }")
            # print(f"{y.shape = }")


            x.steal_data(batch_x)
            y.steal_data(batch_y)

            # bobr.eval(loss, True)
            g.eval(False)
            g.sgd_step(learning_rate)

            # print(f"Targets: {y.data[0] = }")
            # print(f"Proper Targets: {batch_y[0] = }")
            # print(f"Logits: {logits.data[0] = }")
            # print(f"Loss: {loss.data = }")
            # Progress indicator
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx+1}/{num_batches} ({progress:.0f}%)", end="\r")
                print(f"  Loss: {loss.data}")

        print(f"  Batch {num_batches}/{num_batches} (100%)  ✓")
        
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)