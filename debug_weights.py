#!/usr/bin/env python3
"""
Debug: Check if network weights are actually being updated
"""

import sys
sys.path.insert(0, 'python')
import bobrito as bobr
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import mnist
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

def one_hot(y, num_classes=10):
    n = y.shape[0]
    one_hot_y = np.zeros((n, num_classes), dtype=np.float32)
    one_hot_y[np.arange(n), y] = 1.0
    return one_hot_y

y_train_oh = one_hot(y_train, 10)

print("="*70)
print("WEIGHT UPDATE DEBUG")
print("="*70)

batch_size = 32
batch_x = x_train[:batch_size]
batch_y = y_train_oh[:batch_size]

print(f"\nBatch size: {batch_size}")

# Create nodes
x_flat = batch_x.flatten().astype(np.float32)
y_flat = batch_y.flatten().astype(np.float32)

x_node = bobr.make_node([batch_size, 784], x_flat.tolist())
y_target = bobr.make_node([batch_size, 10], y_flat.tolist())

# Build network
h1 = bobr.linear(x_node, 128)
h1.set_label("h1")
h1 = bobr.relu(h1)

h2 = bobr.linear(h1, 64)
h2.set_label("h2")
h2 = bobr.relu(h2)

logits = bobr.linear(h2, 10)
logits.set_label("logits")

loss_node = bobr.cross_entropy(logits, y_target)
loss_node.set_label("loss")

print("\nNetwork built")

# Create graph
g = bobr.Graph(loss_node, disposable=False)
g.backward()

print("Graph created, backward initialized")

# Manual weight tracking - Compare logits before and after training
print("\n" + "="*70)
print("TRAINING WITH WEIGHT MONITORING")
print("="*70)

logits_before = None
learning_rate = 0.1

for iter_idx in range(3):
    print(f"\nIteration {iter_idx + 1}:")
    print("-"*70)
    
    # Get logits BEFORE forward/step
    try:
        logits_data_before = logits.data().copy()
        logits_sum_before = float(np.sum(np.abs(logits_data_before)))
        print(f"  Logits before forward: sum(abs)={logits_sum_before:.6f}")
    except:
        print(f"  Logits before forward: <could not extract>")
    
    # Forward + backward
    g.forward()
    
    # Get loss
    loss_data = loss_node.data()
    batch_loss = float(np.mean(loss_data))
    print(f"  Loss: {batch_loss:.6f}")
    
    # Take SGD step
    print(f"  Taking SGD step (lr={learning_rate})...")
    g.step(learning_rate)
    
    # Get logits AFTER step to see if they changed
    try:
        logits_data_after = logits.data().copy()
        logits_sum_after = float(np.sum(np.abs(logits_data_after)))
        print(f"  Logits after step:    sum(abs)={logits_sum_after:.6f}")
        
        # Check if logits actually changed
        logits_change = logits_sum_after - logits_sum_before
        if abs(logits_change) < 1e-6:
            print(f"  ✗ WARNING: Logits didn't change! (Δ={logits_change:.9f})")
        else:
            print(f"  ✓ Logits changed by {logits_change:+.6f}")
            
    except Exception as e:
        print(f"  Error extracting logits: {e}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
If logits don't change between iterations, the weights are NOT being updated.
Possible causes:
  1. SGD step not actually updating weights
  2. Weights are in a detached part of the graph
  3. Graph structure issue
  4. Learning rate is zero or not applied
""")
print("="*70)
