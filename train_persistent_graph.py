#!/usr/bin/env python3
"""
MNIST Training - Persistent Graph Approach
Create graph ONCE, reuse across iterations
"""

import sys
sys.path.insert(0, 'python')
import bobr
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
print("MNIST TRAINING - PERSISTENT GRAPH")
print("="*70)

# Use ONE batch and train on it repeatedly
batch_size = 256
num_iters = 50
learning_rate = 0.1  # Large learning rate for these small losses

batch_x = x_train[:batch_size]
batch_y = y_train_oh[:batch_size]

print(f"Batch size: {batch_size}")
print(f"Iterations: {num_iters}")
print(f"Learning rate: {learning_rate}")
print("Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)")
print("="*70)

# CREATE NODES ONCE
print("\n[1/4] Creating input nodes...")
x_flat = batch_x.flatten().astype(np.float32)
y_flat = batch_y.flatten().astype(np.float32)

x_node = bobr.make_node([batch_size, 784], x_flat.tolist())
y_target = bobr.make_node([batch_size, 10], y_flat.tolist())
print(f"  x_node shape: {x_node.shape()}")
print(f"  y_target shape: {y_target.shape()}")

# BUILD NETWORK ONCE
print("\n[2/4] Building network...")
h1 = bobr.linear(x_node, 128)
h1 = bobr.relu(h1)

h2 = bobr.linear(h1, 64)
h2 = bobr.relu(h2)

logits = bobr.linear(h2, 10)
loss_node = bobr.cross_entropy(logits, y_target)
print(f"  Network built: 784 → 128 → 64 → 10")
print(f"  Loss node shape: {loss_node.shape()}")

# CREATE GRAPH ONCE (non-disposable)
print("\n[3/4] Creating computational graph (disposable=False)...")
g = bobr.Graph(loss_node, disposable=False)
print(f"  Graph created with disposable=False")

# Initialize backward computation (identifies which nodes need gradients)
g.backward()
print(f"  Backward computation initialized")

# TRAINING LOOP - JUST EVAL AND STEP
print("\n[4/4] Training loop...")
print("-"*70)

loss_history = []

try:
    for iter_idx in range(num_iters):
        # Run forward + backward
        g.forward()
        
        # Get loss
        loss_data = loss_node.data()
        batch_loss = float(np.mean(loss_data))
        loss_history.append(batch_loss)
        
        # SGD step
        g.step(learning_rate)
        
        if (iter_idx + 1) % max(1, num_iters // 10) == 0 or iter_idx == 0:
            print(f"Iter {iter_idx+1:3d}/{num_iters}: loss = {batch_loss:.6f}")
    
    print(f"Iter {num_iters:3d}/{num_iters}: loss = {loss_history[-1]:.6f}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TRAINING RESULTS")
print("="*70)

initial_loss = loss_history[0]
final_loss = loss_history[-1]
change = initial_loss - final_loss
pct_change = (change / initial_loss * 100) if initial_loss > 0 else 0

print(f"\nInitial loss (iter 1): {initial_loss:.6f}")
print(f"Final loss (iter {num_iters}):   {final_loss:.6f}")
print(f"Loss reduction:        {change:+.6f} ({pct_change:+.1f}%)")

print(f"\nLoss trajectory:")
print(f"  Min: {min(loss_history):.6f} at iteration {np.argmin(loss_history)+1}")
print(f"  Max: {max(loss_history):.6f}")
print(f"  Mean: {np.mean(loss_history):.6f}")

if final_loss < initial_loss:
    print(f"\n✓ SUCCESS: Network learning! Loss decreased by {pct_change:.1f}%")
else:
    print(f"\n✗ PROBLEM: Loss increased (or stayed same). Weights not updating properly.")

print("\n" + "="*70)
