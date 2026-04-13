#!/usr/bin/env python3
"""
Debug: Check what loss values actually look like
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
print("LOSS VERIFICATION TEST")
print("="*70)

# Small batch
batch_size = 4
batch_x = x_train[:batch_size]
batch_y = y_train_oh[:batch_size]

print(f"\nBatch setup:")
print(f"  Input shape: {batch_x.shape}")
print(f"  Label shape: {batch_y.shape}")

# Create nodes
x_flat = batch_x.flatten().astype(np.float32)
y_flat = batch_y.flatten().astype(np.float32)

print(f"\nFlattened:")
print(f"  x_flat shape: {x_flat.shape}, range: [{x_flat.min():.3f}, {x_flat.max():.3f}]")
print(f"  y_flat shape: {y_flat.shape}, range: [{y_flat.min():.3f}, {y_flat.max():.3f}]")

x_node = bobr.make_node([batch_size, 784], x_flat.tolist())
y_target = bobr.make_node([batch_size, 10], y_flat.tolist())

print(f"\nNode creation:")
print(f"  x_node shape: {x_node.shape()}")
print(f"  y_target shape: {y_target.shape()}")

# Forward pass - random network
h1 = bobr.linear(x_node, 128)
h1 = bobr.relu(h1)

h2 = bobr.linear(h1, 64)
h2 = bobr.relu(h2)

logits = bobr.linear(h2, 10)
loss_node = bobr.cross_entropy(logits, y_target)

print(f"\nNetwork forward:")
print(f"  logits shape: {logits.shape()}")
print(f"  loss_node shape: {loss_node.shape()}")

# Create graph
g = bobr.make_model(loss_node)
g.backward()
g.forward()

# Extract loss
print(f"\nExtracting loss data...")
try:
    loss_data = loss_node.data()
    print(f"  loss_data type: {type(loss_data)}")
    print(f"  loss_data shape: {loss_data.shape}")
    print(f"  loss_data dtype: {loss_data.dtype}")
    print(f"  loss_data values: {loss_data}")
    print(f"  loss_data min: {loss_data.min():.6f}")
    print(f"  loss_data max: {loss_data.max():.6f}")
    print(f"  loss_data mean: {loss_data.mean():.6f}")
except Exception as e:
    print(f"  ✗ Error extracting: {e}")
    import traceback
    traceback.print_exc()

# Also try logits
print(f"\nExtract logits to check network output...")
try:
    logits_data = logits.data()
    print(f"  logits shape: {logits_data.shape}")
    print(f"  logits mean: {logits_data.mean():.6f}")
    print(f"  logits std: {logits_data.std():.6f}")
    print(f"  logits range: [{logits_data.min():.3f}, {logits_data.max():.3f}]")
    print(f"  First sample logits: {logits_data[0]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\nExpected loss (untrained random network on 10-class problem):")
print(f"  Theoretical: -log(1/10) = 2.3026")
print(f"  Typical range: 2.0 - 2.5")
print(f"  Loss of 0.01 would indicate EXTREMELY good fit (unrealistic)")

print("="*70)
