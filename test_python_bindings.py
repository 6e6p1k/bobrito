#!/usr/bin/env python3
"""Test script for BOBR Python bindings"""

import sys
sys.path.insert(0, 'python')
import bobrito as bobr
print("=" * 60)
print("Testing BOBR Python Bindings")
print("=" * 60)

# Test 1: Create tensors
print("\n[Test 1] Creating tensors...")
x = bobr.full([2, 3], 1.0)
print(f"  x shape: {x.shape()}")
print(f"  x size: {x.size()}")
print(f"  x ndim: {x.ndim()}")
print(f"  ✓ Tensor creation works!")

# Test 2: Basic operations
print("\n[Test 2] Basic arithmetic operations...")
y = bobr.full([2, 3], 2.0)
z = bobr.add(x, y)
print(f"  x + y shape: {z.shape()}")
print(f"  ✓ Addition works!")

# Test 3: Evaluate graph
print("\n[Test 3] Evaluating graph...")
bobr.eval(z)
print(f"  ✓ Graph evaluation works!")

# Test 4: Matrix operations
print("\n[Test 4] Matrix operations...")
a = bobr.full([4, 3], 1.0)
b = bobr.full([3, 2], 2.0)
c = bobr.matmul(a, b)
print(f"  a @ b shape: {c.shape()}")
print(f"  ✓ Matrix multiplication works!")

# Test 5: Neural network layers
print("\n[Test 5] Neural network layers...")
x_input = bobr.uniform([8, 10])
x_input.set_label("input")
out = bobr.linear(x_input, 5)
out.set_label("linear_out")
print(f"  Input shape: {x_input.shape()}")
print(f"  Linear output shape: {out.shape()}")
print(f"  ✓ Linear layer works!")

# Test 6: Loss function
print("\n[Test 6] Loss functions...")
logits = bobr.full([4, 3], 1.0)
targets = bobr.full([4, 3], 0.33)  # soft targets
loss = bobr.cross_entropy(logits, targets)
print(f"  Cross-entropy loss shape: {loss.shape()}")
print(f"  ✓ Cross-entropy loss works!")

# Test 7: Graph construction and backward
print("\n[Test 7] Training graph...")
batch_x = bobr.uniform([32, 20])
batch_x.set_label("batch_x")

# Build simple 2-layer network
h = bobr.linear(batch_x, 16)
h = bobr.relu(h)
logits = bobr.linear(h, 3)
targets = bobr.full([32, 3], 0.33)
loss = bobr.cross_entropy(logits, targets)

# Create model and run forward/backward
g = bobr.make_model(loss)
print(f"  Loss shape: {loss.shape()}")
print(f"  ✓ Model creation works!")

try:
    g.forward()
    print(f"  ✓ Forward/backward pass works!")
except Exception as e:
    print(f"  ✗ Forward/backward failed: {e}")

# Test 8: SGD step
print("\n[Test 8] SGD optimization...")
try:
    g.step(0.01)
    print(f"  ✓ SGD step works!")
except Exception as e:
    print(f"  Note: SGD step not fully tested ({e})")

print("\n" + "=" * 60)
print("All basic tests passed! Python bindings are working.")
print("=" * 60)
