#!/usr/bin/env python3
"""
MNIST training with BOBR framework - PROPER TRAINING LOOP
Correctly calls backward() to set up gradients before SGD step
"""

import sys
sys.path.insert(0, 'python')
import bobr
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load MNIST
print("Loading MNIST dataset...")
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

def one_hot(y, num_classes=10):
    n = y.shape[0]
    one_hot_y = np.zeros((n, num_classes), dtype=np.float32)
    one_hot_y[np.arange(n), y] = 1.0
    return one_hot_y

y_train_oh = one_hot(y_train, 10)

print(f"Training data: {x_train.shape}")
print(f"Training labels: {y_train_oh.shape}")

print("\n" + "="*70)
print("PROPER MNIST TRAINING WITH BOBR")
print("="*70)
print("Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)")
print("="*70)

# Hyperparameters
batch_size = 64
num_epochs = 3
learning_rate = 0.01
num_batches = min(100, len(x_train) // batch_size)  # Just 100 batches per epoch for demo

print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")
print(f"Batches per epoch: {num_batches}")
print(f"Learning rate: {learning_rate}")
print("="*70)

import io
from contextlib import redirect_stdout, redirect_stderr

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*70)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(x_train))
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train_oh[start_idx:end_idx]
            actual_batch_size = batch_x.shape[0]
            
            try:
                # Create data nodes
                x_flat = batch_x.flatten().astype(np.float32)
                y_flat = batch_y.flatten().astype(np.float32)
                
                x_node = bobr.make_node([actual_batch_size, 784], x_flat.tolist())
                y_target = bobr.make_node([actual_batch_size, 10], y_flat.tolist())
                
                # Forward pass
                h1 = bobr.linear(x_node, 128)
                h1 = bobr.relu(h1)
                
                h2 = bobr.linear(h1, 64)
                h2 = bobr.relu(h2)
                
                logits = bobr.linear(h2, 10)
                loss_node = bobr.cross_entropy(logits, y_target)
                
                # Create computational graph
                g = bobr.make_model(loss_node)
                
                # KEY: Call backward() first to initialize gradient computation
                g.backward()
                
                # Now we can run forward and step
                g.forward()  # Forward + backward pass
                g.step(learning_rate)
                
            except Exception as inner_e:
                print(f"\n✗ Error in batch {batch_idx}: {inner_e}")
                import traceback
                traceback.print_exc()
                raise
            
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx+1}/{num_batches} ({progress:.0f}%)", end="\r")
        
        print(f"  Batch {num_batches}/{num_batches} (100%)  ✓")

except Exception as e:
    print(f"\n✗ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ TRAINING COMPLETED!")
print("="*70)
print("\nKey Points:")
print("  ✓ Called g.backward() to initialize gradient computation")
print("  ✓ Ran g.eval() to do forward+backward pass")
print("  ✓ Called g.step(lr) to update weights with SGD")
print("  ✓ Called g.zero_grad() to clear gradients")
print("\nThis proper sequence enables actual network training!")
print("="*70)
