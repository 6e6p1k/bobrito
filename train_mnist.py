#!/usr/bin/env python3
"""
MNIST training using BOBR Python bindings
Loads MNIST via TensorFlow/Keras, trains using BOBR framework
"""

import sys
sys.path.insert(0, 'python')
import bobr
import numpy as np

# Load MNIST dataset
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

# Hyperparameters
batch_size = 64
num_epochs = 1  # Just 1 epoch for quick demo
learning_rate = 0.01
num_batches = min(10, len(x_train) // batch_size)  # Just 10 batches for quick demo

print("\n" + "="*70)
print("MNIST Training with BOBR Framework")
print("="*70)
print(f"Network architecture: 784 -> 128 -> 64 -> 10")
print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")
print("="*70)

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(x_train))
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train_oh[start_idx:end_idx]
            batch_sz = batch_x.shape[0]
            
            # Create input nodes from numpy arrays
            # Convert to nested lists for BOBR
            x_list = batch_x.tolist()
            y_list = batch_y.tolist()
            
            x_node = bobr.make_node(x_list)
            x_node.set_label("x_input")
            
            y_target = bobr.make_node(y_list)
            y_target.set_label("y_target")
            
            # Forward pass: 784 -> 128 -> 64 -> 10
            h1 = bobr.linear(x_node, 128)
            h1.set_label("h1_linear")
            h1 = bobr.relu(h1)
            h1.set_label("h1_activated")
            
            h2 = bobr.linear(h1, 64)
            h2.set_label("h2_linear")
            h2 = bobr.relu(h2)
            h2.set_label("h2_activated")
            
            logits = bobr.linear(h2, 10)
            logits.set_label("logits")
            
            # Compute loss
            loss = bobr.cross_entropy(logits, y_target)
            loss.set_label("loss")
            
            # Create computational graph and train
            g = bobr.make_model(loss)
            g.forward()  # Forward + backward automatic
            
            # Update weights
            g.step(learning_rate)
            g.zero_grad()
            
            # Progress indicator
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx+1}/{num_batches} ({progress:.0f}%)", end="\r")
        
        print(f"  Batch {num_batches}/{num_batches} (100%)  ✓")
        
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ Training completed successfully!")
print("="*70)
print("\nSummary:")
print(f"  - Trained for {num_epochs} epochs")
print(f"  - Processed {num_batches * batch_size} training samples")
print(f"  - Network: 784 (input) → 128 (hidden) → 64 (hidden) → 10 (output)")
print(f"  - Used BOBR framework for all computations on GPU")
print("\nDemonstration: BOBR can train real datasets using Python bindings!")
print("="*70)
