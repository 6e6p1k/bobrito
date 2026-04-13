#!/usr/bin/env python3
"""
MNIST Training - Corrected Loss Averaging
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
print("MNIST TRAINING - CORRECTED LOSS TRACKING")
print("="*70)

batch_size = 128
num_epochs = 3
learning_rate = 0.01
batches_per_epoch = 5

print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")
print(f"Batches per epoch: {batches_per_epoch}")
print(f"Learning rate: {learning_rate}")
print("="*70)

loss_history = []

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*70)
        
        epoch_losses = []
        
        for batch_idx in range(batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(x_train))
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train_oh[start_idx:end_idx]
            actual_batch_size = batch_x.shape[0]
            
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
            g.backward()
            g.forward()
            
            # Extract loss - loss_node.data() returns array of shape [actual_batch_size]
            loss_data = loss_node.data()
            
            # Mean loss across batch
            batch_loss = float(np.mean(loss_data))
            epoch_losses.append(batch_loss)
            
            print(f"  Batch {batch_idx+1}/{batches_per_epoch}: loss={batch_loss:.4f}, " + 
                  f"shape={loss_data.shape}, values={loss_data}")
            
            # Update weights
            g.step(learning_rate)
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print(f"\n  Epoch {epoch+1} average loss: {avg_loss:.6f}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TRAINING RESULTS")
print("="*70)

for epoch, loss in enumerate(loss_history, 1):
    print(f"Epoch {epoch}: Loss = {loss:.6f}")

if len(loss_history) > 1:
    initial = loss_history[0]
    final = loss_history[-1]
    change = initial - final
    pct = (change / initial * 100) if initial > 0 else 0
    
    print(f"\nInitial: {initial:.6f}")
    print(f"Final:   {final:.6f}")
    print(f"Change:  {change:+.6f} ({pct:+.1f}%)")

print("="*70)
