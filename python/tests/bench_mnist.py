"""
MNIST benchmark: bobr vs PyTorch eager vs PyTorch compiled.

Fairness principles:
  - All data pre-loaded to GPU once, before timing starts.
  - No internal syncs during the timed loop. Single sync at the end.
  - Warmup iterations before every timed region (cuBLAS init, lazy alloc, etc.).
  - Matched workload: identical architecture, batch size, epoch count.
  - Fresh model weights at the start of each run.
  - Same loss target format for all (class indices for torch, one-hot for bobr
    since bobr's cross_entropy takes one-hot — this is a bobr implementation
    detail, not a fairness issue).
"""

import sys, time
sys.path.insert(0, 'python')
import numpy as np
import torch
import bobr
torch.set_float32_matmul_precision('high')  # enables TF32

# ─── Data ──────────────────────────────────────────────────────────────

def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32).reshape(-1, 784) / 255.0
    x_test  = x_test.astype(np.float32).reshape(-1, 784) / 255.0

    def one_hot(y, c=10):
        oh = np.zeros((len(y), c), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    return x_train, one_hot(y_train), y_train.astype(np.int64), \
           x_test,  one_hot(y_test),  y_test.astype(np.int64)


x_train_np, y_train_oh_np, y_train_idx_np, \
x_test_np,  y_test_oh_np,  y_test_idx_np = load_mnist()

# Preload everything to GPU once.
x_train_gpu     = torch.from_numpy(x_train_np).cuda()
y_train_oh_gpu  = torch.from_numpy(y_train_oh_np).cuda()
y_train_idx_gpu = torch.from_numpy(y_train_idx_np).cuda()
x_test_gpu      = torch.from_numpy(x_test_np).cuda()
y_test_idx_gpu  = torch.from_numpy(y_test_idx_np).cuda()


# ─── Config ────────────────────────────────────────────────────────────

BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LR           = 1e-3          # actually trains, unlike 1e-5
WEIGHT_DECAY = 0.0
N_WARMUP     = 20
NUM_BATCHES  = len(x_train_np) // BATCH_SIZE


# ─── Bobr setup + step ────────────────────────────────────────────────

def build_bobr():
    model = bobr.Sequential()
    model.add(bobr.LinearRelu(784, 128))
    model.add(bobr.LinearRelu(128, 64))
    model.add(bobr.Linear(64, 10))

    # Source nodes — data gets stolen in each step.
    x = bobr.from_numpy(x_train_np[:BATCH_SIZE]); x.label = "X"
    y = bobr.from_numpy(y_train_oh_np[:BATCH_SIZE]); y.label = "Y"

    logits = model.forward(x); logits.label = "logits"
    loss   = bobr.cross_entropy(logits, y); loss.label = "loss"

    g = bobr.make_model(loss)
    g.build_backward()

    optim = bobr.AdamW(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return model, g, optim, x, y, logits


def bobr_step(g, optim, x, y, batch_idx):
    s = batch_idx * BATCH_SIZE
    e = s + BATCH_SIZE
    # steal_data does a device-to-device copy (no H→D round trip)
    # since x_train_gpu is already on CUDA.
    x.data_from_cuda(x_train_gpu[s:e])
    y.data_from_cuda(y_train_oh_gpu[s:e])

    g.fwd_bwd_pass()
    optim.step()


# ─── Torch setup + step ───────────────────────────────────────────────

def build_torch(compile_model=False):
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    ).cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optim   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if compile_model:
        model = torch.compile(model)

    return model, loss_fn, optim


def torch_step(model, loss_fn, optim, batch_idx):
    s = batch_idx * BATCH_SIZE
    e = s + BATCH_SIZE
    X = x_train_gpu[s:e]
    Y = y_train_idx_gpu[s:e]   # class indices, fast CE path

    optim.zero_grad(set_to_none=True)
    logits = model(X)
    loss = loss_fn(logits, Y)
    loss.backward()
    optim.step()


# ─── Timing harness ────────────────────────────────────────────────────

def bench(name, step_fn, n_warmup=N_WARMUP, n_iters=NUM_EPOCHS * NUM_BATCHES):
    # Warmup — first iterations are always slower (lazy init, cache effects,
    # compilation for torch.compile). Sync at the end of warmup so we don't
    # inherit queued work.
    for i in range(n_warmup):
        step_fn(i % NUM_BATCHES)
    torch.cuda.synchronize()

    # Timed region: no syncs inside, single sync at end for wall-clock.
    t0 = time.perf_counter()
    for i in range(n_iters):
        step_fn(i % NUM_BATCHES)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    iter_ms  = total_ms / n_iters
    print(f"{name:40s}  {total_ms:8.1f}ms total   {iter_ms:7.3f}ms/iter")
    return total_ms


# ─── Accuracy check ────────────────────────────────────────────────────

def check_acc_bobr(x_src, logits, g):
    correct = total = 0
    n = len(x_test_np) // BATCH_SIZE
    for i in range(n):
        s = i * BATCH_SIZE
        e = s + BATCH_SIZE
        x_src.data_from_cuda(x_test_gpu[s:e])
        g.fwd()
        torch.cuda.synchronize()
        preds = np.argmax(np.array(logits.data), axis=-1)
        correct += (preds == y_test_idx_np[s:e]).sum()
        total += BATCH_SIZE
    return 100.0 * correct / total


def check_acc_torch(model):
    model.eval()
    correct = total = 0
    n = len(x_test_np) // BATCH_SIZE
    with torch.no_grad():
        for i in range(n):
            s = i * BATCH_SIZE
            e = s + BATCH_SIZE
            logits = model(x_test_gpu[s:e])
            preds = logits.argmax(dim=-1)
            correct += (preds == y_test_idx_gpu[s:e]).sum().item()
            total  += BATCH_SIZE
    model.train()
    return 100.0 * correct / total


# ─── Run ──────────────────────────────────────────────────────────────

print(f"MNIST MLP 784-128-64-10  |  batch={BATCH_SIZE}  "
      f"epochs={NUM_EPOCHS}  lr={LR}  warmup={N_WARMUP}")
print("─" * 80)

# Bobr
bobr_model, g, bobr_optim, x_src, y_src, logits = build_bobr()
t_bobr = bench(
    "bobr (AdamW)",
    lambda i: bobr_step(g, bobr_optim, x_src, y_src, i),
)
acc_bobr = check_acc_bobr(x_src, logits, g)
print(f"   bobr test acc: {acc_bobr:.2f}%")
print()

# Torch eager (fresh weights)
torch_model, torch_loss, torch_optim = build_torch(compile_model=False)
t_torch = bench(
    "torch eager (AdamW)",
    lambda i: torch_step(torch_model, torch_loss, torch_optim, i),
)
acc_torch = check_acc_torch(torch_model)
print(f"   torch eager test acc: {acc_torch:.2f}%")
print()

# Torch compiled (fresh weights)
torch_model_c, torch_loss_c, torch_optim_c = build_torch(compile_model=True)
t_torch_c = bench(
    "torch compile (AdamW)",
    lambda i: torch_step(torch_model_c, torch_loss_c, torch_optim_c, i),
)
acc_torch_c = check_acc_torch(torch_model_c)
print(f"   torch compile test acc: {acc_torch_c:.2f}%")
print()

# Summary
print("─" * 80)
print(f"bobr vs torch eager:    {t_torch / t_bobr:.2f}x")
print(f"bobr vs torch compile:  {t_torch_c / t_bobr:.2f}x")