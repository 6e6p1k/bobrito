"""
MNIST comparison: bobr (eager and CUDA Graph) vs PyTorch eager, a hand-captured
torch CUDA Graph, and torch.compile (default and reduce-overhead, i.e. with
CUDA Graphs).

Trains the same 784-128-64-10 MLP with AdamW in each framework and prints
per-epoch progress plus a summary table (test accuracy, train time, speed).

Fairness (same as bench_mnist.py):
  - All data pre-loaded to GPU once, before timing starts.
  - Warmup iterations before every timed region.
  - No syncs inside an epoch; one sync at the end of each epoch.
  - Test-accuracy evaluation runs outside the timed region.

Run from the repo root:
    .venv/bin/python python/tests/mnist_compare.py
    .venv/bin/python python/tests/mnist_compare.py --epochs 20 --no-compile
"""

import argparse, os, sys, time
sys.path.insert(0, 'python')
import numpy as np
import torch
import bobrito as bobr
torch.set_float32_matmul_precision('high')  # enables TF32, like bobr's cuBLAS mode

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--no-compile', action='store_true', help='skip both torch.compile modes')
parser.add_argument('--data', default=os.path.expanduser('~/.keras/datasets/mnist.npz'))
args = parser.parse_args()


# ─── Data ──────────────────────────────────────────────────────────────

def load_mnist(path):
    if not os.path.exists(path):
        sys.exit(f"MNIST not found at {path} (pass --data path/to/mnist.npz)")
    d = np.load(path)

    def one_hot(y, c=10):
        oh = np.zeros((len(y), c), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    x_train = d['x_train'].astype(np.float32).reshape(-1, 784) / 255.0
    x_test  = d['x_test'].astype(np.float32).reshape(-1, 784) / 255.0
    return x_train, one_hot(d['y_train']), d['y_train'].astype(np.int64), \
           x_test,  d['y_test'].astype(np.int64)


x_train_np, y_train_oh_np, y_train_idx_np, x_test_np, y_test_idx_np = load_mnist(args.data)

x_train_gpu     = torch.from_numpy(x_train_np).cuda()
y_train_oh_gpu  = torch.from_numpy(y_train_oh_np).cuda()
y_train_idx_gpu = torch.from_numpy(y_train_idx_np).cuda()
x_test_gpu      = torch.from_numpy(x_test_np).cuda()
y_test_idx_gpu  = torch.from_numpy(y_test_idx_np).cuda()

B           = args.batch
NUM_BATCHES = len(x_train_np) // B
NUM_TEST    = len(x_test_np) // B


# ─── bobr ──────────────────────────────────────────────────────────────

class Bobr:
    name = "bobr"

    def __init__(self):
        model = bobr.Sequential()
        model.add(bobr.LinearRelu(784, 128))
        model.add(bobr.LinearRelu(128, 64))
        model.add(bobr.Linear(64, 10))

        self.x = bobr.from_numpy(x_train_np[:B])
        self.y = bobr.from_numpy(y_train_oh_np[:B])
        self.logits = model.forward(self.x)
        loss = bobr.cross_entropy(self.logits, self.y)

        self.g = bobr.make_model(loss)
        self.g.build_backward()
        self.optim = bobr.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.0)

    def step(self, i):
        s = i * B
        self.x.data_from_cuda(x_train_gpu[s:s + B])
        self.y.data_from_cuda(y_train_oh_gpu[s:s + B])
        self.g.fwd_bwd_pass()
        self.optim.step()

    def test_acc(self):
        correct = 0
        for i in range(NUM_TEST):
            s = i * B
            self.x.data_from_cuda(x_test_gpu[s:s + B])
            self.g.fwd()
            preds = np.argmax(np.array(self.logits.data), axis=-1)
            correct += (preds == y_test_idx_np[s:s + B]).sum()
        return 100.0 * correct / (NUM_TEST * B)


class BobrGraphed(Bobr):
    """bobr with the whole step (fwd, bwd, AdamW) run through g.train_step(opt),
    which captures it into one CUDA Graph and replays it."""
    name = "bobr graph"

    def __init__(self):
        super().__init__()
        # 1st call runs eagerly (allocates every buffer, inits cuBLAS/cub),
        # 2nd captures + replays, later calls just replay; recaptures if a buffer moves
        self.train_step = self.g.train_step(self.optim)

    def step(self, i):
        s = i * B
        self.x.data_from_cuda(x_train_gpu[s:s + B])   # copies stay outside the graph
        self.y.data_from_cuda(y_train_oh_gpu[s:s + B])
        self.train_step()                             # one launch for the whole step


# ─── torch ─────────────────────────────────────────────────────────────

class Torch:
    NAMES = {None: "torch eager", "default": "torch compile", "reduce-overhead": "torch compile RO"}

    def __init__(self, compile_mode=None):
        # compile_mode: None = eager, "default" = Inductor fusion only,
        # "reduce-overhead" = Inductor fusion + CUDA Graphs
        self.name = self.NAMES[compile_mode]
        self.cudagraphs = compile_mode == "reduce-overhead"
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64),  torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        ).cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.0)
        if compile_mode is not None:
            self.model = torch.compile(self.model, mode=compile_mode)

    def step(self, i):
        s = i * B
        if self.cudagraphs:
            # tells the CUDA Graph manager a new iteration starts, so it may
            # overwrite the previous replay's outputs
            torch.compiler.cudagraph_mark_step_begin()
        self.optim.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(x_train_gpu[s:s + B]), y_train_idx_gpu[s:s + B])
        loss.backward()
        self.optim.step()

    def test_acc(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(NUM_TEST):
                s = i * B
                preds = self.model(x_test_gpu[s:s + B]).argmax(dim=-1)
                correct += (preds == y_test_idx_gpu[s:s + B]).sum().item()
        self.model.train()
        return 100.0 * correct / (NUM_TEST * B)


class TorchGraphed(Torch):
    """Eager torch with the WHOLE step (forward, loss, backward, AdamW) captured
    into one CUDA Graph by hand (torch.cuda.graph). No fusion, so this is the
    closest analogue of bobr's train_step."""

    def __init__(self):
        super().__init__()
        self.name = "torch full graph"
        # capturable=True keeps AdamW's step counter on the GPU, so it isn't
        # frozen into the graph as a constant
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                       weight_decay=0.0, capturable=True)
        # the graph reads its inputs from these fixed buffers
        self.static_x = x_train_gpu[:B].clone()
        self.static_y = y_train_idx_gpu[:B].clone()

        # warm up on a side stream (required before capture: lazy init, workspaces)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._eager_step()
        torch.cuda.current_stream().wait_stream(side)

        # capture once; grads set to None so backward allocates them inside the graph
        self.graph = torch.cuda.CUDAGraph()
        self.optim.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            loss = self.loss_fn(self.model(self.static_x), self.static_y)
            loss.backward()
            self.optim.step()

    def _eager_step(self):
        self.optim.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(self.static_x), self.static_y)
        loss.backward()
        self.optim.step()

    def step(self, i):
        s = i * B
        self.static_x.copy_(x_train_gpu[s:s + B])   # copies stay outside the graph
        self.static_y.copy_(y_train_idx_gpu[s:s + B])
        self.graph.replay()                          # one launch for the whole step


# ─── Harness ───────────────────────────────────────────────────────────

def run(fw):
    print(f"\n▶ {fw.name}")
    t0 = time.perf_counter()
    for i in range(args.warmup):
        fw.step(i % NUM_BATCHES)
    torch.cuda.synchronize()  # device-wide, so it also waits for bobr's stream
    print(f"  warmup: {args.warmup} iters in {(time.perf_counter() - t0) * 1000:.0f} ms")

    train_ms = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        for i in range(NUM_BATCHES):
            fw.step(i)
        torch.cuda.synchronize()
        epoch_ms = (time.perf_counter() - t0) * 1000
        train_ms += epoch_ms

        acc = fw.test_acc()  # outside the timed region
        print(f"  epoch {epoch:3d}/{args.epochs}   {epoch_ms:8.1f} ms   "
              f"{epoch_ms / NUM_BATCHES:6.3f} ms/iter   test acc {acc:6.2f}%")

    return {"name": fw.name, "acc": acc, "train_ms": train_ms,
            "iter_ms": train_ms / (args.epochs * NUM_BATCHES)}


print(f"GPU: {torch.cuda.get_device_name()}   torch {torch.__version__}")
print(f"MNIST MLP 784-128-64-10 | AdamW lr={args.lr} | batch={B} | "
      f"epochs={args.epochs} | {NUM_BATCHES} iters/epoch | warmup={args.warmup}")

results = [run(Bobr()), run(BobrGraphed()), run(Torch()), run(TorchGraphed())]
if not args.no_compile:
    results.append(run(Torch("default")))
    results.append(run(Torch("reduce-overhead")))

bobr_ms = results[0]["train_ms"]
print("\n" + "─" * 75)
print(f"{'framework':<18}{'test acc':>10}{'train time':>14}{'ms/iter':>10}{'bobr speedup':>16}")
print("─" * 75)
for r in results:
    speedup = "—" if r["name"] == "bobr" else f"{r['train_ms'] / bobr_ms:.2f}x"
    print(f"{r['name']:<18}{r['acc']:>9.2f}%{r['train_ms'] / 1000:>13.2f}s"
          f"{r['iter_ms']:>10.3f}{speedup:>16}")
print("─" * 75)
print("bobr speedup = that framework's train time / bobr's train time (>1 means bobr is faster)")
