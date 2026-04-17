# bobr-framework-cuda

bobr is a from-scratch CUDA autograd engine with a static lazy graph. the design trades off eager-mode flexibility for lower per-iteration overhead, making it faster than PyTorch on small models where dispatch cost dominates compute. on a 784→128→64→10 MLP with batch 64, it's 1.71x faster than torch eager and 2.14x faster than torch compile, with matched accuracy


bench_mnist.py trains a 784-128-64-10 MLP on MNIST with AdamW, batch 64, 50 epochs, matched architecture and precision (fp32, no TF32 for either framework). On RTX Blackwell (sm_120):

bobr: 11.6s, 0.247 ms/iter, 96.9% test acc
torch eager: 19.8s, 0.422 ms/iter, 97.6% test acc
torch compile: 24.8s, 0.529 ms/iter, 97.7% test acc

1.71x faster than torch eager, 2.14x faster than torch compile.