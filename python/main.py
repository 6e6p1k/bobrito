import bobr
import torch
import time

# ============ Config ============
batch = 64
in_dim = 784
hidden = 256
out_dim = 10
n_warmup = 20
n_runs = 10000

# ============ Bobr ============
x = bobr.Tensor([batch, in_dim], True); x.fill(0.5)
W1 = bobr.Tensor([in_dim, hidden], True); W1.fill(0.01)
b1 = bobr.Tensor([1, hidden], True); b1.fill(0.0)
W2 = bobr.Tensor([hidden, out_dim], True); W2.fill(0.01)
b2 = bobr.Tensor([1, out_dim], True); b2.fill(0.0)

h = (x @ W1 + b1).relu()
out = h @ W2 + b2
loss = out.sum(0)
loss.build_backward()
def bobr_forward_backward():
    loss.eval()
    loss.geval()
    loss.sync()

# warmup
for _ in range(n_warmup):
    bobr_forward_backward()

start = time.perf_counter()
for _ in range(n_runs):
    bobr_forward_backward()
end = time.perf_counter()
bobr_us = (end - start) / n_runs * 1e6
print(f"Bobr:    {bobr_us:.1f} μs per forward+backward")

# ============ PyTorch ============
device = "cuda" if torch.cuda.is_available() else "cpu"
x_t = torch.full((batch, in_dim), 0.5, device=device, requires_grad=True)
W1_t = torch.full((in_dim, hidden), 0.01, device=device, requires_grad=True)
b1_t = torch.full((1, hidden), 0.0, device=device, requires_grad=True)
W2_t = torch.full((hidden, out_dim), 0.01, device=device, requires_grad=True)
b2_t = torch.full((1, out_dim), 0.0, device=device, requires_grad=True)

def torch_forward_backward():
    h = (x_t @ W1_t + b1_t).relu()
    out = h @ W2_t + b2_t
    loss = out.sum()
    loss.backward()

# warmup
for _ in range(n_warmup):
    torch_forward_backward()
    torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_runs):
    torch_forward_backward()
torch.cuda.synchronize()
end = time.perf_counter()
torch_us = (end - start) / n_runs * 1e6
print(f"PyTorch: {torch_us:.1f} μs per forward+backward")
print(f"Ratio:   {bobr_us / torch_us:.2f}x")

# ============ PyTorch Compile ============
# device = "cuda" if torch.cuda.is_available() else "cpu"
# x_t = torch.full((batch, in_dim), 0.5, device=device, requires_grad=True)
# W1_t = torch.full((in_dim, hidden), 0.01, device=device, requires_grad=True)
# b1_t = torch.full((1, hidden), 0.0, device=device, requires_grad=True)
# W2_t = torch.full((hidden, out_dim), 0.01, device=device, requires_grad=True)
# b2_t = torch.full((1, out_dim), 0.0, device=device, requires_grad=True)
# class net(torch.nn.module):
#     def __init__(self, hidden):
#         super().__init__()
#         w1 = torch.Parameter(torch.full((batch, in_dim), 0.5))
#         w2 = torch.nn.linear(torch.full((hidden, out_dim), 0.01))
#         b1 = torch.nn.linear(torch.full((1, hidden), 0.0))
#         b2 = torch.nn.linear( torch.full((1, out_dim), 0.0))
#     def forward(self, x):
#         h = (x_t @ W1_t + b1_t)
#         out = torch.nn.ReLU(h) @ w2 + b2
#         return out

# net.Compile()
# def torch_compile_forward_backward():
#     loss = net(x_t)
#     loss.backward()

# # warmup
# for _ in range(n_warmup):
#     torch_compile_forward_backward()
#     torch.cuda.synchronize()

# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(n_runs):
#     torch_compile_forward_backward()
# torch.cuda.synchronize()
# end = time.perf_counter()
# torch_us = (end - start) / n_runs * 1e6
# print(f"PyTorch: {torch_us:.1f} μs per forward+backward")
# print(f"Ratio:   {bobr_us / torch_us:.2f}x")
device = "cuda" if torch.cuda.is_available() else "cpu"
x_t = torch.full((batch, in_dim), 0.5, device=device, requires_grad=True)
W1_t = torch.full((in_dim, hidden), 0.01, device=device, requires_grad=True)
b1_t = torch.full((1, hidden), 0.0, device=device, requires_grad=True)
W2_t = torch.full((hidden, out_dim), 0.01, device=device, requires_grad=True)
b2_t = torch.full((1, out_dim), 0.0, device=device, requires_grad=True)

torch_compile = torch.compile(torch_forward_backward)

# warmup
for _ in range(n_warmup):
    torch_compile()
    torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_runs):
    torch_compile()
torch.cuda.synchronize()
end = time.perf_counter()
torch_us = (end - start) / n_runs * 1e6
print(f"PyTorch Compile: {torch_us:.1f} μs per forward+backward")
print(f"Ratio:   {bobr_us / torch_us:.2f}x")


# VERIFY VALUES ARE CORRECT

def check_value(t1, t2):
    correct = all((t1.data == t2.clone().detach().to("cpu")).tolist())
    return correct

def verify_values():
    for i, (t1, t2) in enumerate([
        (W1, W1_t), (b1, b1_t), (W2, W2_t), (b2, b2_t)
    ]):
        print(f"Parameter {i}: {'correct' if check_value(t1, t2) else 'incorrect'}")

print()
verify_values()