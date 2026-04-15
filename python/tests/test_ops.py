import torch
import bobr

SHAPE = [32, 64, 16]
a = torch.rand(SHAPE)
b = torch.rand(SHAPE)

def compute_elemwise_torch(a, b, f):
    return f(a, b)

def compute_elemwise_bobr(a, b, f):
    _a = bobr.Node(a.numpy())
    _b = bobr.Node(b.numpy())
    c =  f(_a, _b)
    c.eval()
    return c

c_torch = compute_elemwise_torch(a, b, torch.add)
c_bobr = compute_elemwise_bobr(a, b, bobr.add)

print((c_torch.data == c_bobr.data).all())