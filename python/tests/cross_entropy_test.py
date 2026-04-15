import bobr
import torch 
from torch.nn import functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

def log_sm(logits):
    maxes = bobr.max(logits, -1, True);           
    shifted = bobr.sub(logits, maxes);     
    exp_shifted = bobr.exp(shifted);        
    sum_exp = bobr.sum(exp_shifted, -1, True);     
    log_sum = bobr.log(sum_exp);          
    log_softmax = bobr.sub(shifted, log_sum); 
    log_softmax.label = "sm_out"
    return log_softmax

def cross_entropy(log_softmax, targets):
    prod = bobr.mul(targets, log_softmax)      
    sum_classes = bobr.sum(prod, -1, False)
    sum_all = bobr.sum(sum_classes, 0, False)
    neg_sum_all = bobr.neg(sum_all)
    loss = bobr.div(neg_sum_all, bobr.full([1], batch_size)); 
    loss.label = "loss"
    return loss

def test_againts_torch(logits, y, W):
    bobr_logits = bobr.Node(logits.numpy())
    bobr_y = bobr.Node(y.numpy())
    bobr_W = bobr.Node(W.detach().numpy(), True)

    assert (logits.data == bobr_logits.data).all()
    assert (y.data == bobr_y.data).all()
    assert (W.data == bobr_W.data).all()


    t_1 = logits @ W
    b_1 = bobr_logits @ bobr_W

    torch_out = F.cross_entropy(t_1, y)
    bobr_out = bobr.cross_entropy(b_1, bobr_y)
    bobr.eval(bobr_out, True)
    
    assert (logits.data == bobr_logits.data).all()
    assert (y.data == bobr_y.data).all()
    assert (W.data == bobr_W.data).all()


    # print(f"{torch_out.shape = }")
    # print(f"{bobr_out.shape = }")
    # print(f"{torch_out.data = }")
    # print(f"{bobr_out.data = }")

    assert ((torch_out.item() - bobr_out.data[0]) < 1e-5)

    torch_out.backward()
    print(bobr_W.grad - W.grad.detach().numpy())
    assert (W.grad == bobr_W.grad).all()






x = torch.randn(4, 16, dtype=torch.float32)
y = torch.rand(4, 32, dtype=torch.float32)
W = torch.rand(16, 32, dtype=torch.float32, requires_grad = True)

test_againts_torch(x, y, W)
