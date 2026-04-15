import sys
sys.path.insert(0, 'python')

import bobr
import torch
import numpy as np

def test_add_broadcast_then_sum():
    """Test: sum(broadcast_add)"""
    a_data = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)  # (2, 3)
    b_data = np.array([[10., 20., 30.]], dtype=np.float32)  # (1, 3)
    
    # Bobr
    a_b = bobr.Node(torch.from_numpy(a_data).cuda(), requires_grad=True)
    b_b = bobr.Node(torch.from_numpy(b_data).cuda(), requires_grad=True)
    c_b = bobr.add(a_b, b_b)
    loss_b = bobr.sum(c_b)
    bobr.eval(loss_b)
    bobr.sync()
    grad_b_b = np.array(b_b.grad)
    
    # Torch
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t + b_t).sum()
    loss_t.backward()
    grad_b_t = b_t.grad.cpu().numpy()
    
    print("Input a (2,3):")
    print(a_data)
    print("\nInput b (1,3):")
    print(b_data)
    print("\nBobr loss:", loss_b.data)
    print("Torch loss:", loss_t.item())
    print("\nBobr grad_b:")
    print(grad_b_b)
    print("Torch grad_b:")
    print(grad_b_t)
    print("\nMatch:", np.allclose(grad_b_b, grad_b_t))

if __name__ == "__main__":
    test_add_broadcast_then_sum()
