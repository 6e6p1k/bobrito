import sys
sys.path.insert(0, 'python')

import bobr
import torch
import numpy as np

def test_reduce_to_shape_simple():
    """Test reduce_to_shape with simple data to understand the permutation issue"""
    # Create a simple (4, 3) gradient 
    grad_data = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
        [10., 11., 12.]
    ], dtype=np.float32)
    
    print("Input grad shape (4, 3):")
    print(grad_data)
    
    # Create nodes
    grad_node = bobr.Node(torch.from_numpy(grad_data).cuda(), requires_grad=False)
    
    # Sum dimension 0 with keepdims to reduce from (4,3) to (1,3)
    result = bobr.sum(grad_node, dim=0, keepdims=True)
    bobr.eval(result)
    bobr.sync()
    
    result_data = np.array(result.data)
    print(f"\nResult shape: {result_data.shape}")
    print("Result (should be [1, 3] with sums [1+4+7+10, 2+5+8+11, 3+6+9+12]):")
    print(result_data)
    print(f"Expected: [[22. 26. 30.]]")
    
    # Compare with PyTorch
    grad_torch = torch.from_numpy(grad_data).cuda()
    result_torch = grad_torch.sum(dim=0, keepdim=True)
    print(f"\nTorch result:")
    print(result_torch.cpu().numpy())

if __name__ == "__main__":
    test_reduce_to_shape_simple()
