import sys
sys.path.insert(0, 'python')

import bobr
import torch
import numpy as np

def test_reshape_contiguous():
    """Test if reshape + contiguous properly reorders data"""
    # Create a simple (4, 3) tensor
    data = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
        [10., 11., 12.]
    ], dtype=np.float32)
    
    print("Original (4, 3):")
    print(data)
    
    tensor = bobr.Node(torch.from_numpy(data).cuda(), requires_grad=False)
    
    # Reshape to (3, 4) - conceptually moves dim 0 to the end
    reshaped = bobr.reshape(tensor, [3, 4])
    print(f"\nAfter reshape to (3, 4):")
    
    # Make contiguous
    contiguous_tensor = bobr.contiguous(reshaped)
    bobr.eval(contiguous_tensor)
    bobr.sync()
    
    result = np.array(contiguous_tensor.data)
    print(result)
    print(f"Shape: {result.shape}")
    
    # For comparison, what PyTorch does
    torch_data = torch.from_numpy(data).cuda()
    torch_reshaped = torch_data.reshape(3, 4)
    print(f"\nPyTorch reshape to (3, 4):")
    print(torch_reshaped.cpu().numpy())
    
    # What we actually want (transpose)
    print(f"\nPyTorch transpose (swap dims):")
    print(torch_data.t().cpu().numpy())

if __name__ == "__main__":
    test_reshape_contiguous()
