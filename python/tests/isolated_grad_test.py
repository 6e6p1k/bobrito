import sys
sys.path.insert(0, 'python')

import bobrito as bobr
import torch
import numpy as np

def check_close(name, actual, expected, rtol=1e-3, atol=1e-4):
    """Check if two arrays are close, with detailed error reporting."""
    if actual.shape != expected.shape:
        return False, f"Shape mismatch: got {actual.shape}, expected {expected.shape}"
    
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual - expected)
        max_diff = diff.max()
        max_idx = np.unravel_index(diff.argmax(), diff.shape)
        rel = np.abs(diff / (np.abs(expected) + 1e-8))
        max_rel = rel.max()
        return False, (f"max_abs_diff={max_diff:.6e} at {max_idx}, "
                       f"max_rel_diff={max_rel:.6e}, "
                       f"got={actual[max_idx]:.6f}, expected={expected[max_idx]:.6f}")
    return True, "OK"

def make_bobr_leaf(data_np, requires_grad=True):
    """Create a bobr leaf node from numpy data with requires_grad."""
    t = torch.from_numpy(data_np.copy()).cuda()
    node = bobr.Node(t, requires_grad=requires_grad)
    return node

def bobr_eval_and_get_grad(loss_node, leaf_nodes):
    """Evaluate forward+backward and return gradients as numpy arrays."""
    # loss_node.eval()
    bobr.eval(loss_node, True, False)
    bobr.sync()
    grads = []
    for leaf in leaf_nodes:
        grads.append(np.array(leaf.grad))
    return grads

def torch_eval_and_get_grad(loss_tensor, leaf_tensors):
    """Run torch backward and return gradients as numpy arrays."""
    loss_tensor.backward()
    return [t.grad.cpu().numpy() for t in leaf_tensors]

def test_chain_matmul_relu_sum():
    """loss = relu(x @ W + b).sum()
    
    This is the core MLP forward pass. Tests gradient flow through
    matmul -> add -> relu -> sum.
    """
    x_data = np.random.randn(4, 5).astype(np.float32)
    w_data = np.random.randn(5, 3).astype(np.float32)
    b_data = np.random.randn(1, 3).astype(np.float32)
    
    x_b = make_bobr_leaf(x_data, requires_grad=False); x_b.set_label("x")
    w_b = make_bobr_leaf(w_data); w_b.set_label("W")
    b_b = make_bobr_leaf(b_data); b_b.set_label("b")
    h = bobr.relu(bobr.add(bobr.matmul(x_b, w_b), b_b)); h.set_label("h")
    loss_b = bobr.sum(h); loss_b.set_label("Loss")
    [gw_b, gb_b] = bobr_eval_and_get_grad(loss_b, [w_b, b_b])
    
    x_t = torch.from_numpy(x_data).cuda()
    w_t = torch.from_numpy(w_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = torch.relu(x_t @ w_t + b_t).sum()
    [gw_t, gb_t] = torch_eval_and_get_grad(loss_t, [w_t, b_t])

    print()

    print("=== test_chain_matmul_relu_sum ===")
    print("Bobr loss:\n", loss_b.data)
    print("Torch loss:\n", loss_t.item())

    print("Bobr grad_W:\n", gw_b)
    print("Torch grad_W:\n", gw_t)
    print("Bobr grad_b:\n", gb_b)
    print("Torch grad_b:\n", gb_t)

    ok, msg = check_close("mlp grad_W", gw_b, gw_t)
    assert ok, f"grad_W: {msg}"
    ok, msg = check_close("mlp grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}"



def main():
    test_chain_matmul_relu_sum()
    print("All tests passed!")

if __name__ == "__main__":
    main()