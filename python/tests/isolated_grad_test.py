import bobr
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

def bobr_eval_and_get_grad(loss_node, leaf_nodes):
    """Evaluate forward+backward and return gradients as numpy arrays."""
    loss_node.eval()
    # bobr.eval(loss_node, True)
    bobr.sync()
    grads = []
    for leaf in leaf_nodes:
        # print(f"    [debug] leaf '{leaf.label}' is_grad_buffer={leaf.is_grad_buffer}, is_grad_node={leaf.is_grad_node}")
        grads.append(np.array(leaf.grad))
    return grads

def torch_eval_and_get_grad(loss_tensor, leaf_tensors):
    """Run torch backward and return gradients as numpy arrays."""
    loss_tensor.backward()
    return [t.grad.cpu().numpy() for t in leaf_tensors]

def test_mul_grad():
    bobr.mem_snapshot()
    """(a * b).sum() -> grad_a = b, grad_b = a"""
    a_data = np.random.randn(3, 4).astype(np.float32)
    b_data = np.random.randn(3, 4).astype(np.float32)
    
    a_b = bobr.Node(a_data, True)
    b_b = bobr.Node(b_data, True)

    loss_b = bobr.sum(bobr.mul(a_b, b_b), 0, False)
    # [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    loss_b.eval()
    bobr.sync()
    ga_b = np.array(a_b.grad)
    gb_b = np.array(b_b.grad)

    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t * b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    ok, msg = check_close("mul grad_a", ga_b, ga_t)
    assert ok, f"grad_a: {msg}"
    ok, msg = check_close("mul grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}"


def test_div_grad():
    # print("[debug] Starting div grad test")
    bobr.mem_snapshot()
    """(a / b).sum() -> grad_a = 1/b, grad_b = -a/b^2"""
    np.random.seed(42)
    a_data = np.random.uniform(0.5, 2.0, (3, 4)).astype(np.float32)
    b_data = np.random.uniform(0.5, 2.0, (3, 4)).astype(np.float32)
    
    a_b = bobr.Node(a_data, True); a_b.label = "A"
    b_b = bobr.Node(b_data, True); b_b.label = "B"

    print("DATA BEFORE EVAL:")
    print(a_b.data)
    print(b_b.data)

    loss_b = bobr.sum(bobr.div(a_b, b_b), 0, False); loss_b.label = "Loss"

    # [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    bobr.eval(loss_b, True)
    bobr.sync()
    ga_b = np.array(a_b.grad)
    gb_b = np.array(b_b.grad)

    ga_b = np.array(a_b.grad)
    gb_b = np.array(b_b.grad)

    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t / b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    print(f"    [debug] div grad_a bobr={ga_b.flatten()}, \n\ttorch={ga_t.flatten()}"
          f"\n    [debug] div grad_b bobr={gb_b.flatten()}, \n\ttorch={gb_t.flatten()}")

    # print(a_b.is_owned_grad)
    # print(b_b.is_owned_grad)
    # print(f"OWNED_GRAD_A_ptr = {a_b.get_owned_grad_ptr()}")
    # print(f"OWNED_GRAD_B_ptr = {a_b.get_owned_grad_ptr()}")

    ok, msg = check_close("div grad_a", ga_b, ga_t)
    assert ok, f"grad_a: {msg}"
    ok, msg = check_close("div grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}"



def main():
    test_mul_grad()
    test_div_grad()
    print("All tests passed!")

if __name__ == "__main__":
    main()