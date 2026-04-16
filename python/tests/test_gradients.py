"""
BOBR Framework - Gradient Correctness Tests
============================================
Each test:
  1. Builds a computation graph with a scalar loss = op(...).sum()
  2. Runs bobr backward to get analytical gradients
  3. Compares against PyTorch gradients (analytical)
  4. Optionally compares against numerical gradients (finite differences)

Key differences from previous tests:
  - ALWAYS sets requires_grad on bobr leaf nodes
  - Uses fixed random seed for reproducibility
  - Reports actual vs expected values on failure
  - Tests both forward AND backward for every op
  - Numerical gradient checking as independent ground truth
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import torch
import bobr

np.random.seed(42)
torch.manual_seed(42)

# ── Helpers ────────────────────────────────────────────────────

def grad_buffer_node_info(nodes):
    for node in nodes:
        print(f"is_grad_buffer: {node.is_grad_buffer}")
        print(f"is_grad_node: {node.is_grad_node}")


def make_bobr_leaf(data_np, requires_grad=True):
    """Create a bobr leaf node from numpy data with requires_grad."""
    node = bobr.from_numpy(data_np, requires_grad)
    return node

def bobr_eval_and_get_grad(loss_node, leaf_nodes):
    """Evaluate forward+backward and return gradients as numpy arrays."""
    loss_node.eval()
    bobr.eval(loss_node, True)
    # bobr.sync()
    grads = []
    for leaf in leaf_nodes:
        # print(f"    [debug] leaf '{leaf.label}' is_grad_buffer={leaf.is_grad_buffer}, is_grad_node={leaf.is_grad_node}")
        grads.append(np.array(leaf.grad))
    return grads

def torch_eval_and_get_grad(loss_tensor, leaf_tensors):
    """Run torch backward and return gradients as numpy arrays."""
    loss_tensor.backward()
    return [t.grad.cpu().numpy() for t in leaf_tensors]

def numerical_gradient(f, x_np, idx=0, eps=1e-4):
    """Compute numerical gradient via central differences.
    
    f: function that takes a list of numpy arrays and returns a scalar loss (numpy float)
    x_np: list of numpy arrays (the inputs)
    idx: which input to differentiate w.r.t.
    """
    grad = np.zeros_like(x_np[idx])
    it = np.nditer(grad, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        old = x_np[idx][ix]
        
        x_np[idx][ix] = old + eps
        loss_plus = f(x_np)
        
        x_np[idx][ix] = old - eps
        loss_minus = f(x_np)
        
        x_np[idx][ix] = old
        grad[ix] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


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


# ── Test Functions ──────────────────────────────────────────────

passed = 0
failed = 0
errors = []

def run_test(name, test_fn):
    global passed, failed, errors
    try:
        test_fn()
        passed += 1
        print(f"  PASS  {name}")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        failed += 1
        errors.append((name, f"EXCEPTION: {e}"))
        print(f"  ERR   {name}: {e}")


# ============================================================
#   UNARY OPS
# ============================================================

def test_neg_grad():
    """neg(x).sum() -> grad = -1 everywhere"""
    data = np.random.randn(3, 4).astype(np.float32)
    
    # bobr
    x_b = make_bobr_leaf(data)
    loss_b = bobr.sum(bobr.neg(x_b), 0, False)
    [grad_b] = bobr_eval_and_get_grad(loss_b, [x_b])
    
    # torch
    x_t = torch.from_numpy(data).cuda().requires_grad_(True)
    loss_t = (-x_t).sum()
    [grad_t] = torch_eval_and_get_grad(loss_t, [x_t])
    
    ok, msg = check_close("neg grad", grad_b, grad_t)
    assert ok, msg

def test_relu_grad():
    """relu(x).sum() -> grad = (x > 0)"""
    data = np.array([[-1.0, 2.0, -0.5], [3.0, -4.0, 0.1]], dtype=np.float32)
    
    x_b = make_bobr_leaf(data)
    loss_b = bobr.sum(bobr.relu(x_b), 0, False)
    [grad_b] = bobr_eval_and_get_grad(loss_b, [x_b])
    
    x_t = torch.from_numpy(data).cuda().requires_grad_(True)
    loss_t = torch.relu(x_t).sum()
    [grad_t] = torch_eval_and_get_grad(loss_t, [x_t])
    
    ok, msg = check_close("relu grad", grad_b, grad_t)
    assert ok, msg

def test_exp_grad():
    """exp(x).sum() -> grad = exp(x)"""
    data = np.random.uniform(-1, 1, (3, 4)).astype(np.float32)
    
    x_b = make_bobr_leaf(data)
    loss_b = bobr.sum(bobr.exp(x_b), 0, False)
    [grad_b] = bobr_eval_and_get_grad(loss_b, [x_b])
    
    x_t = torch.from_numpy(data).cuda().requires_grad_(True)
    loss_t = torch.exp(x_t).sum()
    [grad_t] = torch_eval_and_get_grad(loss_t, [x_t])
    
    ok, msg = check_close("exp grad", grad_b, grad_t)
    assert ok, msg

def test_log_grad():
    """log(x).sum() -> grad = 1/x"""

    data = np.random.uniform(0.5, 3.0, (3, 4)).astype(np.float32)
    
    x_b = make_bobr_leaf(data)
    loss_b = bobr.sum(bobr.log(x_b), 0, False)
    [grad_b] = bobr_eval_and_get_grad(loss_b, [x_b])
    
    x_t = torch.from_numpy(data).cuda().requires_grad_(True)
    loss_t = torch.log(x_t).sum()
    [grad_t] = torch_eval_and_get_grad(loss_t, [x_t])
    
    ok, msg = check_close("log grad", grad_b, grad_t)
    assert ok, msg


# ============================================================
#   BINARY OPS  
# ============================================================

def test_add_grad():
    """(a + b).sum() -> grad_a = 1, grad_b = 1"""

    a_data = np.random.randn(3, 4).astype(np.float32)
    b_data = np.random.randn(3, 4).astype(np.float32)
    
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)

    grad_buffer_node_info([a_b, b_b])

    loss_b = bobr.sum(bobr.add(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t + b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    ok, msg = check_close("add grad_a", ga_b, ga_t)
    assert ok, f"grad_a: {msg}"
    ok, msg = check_close("add grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}"

def test_sub_grad():
    """(a - b).sum() -> grad_a = 1, grad_b = -1
    
    THIS IS THE CRITICAL TEST. If sub's vjp returns (bwd, bwd) instead of 
    (bwd, neg(bwd)), grad_b will be +1 instead of -1.
    """

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float32)
    
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)

    grad_buffer_node_info([a_b, b_b])

    loss_b = bobr.sum(bobr.sub(a_b, b_b), 0, False)

    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t - b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    ok, msg = check_close("sub grad_a", ga_b, ga_t)
    assert ok, f"grad_a: {msg}"
    ok, msg = check_close("sub grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}  (expected all -1s, this is the sub VJP bug)"

def test_mul_grad():

    """(a * b).sum() -> grad_a = b, grad_b = a"""
    a_data = np.random.randn(3, 4).astype(np.float32)
    b_data = np.random.randn(3, 4).astype(np.float32)
    
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)

    grad_buffer_node_info([a_b, b_b])


    loss_b = bobr.sum(bobr.mul(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
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

    """(a / b).sum() -> grad_a = 1/b, grad_b = -a/b^2"""
    a_data = np.random.uniform(0.5, 2.0, (3, 4)).astype(np.float32)
    b_data = np.random.uniform(0.5, 2.0, (3, 4)).astype(np.float32)
    
    a_b = make_bobr_leaf(a_data); a_b.label = "A"
    b_b = make_bobr_leaf(b_data); b_b.label = "B"

    print("DATA BEFORE EVAL:")
    print(a_b.data)
    print(b_b.data)




    grad_buffer_node_info([a_b, b_b])

    loss_b = bobr.sum(bobr.div(a_b, b_b), 0, False); loss_b.label = "Loss"




    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])


    # print("DATA AFTER EVAL:")
    # print(a_b.data)
    # print(b_b.data)

    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t / b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    # print(f"    [debug] div grad_a bobr={ga_b.flatten()}, \n\ttorch={ga_t.flatten()}"
    #       f"\n    [debug] div grad_b bobr={gb_b.flatten()}, \n\ttorch={gb_t.flatten()}")

    # print(a_b.is_owned_grad)
    # print(b_b.is_owned_grad)
    # print(f"OWNED_GRAD_A_ptr = {a_b.get_owned_grad_ptr()}")
    # print(f"OWNED_GRAD_B_ptr = {a_b.get_owned_grad_ptr()}")

    ok, msg = check_close("div grad_a", ga_b, ga_t)
    assert ok, f"grad_a: {msg}"
    ok, msg = check_close("div grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}"



# ============================================================
#   MATMUL
# ============================================================

def test_matmul_grad():
    """(A @ B).sum() -> grad_A = ones @ B^T, grad_B = A^T @ ones"""
    a_data = np.random.randn(4, 5).astype(np.float32)
    b_data = np.random.randn(5, 3).astype(np.float32)
    
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    loss_b = bobr.sum(bobr.matmul(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t @ b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    ok, msg = check_close("matmul grad_a", ga_b, ga_t)
    assert ok, f"grad_A: {msg}"
    ok, msg = check_close("matmul grad_b", gb_b, gb_t)
    assert ok, f"grad_B: {msg}"

def test_matmul_grad_nonsquare():
    """Non-square matmul to catch transposition bugs."""
    a_data = np.random.randn(2, 7).astype(np.float32)
    b_data = np.random.randn(7, 3).astype(np.float32)
    
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    loss_b = bobr.sum(bobr.matmul(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = (a_t @ b_t).sum()
    [ga_t, gb_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t])
    
    ok, msg = check_close("matmul_nonsquare grad_a", ga_b, ga_t)
    assert ok, f"grad_A: {msg}"
    ok, msg = check_close("matmul_nonsquare grad_b", gb_b, gb_t)
    assert ok, f"grad_B: {msg}"


# ============================================================
#   REDUCTIONS
# ============================================================

def test_sum_dim_grad():
    """sum(x, dim=1).sum() -> grad = ones"""
    data = np.random.randn(3, 4, 5).astype(np.float32)
    
    x_b = make_bobr_leaf(data)
    reduced = bobr.sum(x_b, 1, False)
    loss_b = bobr.sum(reduced, 0, False)
    [grad_b] = bobr_eval_and_get_grad(loss_b, [x_b])
    
    x_t = torch.from_numpy(data).cuda().requires_grad_(True)
    loss_t = x_t.sum(dim=1).sum()
    [grad_t] = torch_eval_and_get_grad(loss_t, [x_t])
    
    ok, msg = check_close("sum_dim grad", grad_b, grad_t)
    assert ok, msg


# ============================================================
#   CHAINS (multi-op compositions)
# ============================================================

def test_chain_sub_then_mul():
    """loss = ((a - b) * c).sum()
    
    Tests that sub gradient flows correctly into downstream ops.
    If sub VJP is wrong (returns bwd instead of neg(bwd) for input[1]),
    grad_b will have wrong sign.
    """
    a_data = np.random.randn(3, 4).astype(np.float32)
    b_data = np.random.randn(3, 4).astype(np.float32)
    c_data = np.random.randn(3, 4).astype(np.float32)
    
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    c_b = make_bobr_leaf(c_data)
    loss_b = bobr.sum(bobr.mul(bobr.sub(a_b, b_b), c_b), 0, False)
    [ga_b, gb_b, gc_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b, c_b])
    
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    c_t = torch.from_numpy(c_data).cuda().requires_grad_(True)
    loss_t = ((a_t - b_t) * c_t).sum()
    [ga_t, gb_t, gc_t] = torch_eval_and_get_grad(loss_t, [a_t, b_t, c_t])
    
    ok, msg = check_close("chain_sub_mul grad_a", ga_b, ga_t)
    assert ok, f"grad_a: {msg}"
    ok, msg = check_close("chain_sub_mul grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}  (sub VJP bug would cause sign error here)"
    ok, msg = check_close("chain_sub_mul grad_c", gc_b, gc_t)
    assert ok, f"grad_c: {msg}"

def test_chain_matmul_relu_sum():
    """loss = relu(x @ W + b).sum()
    
    This is the core MLP forward pass. Tests gradient flow through
    matmul -> add -> relu -> sum.
    """
    x_data = np.random.randn(4, 5).astype(np.float32)
    w_data = np.random.randn(5, 3).astype(np.float32)
    b_data = np.random.randn(1, 3).astype(np.float32)
    
    x_b = make_bobr_leaf(x_data, requires_grad=False)
    w_b = make_bobr_leaf(w_data)
    b_b = make_bobr_leaf(b_data)
    h = bobr.relu(bobr.add(bobr.matmul(x_b, w_b), b_b))
    loss_b = bobr.sum(h, 0, False)
    [gw_b, gb_b] = bobr_eval_and_get_grad(loss_b, [w_b, b_b])
    
    x_t = torch.from_numpy(x_data).cuda()
    w_t = torch.from_numpy(w_data).cuda().requires_grad_(True)
    b_t = torch.from_numpy(b_data).cuda().requires_grad_(True)
    loss_t = torch.relu(x_t @ w_t + b_t).sum()
    [gw_t, gb_t] = torch_eval_and_get_grad(loss_t, [w_t, b_t])
    
    ok, msg = check_close("mlp grad_W", gw_b, gw_t)
    assert ok, f"grad_W: {msg}"
    ok, msg = check_close("mlp grad_b", gb_b, gb_t)
    assert ok, f"grad_b: {msg}"

def test_chain_log_softmax():
    """Simplified log-softmax: log(exp(x) / sum(exp(x)))
    
    This mimics the cross-entropy internals. If sub VJP is broken,
    this will fail because log_softmax = shifted - log_sum uses sub.
    """
    data = np.random.randn(4, 5).astype(np.float32)
    
    # bobr: log_softmax via decomposed ops (matches cross_entropy internals)
    x_b = make_bobr_leaf(data)
    maxes = bobr.max(x_b, -1, True)
    shifted = bobr.sub(x_b, maxes)
    exp_shifted = bobr.exp(shifted)
    sum_exp = bobr.sum(exp_shifted, -1, True)
    log_sum = bobr.log(sum_exp)
    log_softmax = bobr.sub(shifted, log_sum)
    loss_b = bobr.sum(log_softmax, 0, False)
    [grad_b] = bobr_eval_and_get_grad(loss_b, [x_b])
    
    # torch
    x_t = torch.from_numpy(data).cuda().requires_grad_(True)
    loss_t = torch.nn.functional.log_softmax(x_t, dim=-1).sum()
    [grad_t] = torch_eval_and_get_grad(loss_t, [x_t])
    
    ok, msg = check_close("log_softmax grad", grad_b, grad_t, rtol=1e-2, atol=1e-3)
    assert ok, msg

def test_shared_input_grad_accumulation():
    """loss = a*a + a  (a used 3 times)
    
    grad_a = 2*a + 1. Tests that count_uses correctly handles
    multi-consumer gradient accumulation.
    """
    a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    a_b = make_bobr_leaf(a_data)
    loss_b = bobr.sum(bobr.add(bobr.mul(a_b, a_b), a_b), 0, False)
    [ga_b] = bobr_eval_and_get_grad(loss_b, [a_b])
    
    a_t = torch.from_numpy(a_data).cuda().requires_grad_(True)
    loss_t = (a_t * a_t + a_t).sum()
    [ga_t] = torch_eval_and_get_grad(loss_t, [a_t])
    
    ok, msg = check_close("shared_input grad", ga_b, ga_t)
    assert ok, msg


# ============================================================
#   NUMERICAL GRADIENT CHECKS (independent ground truth)
# ============================================================

def test_numerical_grad_mul():
    """Verify mul gradient against finite differences."""
    a_data = np.random.randn(2, 3).astype(np.float32)
    b_data = np.random.randn(2, 3).astype(np.float32)
    
    def f(inputs):
        return (inputs[0] * inputs[1]).sum()
    
    # bobr analytical
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    loss_b = bobr.sum(bobr.mul(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    # numerical
    ga_num = numerical_gradient(f, [a_data.copy(), b_data.copy()], idx=0)
    gb_num = numerical_gradient(f, [a_data.copy(), b_data.copy()], idx=1)
    
    ok, msg = check_close("numerical_mul grad_a", ga_b, ga_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_a vs numerical: {msg}"
    ok, msg = check_close("numerical_mul grad_b", gb_b, gb_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_b vs numerical: {msg}"

def test_numerical_grad_sub():
    """Verify sub gradient against finite differences.
    
    This is the smoking gun test: numerical gradient for b in (a-b).sum()
    should be -1 everywhere. If sub VJP is wrong, bobr will say +1.
    """
    a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    
    def f(inputs):
        return (inputs[0] - inputs[1]).sum()
    
    # bobr analytical
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    loss_b = bobr.sum(bobr.sub(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    # numerical
    ga_num = numerical_gradient(f, [a_data.copy(), b_data.copy()], idx=0)
    gb_num = numerical_gradient(f, [a_data.copy(), b_data.copy()], idx=1)
    
    # print(f"    [debug] sub grad_b bobr={gb_b.flatten()}, numerical={gb_num.flatten()}")
    
    ok, msg = check_close("numerical_sub grad_a", ga_b, ga_num)
    assert ok, f"grad_a vs numerical: {msg}"
    ok, msg = check_close("numerical_sub grad_b", gb_b, gb_num)
    assert ok, f"grad_b vs numerical: {msg}  (EXPECTED -1s, sub VJP bug gives +1s)"

def test_numerical_grad_div():
    """Verify div gradient against finite differences."""
    a_data = np.random.uniform(0.5, 2.0, (2, 3)).astype(np.float32)
    b_data = np.random.uniform(0.5, 2.0, (2, 3)).astype(np.float32)
    
    def f(inputs):
        return (inputs[0] / inputs[1]).sum()
    
    # bobr analytical
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    loss_b = bobr.sum(bobr.div(a_b, b_b), 0, False)
    [ga_b, gb_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b])
    
    # numerical
    ga_num = numerical_gradient(f, [a_data.copy(), b_data.copy()], idx=0)
    gb_num = numerical_gradient(f, [a_data.copy(), b_data.copy()], idx=1)
    
    ok, msg = check_close("numerical_div grad_a", ga_b, ga_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_a vs numerical: {msg}"
    ok, msg = check_close("numerical_div grad_b", gb_b, gb_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_b vs numerical: {msg}"

def test_numerical_grad_chain():
    """Numerical check: loss = sum(relu(a @ b + c))"""
    a_data = np.random.randn(3, 4).astype(np.float32)
    b_data = np.random.randn(4, 2).astype(np.float32)
    c_data = np.random.randn(1, 2).astype(np.float32)
    
    def f(inputs):
        return np.maximum(0, inputs[0] @ inputs[1] + inputs[2]).sum()
    
    # bobr
    a_b = make_bobr_leaf(a_data)
    b_b = make_bobr_leaf(b_data)
    c_b = make_bobr_leaf(c_data)
    h = bobr.relu(bobr.add(bobr.matmul(a_b, b_b), c_b))
    loss_b = bobr.sum(h, 0, False)
    [ga_b, gb_b, gc_b] = bobr_eval_and_get_grad(loss_b, [a_b, b_b, c_b])
    
    # numerical
    ga_num = numerical_gradient(f, [a_data.copy(), b_data.copy(), c_data.copy()], idx=0)
    gb_num = numerical_gradient(f, [a_data.copy(), b_data.copy(), c_data.copy()], idx=1)
    gc_num = numerical_gradient(f, [a_data.copy(), b_data.copy(), c_data.copy()], idx=2)
    
    ok, msg = check_close("numerical_chain grad_a", ga_b, ga_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_a vs numerical: {msg}"
    ok, msg = check_close("numerical_chain grad_b", gb_b, gb_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_b vs numerical: {msg}"
    ok, msg = check_close("numerical_chain grad_c", gc_b, gc_num, rtol=1e-2, atol=1e-3)
    assert ok, f"grad_c vs numerical: {msg}"


# ============================================================
#   FORWARD-ONLY SANITY CHECKS
# ============================================================

def test_forward_exp():
    data = np.random.randn(3, 4).astype(np.float32)
    x_b = make_bobr_leaf(data, requires_grad=False)
    y = bobr.exp(x_b)
    y.eval()
    bobr.sync()
    ok, msg = check_close("fwd exp", np.array(y.data), np.exp(data), rtol=1e-5, atol=1e-6)
    assert ok, msg

def test_forward_log():
    data = np.random.uniform(0.1, 5.0, (3, 4)).astype(np.float32)
    x_b = make_bobr_leaf(data, requires_grad=False)
    y = bobr.log(x_b)
    y.eval()
    bobr.sync()
    ok, msg = check_close("fwd log", np.array(y.data), np.log(data), rtol=1e-5, atol=1e-6)
    assert ok, msg

def test_forward_matmul():
    a = np.random.randn(4, 5).astype(np.float32)
    b = np.random.randn(5, 3).astype(np.float32)
    a_b = make_bobr_leaf(a, requires_grad=False)
    b_b = make_bobr_leaf(b, requires_grad=False)
    c = bobr.matmul(a_b, b_b)
    c.eval()
    bobr.sync()
    ok, msg = check_close("fwd matmul", np.array(c.data), a @ b, rtol=1e-4, atol=1e-5)
    assert ok, msg

def test_forward_sub():
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    a_b = make_bobr_leaf(a, requires_grad=False)
    b_b = make_bobr_leaf(b, requires_grad=False)
    c = bobr.sub(a_b, b_b)
    c.eval()
    bobr.sync()
    ok, msg = check_close("fwd sub", np.array(c.data), a - b, rtol=1e-5, atol=1e-6)
    assert ok, msg


# ============================================================
#   CROSS-ENTROPY (end-to-end)
# ============================================================

def test_cross_entropy_forward():
    """Test cross_entropy forward matches PyTorch."""
    logits = np.random.randn(8, 5).astype(np.float32)
    labels = np.zeros((8, 5), dtype=np.float32)
    for i in range(8):
        labels[i, np.random.randint(5)] = 1.0    

    # bobr
    l_b = make_bobr_leaf(logits, requires_grad=False)
    t_b = make_bobr_leaf(labels, requires_grad=False)
    loss_b = bobr.cross_entropy(l_b, t_b)
    loss_b.eval()
    bobr.sync()
    loss_val_b = np.array(loss_b.data).item()
    print(f"{loss_b = }")
    
    # torch
    l_t = torch.from_numpy(logits).cuda()
    # torch cross_entropy expects class indices, not one-hot
    targets_idx = torch.from_numpy(labels.argmax(axis=1)).cuda().long()
    loss_val_t = torch.nn.functional.cross_entropy(l_t, targets_idx).item()
    print(f"{loss_val_t = }")

    diff = abs(loss_val_b - loss_val_t)
    assert diff < 1e-3, f"CE forward: bobr={loss_val_b:.6f}, torch={loss_val_t:.6f}, diff={diff:.6e}"

def test_cross_entropy_backward():
    """Test cross_entropy backward matches PyTorch."""
    logits = np.random.randn(8, 5).astype(np.float32)
    # One-hot labels: must start from zeros so argmax picks the correct class
    labels = np.zeros((8, 5), dtype=np.float32)
    for i in range(8):
        labels[i, np.random.randint(5)] = 1.0

    # bobr
    l_b = make_bobr_leaf(logits)
    t_b = make_bobr_leaf(labels, requires_grad=False)
    loss_b = bobr.cross_entropy(l_b, t_b)
    [gl_b] = bobr_eval_and_get_grad(loss_b, [l_b])

    # torch
    l_t = torch.from_numpy(logits).cuda().requires_grad_(True)
    targets_idx = torch.from_numpy(labels.argmax(axis=1)).cuda().long()
    loss_t = torch.nn.functional.cross_entropy(l_t, targets_idx)
    loss_t.backward()
    gl_t = l_t.grad.cpu().numpy()

    ok, msg = check_close("CE grad_logits", gl_b, gl_t, rtol=1e-2, atol=1e-3)
    assert ok, msg


# ============================================================
#   RUN ALL
# ============================================================

if __name__ == "__main__":
    # print("=" * 60)
    # print("BOBR Gradient Correctness Tests")
    # print("=" * 60)
    
    # print("\n-- Forward sanity checks --")
    # run_test("forward_exp", test_forward_exp)
    # run_test("forward_log", test_forward_log)
    # run_test("forward_matmul", test_forward_matmul)
    # run_test("forward_sub", test_forward_sub)
    
    # print("\n-- Unary op gradients --")
    # run_test("neg_grad", test_neg_grad)
    # run_test("relu_grad", test_relu_grad)
    # run_test("exp_grad", test_exp_grad)
    # run_test("log_grad", test_log_grad)
    
    # print("\n-- Binary op gradients --")
    # run_test("add_grad", test_add_grad)
    # run_test("sub_grad", test_sub_grad)
    # run_test("mul_grad", test_mul_grad)
    # run_test("div_grad", test_div_grad)
    
    # print("\n-- Matmul gradients --")
    # run_test("matmul_grad", test_matmul_grad)
    # run_test("matmul_grad_nonsquare", test_matmul_grad_nonsquare)
    
    # print("\n-- Reduction gradients --")
    # run_test("sum_dim_grad", test_sum_dim_grad)
    
    # print("\n-- Chain gradients --")
    # run_test("chain_sub_then_mul", test_chain_sub_then_mul)
    # run_test("chain_matmul_relu_sum", test_chain_matmul_relu_sum)
    # run_test("chain_log_softmax", test_chain_log_softmax)
    # run_test("shared_input_accumulation", test_shared_input_grad_accumulation)
    
    # print("\n-- Numerical gradient checks --")
    # run_test("numerical_mul", test_numerical_grad_mul)
    # run_test("numerical_sub", test_numerical_grad_sub)
    # run_test("numerical_div", test_numerical_grad_div)
    # run_test("numerical_chain (relu(a@b+c))", test_numerical_grad_chain)
    
    print("\n-- Cross-entropy --")
    run_test("cross_entropy_forward", test_cross_entropy_forward)
    run_test("cross_entropy_backward", test_cross_entropy_backward)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print("\nFailed tests:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)