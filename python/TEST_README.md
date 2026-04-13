# BOBR Framework - Test Suite

This directory contains comprehensive tests for the BOBR autograd framework, comparing operations and gradients to PyTorch.

## Test Files

### 1. `test_ops_vs_torch.py` - Basic Operations
Tests forward pass and backward pass (gradients) for all basic operations.

**Unary Operations:**
- `relu` - Rectified Linear Unit activation
- `exp` - Element-wise exponential
- `log` - Element-wise natural logarithm
- `neg` - Element-wise negation
- `sqrt` - Element-wise square root

**Binary Operations:**
- `add` - Element-wise addition
- `sub` - Element-wise subtraction
- `mul` - Element-wise multiplication
- `div` - Element-wise division

**Matrix Operations:**
- `matmul` - Matrix multiplication (also tests with different shapes)

**Reduction Operations:**
- `sum` - Sum along dimension (with/without keepdims)
- `min` - Minimum along dimension
- `max` - Maximum along dimension

**Running:**
```bash
python python/test_ops_vs_torch.py
```

### 2. `test_advanced_ops.py` - Advanced Tests
Tests numerical gradient checking and chained operations.

**Numerical Gradient Tests:**
- Validates framework gradients against numerical gradients computed via finite differences
- Tests: `relu`, `exp`, `matmul`
- Uses central differences: `grad[i] ≈ (f(x+ε) - f(x-ε)) / (2ε)`

**Chained Operation Tests:**
- `relu(a) + relu(b)` - Tests operator combinations
- `(A @ B) * C` - Matrix multiplication followed by element-wise multiply
- `(exp(a) + b) * c` - Mixed chained operations

These tests verify that the computational graph correctly chains gradients through multiple operations.

**Running:**
```bash
python python/test_advanced_ops.py
```

### 3. `test_shapes.py` - Shape & Reshape Operations
Tests reshape, contiguous, and reductions along different dimensions.

**Reshape Tests:**
- Various reshape operations: `(2,3,4) → (6,4)`, `(2,3,4) → (24,)`, etc.
- Tests both forward and backward passes

**Reduction Dimension Tests:**
- Tests sum/min/max along all possible dimensions
- Example: reduce along axis 0, 1, 2 for a 3D tensor

**Contiguous Tests:**
- Verifies contiguous memory layout operations

**Running:**
```bash
python python/test_shapes.py
```

## Test Structure

Each test follows this pattern:

1. **Forward Pass Validation:**
   - Create random input data
   - Compute output in BOBR
   - Compute output in PyTorch
   - Compare results (rtol=1e-4, atol=1e-5)

2. **Backward Pass Validation:**
   - Initialize gradients with `.build_backward()`
   - Call `.eval()` which handles both forward and backward
   - Extract gradients and compare to PyTorch (rtol=1e-3, atol=1e-4)

3. **Numerical Gradient Testing (advanced only):**
   - Compute numerical gradients using finite differences
   - Compare to analytical gradients
   - Verifies gradient correctness

## Tolerance Settings

### Forward Pass
- Relative tolerance: 1e-4
- Absolute tolerance: 1e-5

### Backward Pass
- Relative tolerance: 1e-3
- Absolute tolerance: 1e-4

These tolerances account for:
- CUDA floating-point arithmetic differences
- Reduction order differences
- Numerical precision limits

## Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- BOBR compiled with CUDA support

## Expected Output

Successful test run:
```
======================================================================
BOBR Framework - Operation Tests vs PyTorch
======================================================================

--- Unary Operations ---
✓ relu
✓ exp
✓ log
✓ neg
✓ sqrt

--- Binary Operations ---
✓ add
✓ sub
✓ mul
✓ div

--- Matrix Operations ---
✓ matmul
✓ matmul (square)
✓ matmul (batch-like)

--- Reduction Operations ---
✓ sum (dim=0)
✓ sum (dim=1)
✓ sum (dim=2)
✓ sum (keepdims)
✓ min (dim=1)
✓ max (dim=1)

======================================================================
Results: 20 passed, 0 failed out of 20 tests
======================================================================
```

## Debugging Failed Tests

If a test fails:

1. **Check tolerance values:** The error message shows `max_diff` and `max_rel` (relative error)
2. **Verify CUDA correctness:** Ensure GPU memory is not corrupted
3. **Check data initialization:** Some operations require positive inputs (log, sqrt)
4. **Inspect shapes:** Shape mismatches are clearly reported

Example error message:
```
✗ relu: Forward: Values differ (max_diff=1.23e-4, max_rel=2.34e-3)
```

This likely indicates a numerical precision issue rather than a correctness bug.

## Adding New Tests

To add a test for a new operation:

```python
class MyOpTest(OpTest):
    def __init__(self):
        super().__init__("my_op")
    
    def forward_pass(self):
        # Create test data
        # Run BOBR operation
        # Run PyTorch operation
        # Compare with self.allclose()
    
    def backward_pass(self):
        # Similar to forward, but also check gradients
```

Then add to test list:
```python
tests.append(MyOpTest())
```

## Quick Test Commands

Run all basic operation tests:
```bash
cd /home/hz/nothing/bobr-framework-cuda
python python/test_ops_vs_torch.py
```

Run advanced tests only:
```bash
python python/test_advanced_ops.py
```

Run shape tests only:
```bash
python python/test_shapes.py
```

Run all tests sequentially:
```bash
python python/test_ops_vs_torch.py && \
python python/test_advanced_ops.py && \
python python/test_shapes.py
```

## Test Coverage

| Category | Count | Status |
|----------|-------|--------|
| Unary Ops | 5 | ✓ |
| Binary Ops | 4 | ✓ |
| Matrix Ops | 3 | ✓ |
| Reduction Ops | 6 | ✓ |
| Numerical Grads | 3 | ✓ |
| Chained Ops | 3 | ✓ |
| Reshape | 5 | ✓ |
| Reduction Dims | 11 | ✓ |
| Contiguous | 1 | ✓ |
| **Total** | **41** | **✓** |

## Notes

- All tests use float32 precision (matching BOBR kernel specification)
- Tests use CUDA tensors for both frameworks (no CPU fallback)
- Gradient tests always create a sum loss before backward to ensure scalar loss
- `keepdims` parameter tested for reductions
- Both positive and negative values used (handling of negative inputs for log/sqrt)
- Random seed is not fixed (tests use different random data each run for robustness)

## Implementation Details

### steal_data Pattern
Tests use `.steal_data()` to zero-copy import PyTorch tensors to BOBR:
```python
x_bobr = bobr.full([2, 3], 0.0)  # Create empty BOBR tensor
x_bobr.steal_data(torch_tensor)   # Copy GPU data from torch to bobr
```

### Graph Building
The backward pass uses the graph API with eval():
```python
loss_node = bobr.sum(operation, 0, False)   # Scalar loss
loss_node.build_backward()                   # Build backward graph
loss_node.eval()                             # Compute forward AND backward
```

### Gradient Access
Gradients are retrieved as numpy arrays:
```python
grad = np.array(node.grad)  # Returns numpy copy of gradient
```
