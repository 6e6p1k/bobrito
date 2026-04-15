"""
Comprehensive test suite for BOBR operations vs PyTorch.
Tests forward pass, backward pass, and gradients for all basic operations.
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import torch
import bobr


class OpTest:
    """Base class for operation testing."""
    
    def __init__(self, name, tolerances=None):
        self.name = name
        # Default tolerances for forward and backward passes
        self.rtol_fwd = 1e-4 if tolerances is None else tolerances.get('rtol_fwd', 1e-4)
        self.atol_fwd = 1e-5 if tolerances is None else tolerances.get('atol_fwd', 1e-5)
        self.rtol_bwd = 1e-3 if tolerances is None else tolerances.get('rtol_bwd', 1e-3)
        self.atol_bwd = 1e-4 if tolerances is None else tolerances.get('atol_bwd', 1e-4)
        self.passed = False
        self.error_msg = ""
    
    def allclose(self, bobr_out, torch_out, rtol, atol, name=""):
        """Check if outputs are close."""
        bobr_np = np.array(bobr_out) if not isinstance(bobr_out, np.ndarray) else bobr_out
        torch_np = torch_out.detach().cpu().numpy() if isinstance(torch_out, torch.Tensor) else np.array(torch_out)
        
        if bobr_np.shape != torch_np.shape:
            return False, f"Shape mismatch: {bobr_np.shape} vs {torch_np.shape}"
        
        if not np.allclose(bobr_np, torch_np, rtol=rtol, atol=atol):
            max_diff = np.abs(bobr_np - torch_np).max()
            max_rel = np.abs((bobr_np - torch_np) / (torch_np + 1e-8)).max()
            return False, f"Values differ (max_diff={max_diff:.2e}, max_rel={max_rel:.2e})"

        return True, "OK"
    
    def run(self):
        """Run the test."""
        try:
            self.forward_pass()
            self.backward_pass()
            self.passed = True
            return f"✓ {self.name}"
        except AssertionError as e:
            self.error_msg = str(e)
            return f"✗ {self.name}: {self.error_msg}"
    
    def forward_pass(self):
        """Override this method."""
        raise NotImplementedError
    
    def backward_pass(self):
        """Override this method."""
        raise NotImplementedError


class UnaryOpTest(OpTest):
    """Test unary operations."""
    
    def __init__(self, name, bobr_op, torch_op, shape=(2, 3, 4), tolerances=None):
        super().__init__(name, tolerances)
        self.bobr_op = bobr_op
        self.torch_op = torch_op
        self.shape = shape
    
    def forward_pass(self):
        # Create test data in positive range to avoid issues with log/sqrt
        data = np.random.uniform(0.1, 2.0, self.shape).astype(np.float32)
        
        # BOBR forward - need to create scalar loss and call build_backward for eval() to work
        x_bobr = bobr.full(list(self.shape), 0.0)
        x_bobr.steal_data(torch.from_numpy(data).cuda())
        y_bobr = self.bobr_op(x_bobr)
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        y_bobr_out = np.array(y_bobr.data)
        
        # PyTorch forward
        x_torch = torch.from_numpy(data).cuda()
        y_torch = self.torch_op(x_torch)
        
        # Compare
        assert self.allclose(y_bobr_out, y_torch, self.rtol_fwd, self.atol_fwd)[0], \
            f"Forward: {self.allclose(y_bobr_out, y_torch, self.rtol_fwd, self.atol_fwd)[1]}"
    
    def backward_pass(self):
        # Create test data
        data = np.random.uniform(0.1, 2.0, self.shape).astype(np.float32)
        
        # BOBR backward - eval() does both forward and backward
        x_bobr = bobr.full(list(self.shape), 0.0)
        x_bobr.steal_data(torch.from_numpy(data.copy()).cuda())
        y_bobr = self.bobr_op(x_bobr)
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        grad_bobr = np.array(x_bobr.grad) if x_bobr.requires_grad else None
        
        # PyTorch backward
        x_torch = torch.from_numpy(data).cuda()
        x_torch.requires_grad = True
        y_torch = self.torch_op(x_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()
        grad_torch = x_torch.grad
        
        # Compare gradients
        if grad_bobr is not None:
            assert self.allclose(grad_bobr, grad_torch, self.rtol_bwd, self.atol_bwd)[0], \
                f"Backward: {self.allclose(grad_bobr, grad_torch, self.rtol_bwd, self.atol_bwd)[1]}"


class BinaryOpTest(OpTest):
    """Test binary operations."""
    
    def __init__(self, name, bobr_op, torch_op, shape_a=(2, 3, 4), shape_b=(2, 3, 4), tolerances=None):
        super().__init__(name, tolerances)
        self.bobr_op = bobr_op
        self.torch_op = torch_op
        self.shape_a = shape_a
        self.shape_b = shape_b
    
    def forward_pass(self):
        # Create test data
        data_a = np.random.uniform(0.1, 2.0, self.shape_a).astype(np.float32)
        data_b = np.random.uniform(0.1, 2.0, self.shape_b).astype(np.float32)
        
        # BOBR forward - need scalar loss and build_backward for eval() to work
        a_bobr = bobr.full(list(self.shape_a), 0.0)
        b_bobr = bobr.full(list(self.shape_b), 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b).cuda())
        c_bobr = self.bobr_op(a_bobr, b_bobr)
        loss_bobr = bobr.sum(c_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        c_bobr_out = np.array(c_bobr.data)
        
        # PyTorch forward
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        c_torch = self.torch_op(a_torch, b_torch)
        
        # Compare
        assert self.allclose(c_bobr_out, c_torch, self.rtol_fwd, self.atol_fwd)[0], \
            f"Forward: {self.allclose(c_bobr_out, c_torch, self.rtol_fwd, self.atol_fwd)[1]}"
    
    def backward_pass(self):
        # Create test data
        data_a = np.random.uniform(0.1, 2.0, self.shape_a).astype(np.float32)
        data_b = np.random.uniform(0.1, 2.0, self.shape_b).astype(np.float32)
        
        # BOBR backward - eval() does both forward and backward
        a_bobr = bobr.full(list(self.shape_a), 0.0)
        b_bobr = bobr.full(list(self.shape_b), 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a.copy()).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b.copy()).cuda())
        c_bobr = self.bobr_op(a_bobr, b_bobr)
        loss_bobr = bobr.sum(c_bobr, 0, False)
        loss_bobr.eval()  # Does both forward and backward
        bobr.sync()
        grad_a_bobr = np.array(a_bobr.grad) if a_bobr.requires_grad else None
        grad_b_bobr = np.array(b_bobr.grad) if b_bobr.requires_grad else None
        
        # PyTorch backward
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        c_torch = self.torch_op(a_torch, b_torch)
        loss_torch = c_torch.sum()
        loss_torch.backward()
        grad_a_torch = a_torch.grad
        grad_b_torch = b_torch.grad
        
        # Compare gradients
        if grad_a_bobr is not None:
            assert self.allclose(grad_a_bobr, grad_a_torch, self.rtol_bwd, self.atol_bwd)[0], \
                f"Backward (a): {self.allclose(grad_a_bobr, grad_a_torch, self.rtol_bwd, self.atol_bwd)[1]}"
        if grad_b_bobr is not None:
            assert self.allclose(grad_b_bobr, grad_b_torch, self.rtol_bwd, self.atol_bwd)[0], \
                f"Backward (b): {self.allclose(grad_b_bobr, grad_b_torch, self.rtol_bwd, self.atol_bwd)[1]}"


class MatmulTest(OpTest):
    """Test matrix multiplication."""
    
    def __init__(self, name="matmul", m=4, k=5, n=3):
        super().__init__(name)
        self.m = m
        self.k = k
        self.n = n
    
    def forward_pass(self):
        data_a = np.random.randn(self.m, self.k).astype(np.float32)
        data_b = np.random.randn(self.k, self.n).astype(np.float32)
        
        # BOBR - need scalar loss and build_backward for eval() to work
        a_bobr = bobr.full([self.m, self.k], 0.0)
        b_bobr = bobr.full([self.k, self.n], 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b).cuda())
        c_bobr = bobr.matmul(a_bobr, b_bobr)
        loss_bobr = bobr.sum(c_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        c_bobr_out = np.array(c_bobr.data)
        
        # PyTorch
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        c_torch = a_torch @ b_torch
        
        assert self.allclose(c_bobr_out, c_torch, self.rtol_fwd, self.atol_fwd)[0], \
            f"Forward: {self.allclose(c_bobr_out, c_torch, self.rtol_fwd, self.atol_fwd)[1]}"
    
    def backward_pass(self):
        data_a = np.random.randn(self.m, self.k).astype(np.float32)
        data_b = np.random.randn(self.k, self.n).astype(np.float32)
        
        # BOBR - eval() does both forward and backward
        a_bobr = bobr.full([self.m, self.k], 0.0)
        b_bobr = bobr.full([self.k, self.n], 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a.copy()).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b.copy()).cuda())
        c_bobr = bobr.matmul(a_bobr, b_bobr)
        loss_bobr = bobr.sum(c_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        grad_a_bobr = np.array(a_bobr.grad)
        grad_b_bobr = np.array(b_bobr.grad)
        
        # PyTorch
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        c_torch = a_torch @ b_torch
        loss_torch = c_torch.sum()
        loss_torch.backward()
        grad_a_torch = a_torch.grad
        grad_b_torch = b_torch.grad
        
        assert self.allclose(grad_a_bobr, grad_a_torch, self.rtol_bwd, self.atol_bwd)[0], \
            f"Backward (a): {self.allclose(grad_a_bobr, grad_a_torch, self.rtol_bwd, self.atol_bwd)[1]}"
        assert self.allclose(grad_b_bobr, grad_b_torch, self.rtol_bwd, self.atol_bwd)[0], \
            f"Backward (b): {self.allclose(grad_b_bobr, grad_b_torch, self.rtol_bwd, self.atol_bwd)[1]}"


class ReductionOpTest(OpTest):
    """Test reduction operations."""
    
    def __init__(self, name, bobr_op, torch_op, shape=(2, 3, 4, 5), dim=2, keepdims=False, tolerances=None):
        super().__init__(name, tolerances)
        self.bobr_op = bobr_op
        self.torch_op = torch_op
        self.shape = shape
        self.dim = dim
        self.keepdims = keepdims
    
    def forward_pass(self):
        data = np.random.randn(*self.shape).astype(np.float32)
        
        # BOBR - need scalar loss and build_backward for eval() to work
        x_bobr = bobr.full(list(self.shape), 0.0)
        x_bobr.steal_data(torch.from_numpy(data).cuda())
        y_bobr = self.bobr_op(x_bobr, self.dim, self.keepdims)
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        y_bobr_out = np.array(y_bobr.data)
        
        # PyTorch
        x_torch = torch.from_numpy(data).cuda()
        y_torch = self.torch_op(x_torch, dim=self.dim, keepdim=self.keepdims)
        # min/max return (values, indices), extract values
        if isinstance(y_torch, tuple):
            y_torch = y_torch[0]
        
        assert self.allclose(y_bobr_out, y_torch, self.rtol_fwd, self.atol_fwd)[0], \
            f"Forward: {self.allclose(y_bobr_out, y_torch, self.rtol_fwd, self.atol_fwd)[1]}"
    
    def backward_pass(self):
        data = np.random.randn(*self.shape).astype(np.float32)
        
        # BOBR - eval() does both forward and backward
        x_bobr = bobr.full(list(self.shape), 0.0)
        x_bobr.steal_data(torch.from_numpy(data.copy()).cuda())
        y_bobr = self.bobr_op(x_bobr, self.dim, self.keepdims)
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.eval()
        bobr.sync()
        grad_bobr = np.array(x_bobr.grad)
        
        # PyTorch
        x_torch = torch.from_numpy(data).cuda()
        x_torch.requires_grad = True
        y_torch = self.torch_op(x_torch, dim=self.dim, keepdim=self.keepdims)
        # min/max return (values, indices), extract values
        if isinstance(y_torch, tuple):
            y_torch = y_torch[0]
        loss_torch = y_torch.sum()
        loss_torch.backward()
        grad_torch = x_torch.grad
        
        assert self.allclose(grad_bobr, grad_torch, self.rtol_bwd, self.atol_bwd)[0], \
            f"Backward: {self.allclose(grad_bobr, grad_torch, self.rtol_bwd, self.atol_bwd)[1]}"


def run_all_tests():
    """Run all operation tests."""
    
    print("=" * 70)
    print("BOBR Framework - Operation Tests vs PyTorch")
    print("=" * 70)
    
    tests = []
    
    # ==================== Unary Operations ====================
    print("\n--- Unary Operations ---")
    tests.append(UnaryOpTest("relu", bobr.relu, torch.relu))
    tests.append(UnaryOpTest("exp", bobr.exp, torch.exp))
    tests.append(UnaryOpTest("log", bobr.log, torch.log, (2, 3)))
    tests.append(UnaryOpTest("neg", bobr.neg, torch.neg, (2, 3, 4)))
    tests.append(UnaryOpTest("sqrt", bobr.sqrt, torch.sqrt, (2, 3)))
    
    # ==================== Binary Operations ====================
    print("\n--- Binary Operations ---")
    tests.append(BinaryOpTest("add", bobr.add, torch.add))
    tests.append(BinaryOpTest("sub", bobr.sub, torch.sub))
    tests.append(BinaryOpTest("mul", bobr.mul, torch.mul))
    tests.append(BinaryOpTest("div", bobr.div, torch.div))
    
    # ==================== Matrix Multiplication ====================
    print("\n--- Matrix Operations ---")
    tests.append(MatmulTest("matmul", m=4, k=5, n=3))
    tests.append(MatmulTest("matmul (square)", m=5, k=5, n=5))
    tests.append(MatmulTest("matmul (batch-like)", m=8, k=16, n=4))
    
    # ==================== Reduction Operations ====================
    print("\n--- Reduction Operations ---")
    tests.append(ReductionOpTest("sum (dim=0)", bobr.sum, torch.sum, shape=(3, 4, 5), dim=0))
    tests.append(ReductionOpTest("sum (dim=1)", bobr.sum, torch.sum, shape=(3, 4, 5), dim=1))
    tests.append(ReductionOpTest("sum (dim=2)", bobr.sum, torch.sum, shape=(3, 4, 5), dim=2))
    tests.append(ReductionOpTest("sum (keepdims)", bobr.sum, torch.sum, shape=(3, 4, 5), dim=1, keepdims=True))
    
    tests.append(ReductionOpTest("min (dim=1)", bobr.min, torch.min, shape=(3, 4, 5), dim=1))
    tests.append(ReductionOpTest("max (dim=1)", bobr.max, torch.max, shape=(3, 4, 5), dim=1))
    
    # ==================== Run Tests ====================
    passed = 0
    failed = 0
    
    for test in tests:
        result = test.run()
        print(result)
        if test.passed:
            passed += 1
        else:
            failed += 1
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
