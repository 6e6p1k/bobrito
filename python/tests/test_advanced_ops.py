"""
Advanced tests: chained operations and edge cases.
Tests using eval() which handles both forward and backward passes.
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import torch
import bobr


class NumericalGradientTest:
    """Test gradients against numerical gradients (finite differences)."""
    
    def __init__(self, name, shape=(2, 3), eps=1e-5):
        self.name = name
        self.shape = shape
        self.eps = eps
        self.passed = False
        self.error_msg = ""
    
    def compute_numerical_grad(self, func, x_data, eps=None):
        """Compute numerical gradient using central differences."""
        if eps is None:
            eps = self.eps
        
        grad = np.zeros_like(x_data, dtype=np.float32)
        
        for i in range(x_data.size):
            # f(x + eps)
            x_plus = x_data.copy()
            x_plus.flat[i] += eps
            f_plus = func(x_plus)
            
            # f(x - eps)
            x_minus = x_data.copy()
            x_minus.flat[i] -= eps
            f_minus = func(x_minus)
            
            grad.flat[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def run_relu(self):
        """Test ReLU gradient using numerical differentiation."""
        data = np.random.randn(*self.shape).astype(np.float32)
        
        # PyTorch analytical gradient
        x_torch = torch.from_numpy(data).cuda()
        x_torch.requires_grad = True
        y_torch = torch.relu(x_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()
        analytical_grad = x_torch.grad.cpu().numpy()
        
        # Numerical gradient
        def relu_func(x):
            x_bobr = bobr.full(list(self.shape), 0.0)
            x_bobr.steal_data(torch.from_numpy(x).cuda())
            y_bobr = bobr.relu(x_bobr)
            y_bobr.eval()
            return np.array(y_bobr.data).sum()
        
        numerical_grad = self.compute_numerical_grad(relu_func, data)
        
        # Compare
        if not np.allclose(analytical_grad, numerical_grad, rtol=1e-2, atol=1e-3):
            self.error_msg = f"Gradient mismatch: max_diff={np.abs(analytical_grad - numerical_grad).max():.2e}"
            return False
        
        self.passed = True
        return True
    
    def run_exp(self):
        """Test exp gradient."""
        data = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        
        x_torch = torch.from_numpy(data).cuda()
        x_torch.requires_grad = True
        y_torch = torch.exp(x_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()
        analytical_grad = x_torch.grad.cpu().numpy()
        
        def exp_func(x):
            x_bobr = bobr.full(list(self.shape), 0.0)
            x_bobr.steal_data(torch.from_numpy(x).cuda())
            y_bobr = bobr.exp(x_bobr)
            y_bobr.eval()
            return np.array(y_bobr.data).sum()
        
        numerical_grad = self.compute_numerical_grad(exp_func, data)
        
        if not np.allclose(analytical_grad, numerical_grad, rtol=1e-2, atol=1e-3):
            self.error_msg = f"Gradient mismatch: max_diff={np.abs(analytical_grad - numerical_grad).max():.2e}"
            return False
        
        self.passed = True
        return True
    
    def run_matmul(self):
        """Test matmul gradient."""
        m, k, n = 3, 4, 2
        data_a = np.random.randn(m, k).astype(np.float32)
        data_b = np.random.randn(k, n).astype(np.float32)
        
        # PyTorch
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        c_torch = a_torch @ b_torch
        loss_torch = c_torch.sum()
        loss_torch.backward()
        grad_a_torch = a_torch.grad.cpu().numpy()
        grad_b_torch = b_torch.grad.cpu().numpy()
        
        # Numerical for A
        def matmul_func_a(a):
            a_bobr = bobr.full([m, k], 0.0)
            b_bobr = bobr.full([k, n], 0.0)
            a_bobr.steal_data(torch.from_numpy(a).cuda())
            b_bobr.steal_data(torch.from_numpy(data_b).cuda())
            c_bobr = bobr.matmul(a_bobr, b_bobr)
            c_bobr.eval()
            return np.array(c_bobr.data).sum()
        
        numerical_grad_a = self.compute_numerical_grad(matmul_func_a, data_a)
        
        if not np.allclose(grad_a_torch, numerical_grad_a, rtol=1e-2, atol=1e-3):
            self.error_msg = f"Gradient A mismatch: max_diff={np.abs(grad_a_torch - numerical_grad_a).max():.2e}"
            return False
        
        # Numerical for B
        def matmul_func_b(b):
            a_bobr = bobr.full([m, k], 0.0)
            b_bobr = bobr.full([k, n], 0.0)
            a_bobr.steal_data(torch.from_numpy(data_a).cuda())
            b_bobr.steal_data(torch.from_numpy(b).cuda())
            c_bobr = bobr.matmul(a_bobr, b_bobr)
            c_bobr.eval()
            return np.array(c_bobr.data).sum()
        
        numerical_grad_b = self.compute_numerical_grad(matmul_func_b, data_b)
        
        if not np.allclose(grad_b_torch, numerical_grad_b, rtol=1e-2, atol=1e-3):
            self.error_msg = f"Gradient B mismatch: max_diff={np.abs(grad_b_torch - numerical_grad_b).max():.2e}"
            return False
        
        self.passed = True
        return True


class ChainedOpTest:
    """Test chained operations."""
    
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.error_msg = ""
    
    def allclose(self, a, b, rtol=1e-4, atol=1e-5):
        a_np = np.array(a) if not isinstance(a, np.ndarray) else a
        b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.array(b)
        return np.allclose(a_np, b_np, rtol=rtol, atol=atol)
    
    def run_relu_add(self):
        """Test relu(a) + relu(b)."""
        data_a = np.random.randn(2, 3).astype(np.float32)
        data_b = np.random.randn(2, 3).astype(np.float32)
        
        # BOBR - eval() does both forward and backward
        a_bobr = bobr.full([2, 3], 0.0)
        b_bobr = bobr.full([2, 3], 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a.copy()).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b.copy()).cuda())
        y_bobr = bobr.add(bobr.relu(a_bobr), bobr.relu(b_bobr))
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.build_backward()
        loss_bobr.eval()
        y_bobr_np = np.array(y_bobr.data)
        grad_a_bobr = np.array(a_bobr.grad)
        grad_b_bobr = np.array(b_bobr.grad)
        
        # PyTorch
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        y_torch = torch.relu(a_torch) + torch.relu(b_torch)
        loss_torch = y_torch.sum()
        loss_torch.backward()
        y_torch_np = y_torch.detach().cpu().numpy()
        grad_a_torch = a_torch.grad
        grad_b_torch = b_torch.grad
        
        # Compare
        if not (self.allclose(y_bobr_np, y_torch_np) and 
                self.allclose(grad_a_bobr, grad_a_torch) and
                self.allclose(grad_b_bobr, grad_b_torch)):
            self.error_msg = "Forward or backward mismatch"
            return False
        
        self.passed = True
        return True
    
    def run_matmul_mul_sum(self):
        """Test (A @ B) * C."""
        data_a = np.random.randn(3, 4).astype(np.float32)
        data_b = np.random.randn(4, 2).astype(np.float32)
        data_c = np.random.randn(3, 2).astype(np.float32)
        
        # BOBR - eval() does both forward and backward
        a_bobr = bobr.full([3, 4], 0.0)
        b_bobr = bobr.full([4, 2], 0.0)
        c_bobr = bobr.full([3, 2], 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a.copy()).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b.copy()).cuda())
        c_bobr.steal_data(torch.from_numpy(data_c.copy()).cuda())
        y_bobr = bobr.mul(bobr.matmul(a_bobr, b_bobr), c_bobr)
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.build_backward()
        loss_bobr.eval()
        y_bobr_np = np.array(y_bobr.data)
        grad_a_bobr = np.array(a_bobr.grad)
        grad_b_bobr = np.array(b_bobr.grad)
        grad_c_bobr = np.array(c_bobr.grad)
        
        # PyTorch
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        c_torch = torch.from_numpy(data_c).cuda()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        c_torch.requires_grad = True
        y_torch = (a_torch @ b_torch) * c_torch
        loss_torch = y_torch.sum()
        loss_torch.backward()
        y_torch_np = y_torch.detach().cpu().numpy()
        grad_a_torch = a_torch.grad
        grad_b_torch = b_torch.grad
        grad_c_torch = c_torch.grad
        
        # Compare
        if not (self.allclose(y_bobr_np, y_torch_np, rtol=1e-3) and 
                self.allclose(grad_a_bobr, grad_a_torch, rtol=1e-3) and
                self.allclose(grad_b_bobr, grad_b_torch, rtol=1e-3) and
                self.allclose(grad_c_bobr, grad_c_torch, rtol=1e-3)):
            self.error_msg = "Forward or backward mismatch"
            return False
        
        self.passed = True
        return True
    
    def run_exp_sum_mul(self):
        """Test (exp(a) + b) * c."""
        data_a = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        data_b = np.random.randn(2, 3).astype(np.float32)
        data_c = np.random.uniform(0.1, 2, (2, 3)).astype(np.float32)
        
        # BOBR - eval() does both forward and backward
        a_bobr = bobr.full([2, 3], 0.0)
        b_bobr = bobr.full([2, 3], 0.0)
        c_bobr = bobr.full([2, 3], 0.0)
        a_bobr.steal_data(torch.from_numpy(data_a.copy()).cuda())
        b_bobr.steal_data(torch.from_numpy(data_b.copy()).cuda())
        c_bobr.steal_data(torch.from_numpy(data_c.copy()).cuda())
        y_bobr = bobr.mul(bobr.add(bobr.exp(a_bobr), b_bobr), c_bobr)
        loss_bobr = bobr.sum(y_bobr, 0, False)
        loss_bobr.build_backward()
        loss_bobr.eval()
        grad_a_bobr = np.array(a_bobr.grad)
        grad_b_bobr = np.array(b_bobr.grad)
        grad_c_bobr = np.array(c_bobr.grad)
        
        # PyTorch
        a_torch = torch.from_numpy(data_a).cuda()
        b_torch = torch.from_numpy(data_b).cuda()
        c_torch = torch.from_numpy(data_c).cuda()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        c_torch.requires_grad = True
        y_torch = (torch.exp(a_torch) + b_torch) * c_torch
        loss_torch = y_torch.sum()
        loss_torch.backward()
        grad_a_torch = a_torch.grad
        grad_b_torch = b_torch.grad
        grad_c_torch = c_torch.grad
        
        # Compare
        if not (self.allclose(grad_a_bobr, grad_a_torch) and
                self.allclose(grad_b_bobr, grad_b_torch) and
                self.allclose(grad_c_bobr, grad_c_torch)):
            self.error_msg = "Backward mismatch"
            return False
        
        self.passed = True
        return True


def run_all_advanced_tests():
    """Run all advanced tests."""
    
    print("=" * 70)
    print("BOBR Framework - Advanced Tests (Numerical Grads & Chained Ops)")
    print("=" * 70)
    
    tests = []
    
    # ==================== Numerical Gradient Tests ====================
    print("\n--- Numerical Gradient Checking ---")
    
    grad_test_relu = NumericalGradientTest("relu (numerical grad)")
    grad_test_relu.run_relu()
    tests.append(grad_test_relu)
    
    grad_test_exp = NumericalGradientTest("exp (numerical grad)")
    grad_test_exp.run_exp()
    tests.append(grad_test_exp)
    
    grad_test_matmul = NumericalGradientTest("matmul (numerical grad)")
    grad_test_matmul.run_matmul()
    tests.append(grad_test_matmul)
    
    # ==================== Chained Operation Tests ====================
    print("\n--- Chained Operations ---")
    
    chain_test_1 = ChainedOpTest("relu(a) + relu(b)")
    chain_test_1.run_relu_add()
    tests.append(chain_test_1)
    
    chain_test_2 = ChainedOpTest("(A @ B) * C")
    chain_test_2.run_matmul_mul_sum()
    tests.append(chain_test_2)
    
    chain_test_3 = ChainedOpTest("(exp(a) + b) * c")
    chain_test_3.run_exp_sum_mul()
    tests.append(chain_test_3)
    
    # ==================== Print Results ====================
    passed = 0
    failed = 0
    
    for test in tests:
        if test.passed:
            print(f"✓ {test.name}")
            passed += 1
        else:
            print(f"✗ {test.name}: {test.error_msg}")
            failed += 1
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_advanced_tests()
    sys.exit(0 if success else 1)
