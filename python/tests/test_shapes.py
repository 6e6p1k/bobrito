"""
Shape and reshape operation tests.
"""

import sys
sys.path.insert(0, 'python')

import numpy as np
import torch
import bobrito as bobr
class ReshapeTest:
    """Test reshape operations."""
    
    def __init__(self, name, orig_shape, new_shape):
        self.name = name
        self.orig_shape = orig_shape
        self.new_shape = new_shape
        self.passed = False
        self.error_msg = ""
    
    def run(self):
        """Run the test."""
        try:
            # Create test data
            data = np.random.randn(*self.orig_shape).astype(np.float32)
            
            # BOBR forward
            x_bobr = bobr.from_numpy(data)
            y_bobr = x_bobr.reshape(self.new_shape)
            y_bobr.eval()
            y_bobr_out = np.array(y_bobr.data)
            
            # PyTorch forward
            x_torch = torch.from_numpy(data).cuda()
            y_torch = x_torch.reshape(self.new_shape)
            
            # Compare forward
            if y_bobr_out.shape != y_torch.shape:
                self.error_msg = f"Shape mismatch: {y_bobr_out.shape} vs {y_torch.shape}"
                return False
            
            if not np.allclose(y_bobr_out, y_torch.cpu().numpy(), rtol=1e-5):
                self.error_msg = "Forward values don't match"
                return False
            
            # BOBR backward - eval() does both forward and backward
            x_bobr_req = bobr.from_numpy(data, requires_grad=True)
            y_bobr_req = x_bobr_req.reshape(self.new_shape)
            loss_bobr = bobr.sum(y_bobr_req, 0, False)
            loss_bobr.eval()
            grad_bobr = np.array(x_bobr_req.grad)
            
            # PyTorch backward
            x_torch_req = torch.from_numpy(data).cuda()
            x_torch_req.requires_grad = True
            y_torch_req = x_torch_req.reshape(self.new_shape)
            loss_torch = y_torch_req.sum()
            loss_torch.backward()
            grad_torch = x_torch_req.grad
            
            # Compare backward
            if not np.allclose(grad_bobr, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4):
                self.error_msg = f"Backward mismatch: max_diff={np.abs(grad_bobr - grad_torch.cpu().numpy()).max():.2e}"
                return False
            
            self.passed = True
            return True
        
        except Exception as e:
            self.error_msg = str(e)
            return False


class ReductionDimTest:
    """Test reductions along different dimensions."""
    
    def __init__(self, name, shape, dim, op_name="sum"):
        self.name = name
        self.shape = shape
        self.dim = dim
        self.op_name = op_name
        self.passed = False
        self.error_msg = ""
    
    def run(self):
        """Run the test."""
        try:
            # Create test data
            data = np.random.randn(*self.shape).astype(np.float32)
            
            # Get operation
            if self.op_name == "sum":
                bobr_op = bobr.sum
                torch_op = torch.sum
            elif self.op_name == "min":
                bobr_op = bobr.min
                torch_op = torch.min
            elif self.op_name == "max":
                bobr_op = bobr.max
                torch_op = torch.max
            else:
                self.error_msg = f"Unknown op: {self.op_name}"
                return False
            
            # BOBR forward
            x_bobr = bobr.from_numpy(data)
            y_bobr = bobr_op(x_bobr, self.dim, False)
            y_bobr.eval()
            y_bobr_out = np.array(y_bobr.data)
            
            # PyTorch forward
            x_torch = torch.from_numpy(data).cuda()
            y_torch = torch_op(x_torch, dim=self.dim, keepdim=False)
            # Handle min/max which return (values, indices)
            if isinstance(y_torch, tuple):
                y_torch = y_torch[0]
            
            # Compare forward
            if y_bobr_out.shape != y_torch.shape:
                self.error_msg = f"Shape mismatch: {y_bobr_out.shape} vs {y_torch.shape}"
                return False
            
            if not np.allclose(y_bobr_out, y_torch.cpu().numpy(), rtol=1e-4):
                self.error_msg = f"Forward mismatch: max_diff={np.abs(y_bobr_out - y_torch.cpu().numpy()).max():.2e}"
                return False
            
            # BOBR backward - eval() does both forward and backward
            x_bobr_req = bobr.from_numpy(data, requires_grad=True)
            y_bobr_req = bobr_op(x_bobr_req, self.dim, False)
            # a 1-D input reduces to a scalar already; bobr.sum on a 0-d node crashes (SIGFPE)
            loss_bobr = bobr.sum(y_bobr_req, 0, False) if y_bobr_req.ndim > 0 else y_bobr_req
            loss_bobr.eval()
            grad_bobr = np.array(x_bobr_req.grad)
            
            # PyTorch backward
            x_torch_req = torch.from_numpy(data).cuda()
            x_torch_req.requires_grad = True
            y_torch_req = torch_op(x_torch_req, dim=self.dim, keepdim=False)
            if isinstance(y_torch_req, tuple):
                y_torch_req = y_torch_req[0]
            loss_torch = y_torch_req.sum()
            loss_torch.backward()
            grad_torch = x_torch_req.grad
            
            # Compare backward
            if not np.allclose(grad_bobr, grad_torch.cpu().numpy(), rtol=1e-3, atol=1e-4):
                self.error_msg = f"Backward mismatch: max_diff={np.abs(grad_bobr - grad_torch.cpu().numpy()).max():.2e}"
                return False
            
            self.passed = True
            return True
        
        except Exception as e:
            self.error_msg = str(e)
            return False


class ContiguousTest:
    """Test contiguous operation."""
    
    def __init__(self):
        self.name = "contiguous"
        self.passed = False
        self.error_msg = ""
    
    def run(self):
        """Run the test."""
        try:
            # Create test data
            shape = (3, 4)
            data = np.random.randn(*shape).astype(np.float32)
            
            # BOBR: reshape and contiguous
            x_bobr = bobr.from_numpy(data)
            y_bobr = bobr.contiguous(x_bobr)  # Should not change anything for normal arrays
            y_bobr.eval()
            y_bobr_out = np.array(y_bobr.data)
            
            # Compare
            if not np.allclose(y_bobr_out, data, rtol=1e-5):
                self.error_msg = "Contiguous changed data"
                return False
            
            # Test backward - eval() does both forward and backward
            x_bobr_req = bobr.from_numpy(data, requires_grad=True)
            y_bobr_req = bobr.contiguous(x_bobr_req)
            loss_bobr = bobr.sum(y_bobr_req, 0, False)
            loss_bobr.eval()
            grad_bobr = np.array(x_bobr_req.grad)
            
            # Gradient should be all ones
            if not np.allclose(grad_bobr, np.ones_like(grad_bobr), rtol=1e-5):
                self.error_msg = "Gradient incorrect for contiguous"
                return False
            
            self.passed = True
            return True
        
        except Exception as e:
            self.error_msg = str(e)
            return False


def run_all_shape_tests():
    """Run all shape tests."""
    
    print("=" * 70)
    print("BOBR Framework - Shape & Reshape Tests")
    print("=" * 70)
    
    tests = []
    
    # ==================== Reshape Tests ====================
    print("\n--- Reshape Operations ---")
    tests.append(ReshapeTest("reshape (2,3,4) -> (6,4)", (2, 3, 4), (6, 4)))
    tests.append(ReshapeTest("reshape (2,3,4) -> (24,)", (2, 3, 4), (24,)))
    tests.append(ReshapeTest("reshape (2,3,4) -> (2,12)", (2, 3, 4), (2, 12)))
    tests.append(ReshapeTest("reshape (12,) -> (3,4)", (12,), (3, 4)))
    tests.append(ReshapeTest("reshape (2,3,4) -> (4,3,2)", (2, 3, 4), (4, 3, 2)))
    
    # ==================== Reduction Dimension Tests ====================
    print("\n--- Reduction Along Different Dimensions ---")
    
    # Sum tests
    tests.append(ReductionDimTest("sum (4,5,6) dim=0", (4, 5, 6), 0, "sum"))
    tests.append(ReductionDimTest("sum (4,5,6) dim=1", (4, 5, 6), 1, "sum"))
    tests.append(ReductionDimTest("sum (4,5,6) dim=2", (4, 5, 6), 2, "sum"))
    tests.append(ReductionDimTest("sum (10,) dim=0", (10,), 0, "sum"))
    
    # Min tests
    tests.append(ReductionDimTest("min (4,5,6) dim=0", (4, 5, 6), 0, "min"))
    tests.append(ReductionDimTest("min (4,5,6) dim=1", (4, 5, 6), 1, "min"))
    tests.append(ReductionDimTest("min (4,5,6) dim=2", (4, 5, 6), 2, "min"))
    
    # Max tests
    tests.append(ReductionDimTest("max (4,5,6) dim=0", (4, 5, 6), 0, "max"))
    tests.append(ReductionDimTest("max (4,5,6) dim=1", (4, 5, 6), 1, "max"))
    tests.append(ReductionDimTest("max (4,5,6) dim=2", (4, 5, 6), 2, "max"))
    
    # ==================== Contiguous Test ====================
    print("\n--- Contiguous Operations ---")
    tests.append(ContiguousTest())
    
    # ==================== Print Results ====================
    passed = 0
    failed = 0
    
    for test in tests:
        if test.run():
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
    success = run_all_shape_tests()
    sys.exit(0 if success else 1)
