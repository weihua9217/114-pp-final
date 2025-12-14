import os
import torch
import torch.nn as nn

CUDA_AVAILABLE = False
softmax_cuda = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_SOURCE = os.path.join(CURRENT_DIR, "softmax_kernel.cu")

try:
    import softmax_cuda as _softmax_cuda
    softmax_cuda = _softmax_cuda
    CUDA_AVAILABLE = True
    print("[INFO] Custom CUDA softmax kernel loaded (pre-compiled)!")
except ImportError:
    pass

if not CUDA_AVAILABLE:
    print("[WARNING] Custom CUDA kernel not available, falling back to PyTorch softmax")


class CUDASoftmaxFunction(torch.autograd.Function):
    """
    Custom autograd function for CUDA softmax
    """
    @staticmethod
    def forward(ctx, input, dim):

        original_shape = input.shape
        original_dim = dim

        if dim < 0:
            dim = len(original_shape) + dim

        if dim != len(original_shape) - 1:
            input = input.transpose(dim, -1).contiguous()

        transposed_shape = input.shape
        last_dim_size = input.size(-1)
        input_2d = input.view(-1, last_dim_size)

        output_2d = softmax_cuda.forward(input_2d)

        output = output_2d.view(transposed_shape)

        if original_dim != len(original_shape) - 1:
            if original_dim < 0:
                original_dim = len(original_shape) + original_dim
            output = output.transpose(original_dim, -1).contiguous()

        ctx.save_for_backward(output)
        ctx.dim = original_dim if original_dim >= 0 else len(original_shape) + original_dim

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Softmax backward:
        d_input = softmax * (grad_output - sum(grad_output * softmax))
        """
        output, = ctx.saved_tensors
        dim = ctx.dim

        sum_term = (grad_output * output).sum(dim=dim, keepdim=True)
        grad_input = output * (grad_output - sum_term)

        return grad_input, None


def custom_softmax(input, dim=-1):
    if CUDA_AVAILABLE and input.is_cuda:
        return CUDASoftmaxFunction.apply(input, dim)
    else:
        return torch.softmax(input, dim=dim)


class CustomSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return custom_softmax(x, self.dim)


def patch_vit_attention(model):
    if not CUDA_AVAILABLE:
        print("[WARNING] CUDA kernel not available, model unchanged")
        return model

    patched_count = 0

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Attention':
            if hasattr(module, 'attend'):
                original_attend = module.attend

                class CustomAttend(nn.Module):
                    def __init__(self, scale):
                        super().__init__()
                        self.scale = scale

                    def forward(self, q, k):
                        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                        return custom_softmax(dots, dim=-1)

                scale = getattr(module, 'scale', 1.0)
                module.attend = CustomAttend(scale)
                patched_count += 1
                print(f"[INFO] Patched attention in: {name}")

    print(f"[INFO] Total patched attention modules: {patched_count}")
    return model



def test_custom_softmax():
    if not CUDA_AVAILABLE:
        print("CUDA kernel not available, skipping test")
        return False

    print("\n" + "=" * 50)
    print("Testing Custom CUDA Softmax")
    print("=" * 50)

    print("\n[Test 1] 2D tensor test...")
    x_2d = torch.randn(32, 128, device='cuda')
    out_custom = custom_softmax(x_2d, dim=-1)
    out_pytorch = torch.softmax(x_2d, dim=-1)
    diff_2d = (out_custom - out_pytorch).abs().max().item()
    print(f"  Max diff (2D): {diff_2d:.2e}")

    print("\n[Test 2] 4D tensor test (attention-like)...")
    x_4d = torch.randn(2, 12, 197, 197, device='cuda') 
    out_custom_4d = custom_softmax(x_4d, dim=-1)
    out_pytorch_4d = torch.softmax(x_4d, dim=-1)
    diff_4d = (out_custom_4d - out_pytorch_4d).abs().max().item()
    print(f"  Max diff (4D): {diff_4d:.2e}")

    print("\n[Test 3] Softmax sum check...")
    sum_check = out_custom_4d.sum(dim=-1)
    sum_error = (sum_check - 1.0).abs().max().item()
    print(f"  Max sum error: {sum_error:.2e}")

    print("\n[Test 4] Performance comparison...")
    x_perf = torch.randn(32, 12, 197, 197, device='cuda')

    for _ in range(10):
        _ = custom_softmax(x_perf, dim=-1)
        _ = torch.softmax(x_perf, dim=-1)
    torch.cuda.synchronize()

    import time

    # Custom CUDA
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = custom_softmax(x_perf, dim=-1)
    torch.cuda.synchronize()
    custom_time = time.time() - start

    # PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.softmax(x_perf, dim=-1)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start

    print(f"  Custom CUDA: {custom_time*1000:.2f} ms (100 iterations)")
    print(f"  PyTorch:     {pytorch_time*1000:.2f} ms (100 iterations)")
    print(f"  Speedup:     {pytorch_time/custom_time:.2f}x")

    print("\n" + "=" * 50)

    passed = diff_2d < 1e-5 and diff_4d < 1e-5 and sum_error < 1e-5
    print(f"Test {'PASSED' if passed else 'FAILED'}!")
    print("=" * 50)

    return passed


if __name__ == "__main__":
    test_custom_softmax()
