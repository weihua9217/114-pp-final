"""
ViT Inference with Custom CUDA Softmax Kernel
使用自定義 CUDA kernel 的 ViT 推論腳本
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from custom_softmax import custom_softmax, CustomSoftmax, CUDA_AVAILABLE

class CustomAttention(nn.Module):
    """使用自定義 CUDA softmax 的 Attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        with nvtx.range("attention_matmul_qk"):
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        with nvtx.range("custom_cuda_softmax"):
            attn = custom_softmax(dots, dim=-1)

        with nvtx.range("attention_matmul_v"):
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """MLP Block"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LayerNorm"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CustomAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CustomViT(nn.Module):
    """
    Vision Transformer with Custom CUDA Softmax
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        b = img.shape[0]

        # Patch embedding
        with nvtx.range("patch_embedding"):
            x = self.patch_embedding(img)  # (B, dim, H/P, W/P)
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        with nvtx.range("position_embedding"):
            x = x + self.pos_embedding
            x = self.dropout(x)

        # Transformer
        with nvtx.range("transformer"):
            x = self.transformer(x)

        # Classification
        with nvtx.range("classification"):
            x = self.norm(x[:, 0])
            x = self.head(x)

        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA kernel available: {CUDA_AVAILABLE}")

    print("\n" + "=" * 50)
    print("Creating Custom ViT with CUDA Softmax")
    print("=" * 50)

    model = CustomViT(
        image_size=224,
        patch_size=16,
        num_classes=10,       # CIFAR-10
        dim=768,              # embedding dimension
        depth=12,             # transformer blocks
        heads=12,             # attention heads
        mlp_dim=3072,         # feedforward hidden dim
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nLoading CIFAR-10 dataset...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========== Warmup ==========
    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # ========== Inference ==========
    print("Starting inference...")
    correct = 0
    total = 0
    total_time = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            torch.cuda.synchronize()
            start_time = time.time()

            with nvtx.range(f"batch_{batch_idx}"):
                outputs = model(images)

            torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Batch [{batch_idx + 1}/{len(test_loader)}] - "
                      f"Accuracy: {100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    avg_time_per_batch = total_time / len(test_loader) * 1000
    avg_time_per_image = total_time / total * 1000
    throughput = total / total_time

    print("\n" + "=" * 50)
    print("Custom ViT + CUDA Softmax Results")
    print("=" * 50)
    print(f"CUDA Kernel: {'Enabled' if CUDA_AVAILABLE else 'Disabled (fallback to PyTorch)'}")
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time per batch: {avg_time_per_batch:.2f} ms")
    print(f"Average time per image: {avg_time_per_image:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("=" * 50)


if __name__ == '__main__':
    main()
