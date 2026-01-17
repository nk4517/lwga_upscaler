# Lightweight Gradient-Aware Upscaler for 3DGS

Unofficial CUDA implementation of the gradient-aware bicubic spline upscaling kernel from:

> Niedermayr, S., Neuhauser, C., & Westermann, R. (2025).  
> *Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images*  
> arXiv:2503.14171v2 [cs.CV]  
> https://arxiv.org/abs/2503.14171

This repository implements **only the upscaling component** — the bicubic spline interpolation that uses analytical image gradients (∂I/∂x, ∂I/∂y, ∂²I/∂x∂y) from 3DGS rendering instead of finite-difference approximations.

## What's Implemented

- Forward pass: bicubic spline interpolation using analytical gradients
- Backward pass: gradient backpropagation for differentiable training
- Two backward kernel variants:
  - Destination-centric (atomicAdd-based)
  - Source-centric (no atomics, faster)
- Support for float, float2, float3, float4, floatN data types
- ROI (region of interest) support

## Requirements

- CUDA Toolkit
- PyTorch (for Python bindings)

## Installation

```bash
pip install -e .
```

## Usage

The upscaler expects four input tensors from 3DGS rendering:
- `render`: pixel colors [B, H, W, C] or [H, W, C]
- `dx`: ∂I/∂x analytical gradients [B, H, W, C] or [H, W, C]
- `dy`: ∂I/∂y analytical gradients [B, H, W, C] or [H, W, C]
- `dxy`: ∂²I/∂x∂y mixed partials [B, H, W, C] or [H, W, C]

```python
from lwga_upscaler import gradient_aware_upscale

output = gradient_aware_upscale(
    render, dx, dy, dxy,
    dst_h, dst_w,
    src_roi=roi_tensor  # optional, [B, 4] or [4] with (x1, y1, x2, y2)
)
```

Batched inputs return [B, dst_h, dst_w, C], unbatched return [dst_h, dst_w, C].
When `src_roi` is [4], the same ROI applies to all batch elements.

The operation is fully differentiable — gradients flow back through all four input tensors.

## How It Works

Standard bicubic interpolation estimates gradients via finite differences. This implementation instead uses the exact analytical gradients that 3DGS computes during rendering (Equations 8-10 in the paper), enabling more accurate spline fitting with minimal overhead.

The spline coefficients are computed as A = C⁻¹ · F · (C⁻¹)ᵀ where F contains function values and derivatives at the four corners of each interpolation cell.

## References

- [Original paper](https://arxiv.org/abs/2503.14171)

## License

This is an unofficial implementation. See the original paper for the method's licensing terms.
