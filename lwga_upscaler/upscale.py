"""Bicubic spline upscaling using analytical gradients from rasterization"""

from typing import Optional, Tuple
from torch import Tensor
from torch.autograd import Function

import lwga_upscaler.cuda as _C


class _GradientAwareSplineUpscale(Function):
    @staticmethod
    def forward(
        ctx,
        render: Tensor,  # [H, W, 3]
        dx: Tensor,
        dy: Tensor,
        dxy: Tensor,
        dst_h: int,
        dst_w: int,
        src_roi: tuple[float, float, float, float],
    ) -> Tensor:
        ctx.src_h = render.shape[0]
        ctx.src_w = render.shape[1]
        ctx.src_roi = src_roi
        
        upscaled = _C.gradient_aware_upscale_forward(
            render, dx, dy, dxy, dst_h, dst_w, src_roi
        )
        return upscaled
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        v_render, v_dx, v_dy, v_dxy = _C.gradient_aware_upscale_backward(
            grad_output.contiguous(),
            ctx.src_h,
            ctx.src_w,
            ctx.src_roi,
        )
        
        return v_render, v_dx, v_dy, v_dxy, None, None, None


def gradient_aware_upscale(
    render: Tensor,  # [H, W, C]
    dx: Tensor,      # [H, W, C]
    dy: Tensor,      # [H, W, C]
    dxy: Tensor,     # [H, W, C]
    dst_h: int,
    dst_w: int,
    src_roi: Optional[Tuple[float, float, float, float]] = None,  # (x1, y1, x2, y2)
) -> Tensor:
    """
    Bicubic spline interpolation using analytical gradients.
    
    Args:
        render: Rendered image [H, W, C]
        dx: Gradient w.r.t. x [H, W, C]
        dy: Gradient w.r.t. y [H, W, C]
        dxy: Mixed partial derivative [H, W, C]
        dst_h: Output height
        dst_w: Output width
        src_roi: Source region of interest (x1, y1, x2, y2) in src pixel coordinates, defaults to full image
    
    Returns:
        Upscaled image [dst_h, dst_w, C]
    """
    h, w, c = render.shape
    
    if src_roi is None:
        src_roi = (0.0, 0.0, float(w), float(h))
    
    return _GradientAwareSplineUpscale.apply(
        render.contiguous(), dx.contiguous(), dy.contiguous(), dxy.contiguous(),
        dst_h, dst_w, src_roi
    )
