"""Bicubic spline upscaling using analytical gradients from rasterization"""

from typing import Optional
import torch
from torch import Tensor
from torch.autograd import Function

import lwga_upscaler.cuda as _C


class _GradientAwareSplineUpscale(Function):
    @staticmethod
    def forward(
        ctx,
        render: Tensor,  # [B, H, W, C]
        dx: Tensor,
        dy: Tensor,
        dxy: Tensor,
        dst_h: int,
        dst_w: int,
        src_roi: Tensor,  # [B, 4]
    ) -> Tensor:
        ctx.src_h = render.shape[1]
        ctx.src_w = render.shape[2]
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
    render: Tensor,  # [B, H, W, C] or [H, W, C]
    dx: Tensor,      # [B, H, W, C] or [H, W, C]
    dy: Tensor,      # [B, H, W, C] or [H, W, C]
    dxy: Tensor,     # [B, H, W, C] or [H, W, C]
    dst_h: int,
    dst_w: int,
    src_roi: Optional[Tensor] = None,  # [B, 4] or None
) -> Tensor:
    """
    Bicubic spline interpolation using analytical gradients.
    
    Args:
        render: Rendered image [B, H, W, C] or [H, W, C]
        dx: Gradient w.r.t. x [B, H, W, C] or [H, W, C]
        dy: Gradient w.r.t. y [B, H, W, C] or [H, W, C]
        dxy: Mixed partial derivative [B, H, W, C] or [H, W, C]
        dst_h: Output height
        dst_w: Output width
        src_roi: Source ROI [B, 4] with (x1, y1, x2, y2) per batch, defaults to full image
    
    Returns:
        Upscaled image [B, dst_h, dst_w, C] or [dst_h, dst_w, C]
    """
    unbatched = render.ndim == 3
    if unbatched:
        render = render.unsqueeze(0)
        dx = dx.unsqueeze(0)
        dy = dy.unsqueeze(0)
        dxy = dxy.unsqueeze(0)
    
    b, h, w, c = render.shape

    if src_roi is None:
        src_roi = torch.tensor([[0.0, 0.0, float(w), float(h)]], device=render.device).expand(b, 4).contiguous()
    elif src_roi.ndim == 1:
        src_roi = src_roi.unsqueeze(0).expand(b, 4).contiguous()
    
    result = _GradientAwareSplineUpscale.apply(
        render.contiguous(), dx.contiguous(), dy.contiguous(), dxy.contiguous(),
        dst_h, dst_w, src_roi
    )
    
    if unbatched:
        result = result.squeeze(0)
    
    return result
