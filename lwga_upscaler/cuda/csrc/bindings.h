#include "cuda_runtime.h"
#include "upscale.cuh"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));


torch::Tensor gradient_aware_upscale_forward(
    const torch::Tensor &render,    // [B, H, W, C]
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const torch::Tensor &src_roi    // [B, 4]: x1, y1, x2, y2 per batch
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gradient_aware_upscale_backward(
    const torch::Tensor &grad_output,  // [B, dst_h, dst_w, C]
    int src_h,
    int src_w,
    const torch::Tensor &src_roi       // [B, 4]
);
