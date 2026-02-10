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


torch::Tensor gradient_aware_upscale_forward_tensor(
    const torch::Tensor &render,    // [H, W, 3]
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi  // x1, y1, x2, y2
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gradient_aware_upscale_backward_tensor(
    const torch::Tensor &grad_output,  // [dst_h, dst_w, 3]
    const torch::Tensor &render,       // [H, W, 3]
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi
);