#include "bindings.h"
#include "upscale.cuh"
#include "helpers.cuh"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

torch::Tensor gradient_aware_upscale_forward_tensor(
    const torch::Tensor &render,
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi
) {
    DEVICE_GUARD(render);
    CHECK_INPUT(render);
    CHECK_INPUT(dx);
    CHECK_INPUT(dy);
    CHECK_INPUT(dxy);

    const int src_h = render.size(0);
    const int src_w = render.size(1);
    const int channels = render.size(2);

    const float roi_x1 = std::get<0>(roi);
    const float roi_y1 = std::get<1>(roi);
    const float roi_x2 = std::get<2>(roi);
    const float roi_y2 = std::get<3>(roi);

    torch::Tensor output = torch::zeros(
        {dst_h, dst_w, channels},
        render.options().dtype(torch::kFloat32)
    );

    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

    gradient_aware_upscale_kernel<<<grid, block>>>(
            dst_h,
            dst_w,
            src_h,
            src_w,
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,
            (float3*)render.contiguous().data_ptr<float>(),
            (float3*)dx.contiguous().data_ptr<float>(),
            (float3*)dy.contiguous().data_ptr<float>(),
            (float3*)dxy.contiguous().data_ptr<float>(),
            (float3*)output.data_ptr<float>()
    );

    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gradient_aware_upscale_backward_tensor(
    const torch::Tensor &grad_output,
    const torch::Tensor &render,
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi
) {
    DEVICE_GUARD(grad_output);
    CHECK_INPUT(grad_output);

    const int src_h = render.size(0);
    const int src_w = render.size(1);
    const int channels = render.size(2);

    const float roi_x1 = std::get<0>(roi);
    const float roi_y1 = std::get<1>(roi);
    const float roi_x2 = std::get<2>(roi);
    const float roi_y2 = std::get<3>(roi);

    auto grad_render = torch::zeros_like(render);
    auto grad_dx = torch::zeros_like(dx);
    auto grad_dy = torch::zeros_like(dy);
    auto grad_dxy = torch::zeros_like(dxy);

    dim3 block(16, 16);
    dim3 grid_src(
        (src_w + block.x - 1) / block.x,
        (src_h + block.y - 1) / block.y
    );
    dim3 grid_dst(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

    gradient_aware_upscale_backward_kernel<<<grid, block>>>(
            dst_h, dst_w, src_h, src_w,
            roi_x1, roi_y1, roi_x2, roi_y2,
            (float3*)grad_output.contiguous().data_ptr<float>(),
            (float3*)grad_render.data_ptr<float>(),
            (float3*)grad_dx.data_ptr<float>(),
            (float3*)grad_dy.data_ptr<float>(),
            (float3*)grad_dxy.data_ptr<float>()
    );

    return std::make_tuple(grad_render, grad_dx, grad_dy, grad_dxy);
}