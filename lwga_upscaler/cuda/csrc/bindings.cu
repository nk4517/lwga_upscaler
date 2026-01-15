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
    const std::tuple<float, float, float, float> &src_roi
) {
    DEVICE_GUARD(render);
    CHECK_INPUT(render);
    CHECK_INPUT(dx);
    CHECK_INPUT(dy);
    CHECK_INPUT(dxy);

    const int src_h = render.size(0);
    const int src_w = render.size(1);
    const int channels = render.size(2);

    const float roi_x1 = std::get<0>(src_roi);
    const float roi_y1 = std::get<1>(src_roi);
    const float roi_x2 = std::get<2>(src_roi);
    const float roi_y2 = std::get<3>(src_roi);

    torch::Tensor output = torch::zeros(
        {dst_h, dst_w, channels},
        render.options().dtype(torch::kFloat32)
    );

    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

#define UPSCALE_FORWARD_DISPATCH(T) \
    gradient_aware_upscale_kernel<T><<<grid, block>>>( \
        dst_h, dst_w, src_h, src_w, \
        roi_x1, roi_y1, roi_x2, roi_y2, \
        (T*)render.contiguous().data_ptr<float>(), \
        (T*)dx.contiguous().data_ptr<float>(), \
        (T*)dy.contiguous().data_ptr<float>(), \
        (T*)dxy.contiguous().data_ptr<float>(), \
        (T*)output.data_ptr<float>());

    switch (channels) {
    case 1: UPSCALE_FORWARD_DISPATCH(float); break;
    case 2: UPSCALE_FORWARD_DISPATCH(float2); break;
    case 3: UPSCALE_FORWARD_DISPATCH(float3); break;
    case 4: UPSCALE_FORWARD_DISPATCH(float4); break;
    case 5: UPSCALE_FORWARD_DISPATCH(float5); break;
    default: AT_ERROR("gradient_aware_upscale: unsupported channels ", channels);
    }
#undef UPSCALE_FORWARD_DISPATCH

    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gradient_aware_upscale_backward_tensor(
    const torch::Tensor &grad_output,
    int src_h,
    int src_w,
    const std::tuple<float, float, float, float> &src_roi
) {
    DEVICE_GUARD(grad_output);
    CHECK_INPUT(grad_output);

    const int dst_h = grad_output.size(0);
    const int dst_w = grad_output.size(1);
    const int channels = grad_output.size(2);
    const float roi_x1 = std::get<0>(src_roi);
    const float roi_y1 = std::get<1>(src_roi);
    const float roi_x2 = std::get<2>(src_roi);
    const float roi_y2 = std::get<3>(src_roi);

    auto opts = grad_output.options();
    auto grad_render = torch::zeros({src_h, src_w, channels}, opts);
    auto grad_dx = torch::zeros({src_h, src_w, channels}, opts);
    auto grad_dy = torch::zeros({src_h, src_w, channels}, opts);
    auto grad_dxy = torch::zeros({src_h, src_w, channels}, opts);

    dim3 block(16, 16);
    dim3 grid_src(
        (src_w + block.x - 1) / block.x,
        (src_h + block.y - 1) / block.y
    );
    dim3 grid_dst(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

#define UPSCALE_BACKWARD_DISPATCH(T) \
    if constexpr (USE_SRC_CENTRIC_UPSCALE_BACKWARD) { \
        gradient_aware_upscale_backward_src_centric_kernel<T><<<grid_src, block>>>( \
            dst_h, dst_w, src_h, src_w, \
            roi_x1, roi_y1, roi_x2, roi_y2, \
            (T*)grad_output.contiguous().data_ptr<float>(), \
            (T*)grad_render.data_ptr<float>(), \
            (T*)grad_dx.data_ptr<float>(), \
            (T*)grad_dy.data_ptr<float>(), \
            (T*)grad_dxy.data_ptr<float>()); \
    } else { \
        gradient_aware_upscale_backward_kernel<T><<<grid_dst, block>>>( \
            dst_h, dst_w, src_h, src_w, \
            roi_x1, roi_y1, roi_x2, roi_y2, \
            (T*)grad_output.contiguous().data_ptr<float>(), \
            (T*)grad_render.data_ptr<float>(), \
            (T*)grad_dx.data_ptr<float>(), \
            (T*)grad_dy.data_ptr<float>(), \
            (T*)grad_dxy.data_ptr<float>()); \
    }

    switch (channels) {
    case 1: UPSCALE_BACKWARD_DISPATCH(float); break;
    case 2: UPSCALE_BACKWARD_DISPATCH(float2); break;
    case 3: UPSCALE_BACKWARD_DISPATCH(float3); break;
    case 4: UPSCALE_BACKWARD_DISPATCH(float4); break;
    case 5: UPSCALE_BACKWARD_DISPATCH(float5); break;
    default: AT_ERROR("gradient_aware_upscale_backward: unsupported channels ", channels);
    }
#undef UPSCALE_BACKWARD_DISPATCH

    return std::make_tuple(grad_render, grad_dx, grad_dy, grad_dxy);
}