#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "floatN.cuh"

// true = src-centric (no atomics), false = dst-centric (with atomics)
constexpr bool USE_SRC_CENTRIC_UPSCALE_BACKWARD = true;

template<typename T>
__global__ void gradient_aware_upscale_kernel(
    const int batch_size,
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float* __restrict__ roi,  // [B, 4]: x1, y1, x2, y2 per batch
    const T* __restrict__ render,
    const T* __restrict__ dx,
    const T* __restrict__ dy,
    const T* __restrict__ dxy,
    T* __restrict__ output
);

template<typename T>
__global__ void gradient_aware_upscale_backward_kernel(
    const int batch_size,
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float* __restrict__ roi,
    const T* __restrict__ grad_output,  // [dst_H, dst_W] of float3
    T* __restrict__ grad_render,        // [src_H, src_W] of float3
    T* __restrict__ grad_dx,
    T* __restrict__ grad_dy,
    T* __restrict__ grad_dxy
);
template<typename T>
__global__ void gradient_aware_upscale_backward_src_centric_kernel(
    const int batch_size,
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float* __restrict__ roi,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_render,
    T* __restrict__ grad_dx,
    T* __restrict__ grad_dy,
    T* __restrict__ grad_dxy
);
