#pragma once
#include <cuda_runtime.h>
#include "floatN.cuh"


template<typename T>
__device__ void atomic_add_scaled(T* dst, int idx, const T& val, float coeff);

template<>
__device__ __forceinline__ void atomic_add_scaled<float>(float* dst, int idx, const float& val, float coeff) {
    atomicAdd(&dst[idx], val * coeff);
}

template<>
__device__ __forceinline__ void atomic_add_scaled<float2>(float2* dst, int idx, const float2& val, float coeff) {
    static_assert(sizeof(float2) == 2 * sizeof(float), "float2 has unexpected padding");
    float* p = (float*)dst;
    atomicAdd(&p[idx*2 + 0], val.x * coeff);
    atomicAdd(&p[idx*2 + 1], val.y * coeff);
}

template<>
__device__ __forceinline__ void atomic_add_scaled<float3>(float3* dst, int idx, const float3& val, float coeff) {
    static_assert(sizeof(float3) == 3 * sizeof(float), "float3 has unexpected padding");
    float* p = (float*)dst;
    atomicAdd(&p[idx*3 + 0], val.x * coeff);
    atomicAdd(&p[idx*3 + 1], val.y * coeff);
    atomicAdd(&p[idx*3 + 2], val.z * coeff);
}

template<>
__device__ __forceinline__ void atomic_add_scaled<float4>(float4* dst, int idx, const float4& val, float coeff) {
    static_assert(sizeof(float4) == 4 * sizeof(float), "float4 has unexpected padding");
    float* p = (float*)dst;
    atomicAdd(&p[idx*4 + 0], val.x * coeff);
    atomicAdd(&p[idx*4 + 1], val.y * coeff);
    atomicAdd(&p[idx*4 + 2], val.z * coeff);
    atomicAdd(&p[idx*4 + 3], val.w * coeff);
}

template<>
__device__ __forceinline__ void atomic_add_scaled<float5>(float5* dst, int idx, const float5& val, float coeff) {
    static_assert(sizeof(float5) == 5 * sizeof(float), "float5 has unexpected padding");
    float* p = (float*)dst;
    #pragma unroll
    for (int i = 0; i < 5; ++i)
        atomicAdd(&p[idx*5 + i], val.data[i] * coeff);
}
