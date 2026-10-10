#ifndef GGML_SYCL_SAGE_ATTN_HPP
#define GGML_SYCL_SAGE_ATTN_HPP

#include "common.hpp"

// Static check for the SYCL SageAttention kernel: SIMD16 DPAS device, AUTO mode (int8 QK^T, f16 PV),
// head_dim 64 or 128, contiguous f32 Q/K and f16 V, no mask.
bool ggml_sycl_sage_attn_supported(int device, const ggml_tensor * op);

// Output bytes plus the int8 Q/K, their scales and the K channel means, reserved inside the op's own
// allocation so the graph allocator sees the scratch's full lifetime.
size_t ggml_sycl_sage_attn_get_alloc_size(const ggml_tensor * op);

// Run SageAttention like the CUDA backend's Ampere path: Q is quantized to int8 per 32 tokens and
// mean-smoothed K per 64 tokens, QK^T runs on int8 DPAS and PV on f16 DPAS with f32 accumulation.
void ggml_sycl_sage_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_SAGE_ATTN_HPP
