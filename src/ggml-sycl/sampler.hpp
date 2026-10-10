#pragma once

#include "common.hpp"

void ggml_sycl_denoise(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_sample_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
