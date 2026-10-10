#include "sampler.hpp"

#include "ggml-wanghash.h"

#define SYCL_SAMPLER_BLOCK_SIZE 256

enum sampler_cfg { CFG_NONE, CFG_TEXT, CFG_IMAGE, CFG_TEXT_IMAGE };

static void denoise_f32_kernel(const float * __restrict__ x, const float * __restrict__ c,
                               const float * __restrict__ u, const float * __restrict__ iu,
                               const float * __restrict__ m, const float * __restrict__ in,
                               const float * __restrict__ p, float * __restrict__ out, const int cfg,
                               const int64_t k, const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t) item.get_global_id(0);
    if (i >= k) {
        return;
    }
    const float c_skip  = p[0];
    const float c_out   = p[1];
    const float txt_cfg = p[2];
    const float img_cfg = p[3];

    float pred = c[i];
    if (cfg == CFG_TEXT_IMAGE) {
        pred = iu[i] + img_cfg * (u[i] - iu[i]) + txt_cfg * (c[i] - u[i]);
    } else if (cfg == CFG_TEXT) {
        pred = u[i] + txt_cfg * (c[i] - u[i]);
    } else if (cfg == CFG_IMAGE) {
        pred = iu[i] + txt_cfg * (c[i] - iu[i]);
    }
    float d = pred * c_out + x[i] * c_skip;
    if (m) {
        d = d * m[i] + in[i] * (1.0f - m[i]);
    }
    out[i] = d;
}

static void euler_f32_kernel(const float * __restrict__ x, const float * __restrict__ den,
                             const float * __restrict__ p, float * __restrict__ out, const int64_t k,
                             const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t) item.get_global_id(0);
    if (i >= k) {
        return;
    }
    const float sigma = p[0];
    const float dt    = p[1] - p[0];
    const float d     = (x[i] - den[i]) / sigma;
    out[i]            = x[i] + d * dt;
}

static void euler_ancestral_f32_kernel(const float * __restrict__ x, const float * __restrict__ den,
                                       const float * __restrict__ p, float * __restrict__ out,
                                       const uint32_t seed, const int64_t k, const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t) item.get_global_id(0);
    if (i >= k) {
        return;
    }
    const float sigma_up = p[3];
    const float r        = p[2] / p[0];
    float       v        = r * x[i] + (1.0f - r) * den[i];
    if (sigma_up > 0.0f) {
        float u1;
        float u2;
        ggml_wang_uniform2(ggml_wang_key(seed, (uint32_t) p[5]), (uint32_t) i, &u1, &u2);
        const float noise = sycl::sqrt(-2.0f * sycl::log(u1)) * sycl::cos(6.28318530717958647692f * u2);
        v                 = v * p[4] + noise * sigma_up;
    }
    out[i] = v;
}

static const float * sampler_src(const ggml_tensor * dst, int i) {
    return dst->src[i] ? (const float *) dst->src[i]->data : nullptr;
}

void ggml_sycl_denoise(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);

    const float * x  = sampler_src(dst, 0);
    const float * c  = sampler_src(dst, 1);
    const float * u  = sampler_src(dst, 2);
    const float * iu = sampler_src(dst, 3);
    const float * m  = sampler_src(dst, 4);
    const float * in = sampler_src(dst, 5);
    const float * p  = sampler_src(dst, 6);
    float *       o  = (float *) dst->data;
    const int cfg    = u && iu ? CFG_TEXT_IMAGE : u ? CFG_TEXT : iu ? CFG_IMAGE : CFG_NONE;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const int64_t ne         = ggml_nelements(dst);
    const int64_t num_blocks = (ne + SYCL_SAMPLER_BLOCK_SIZE - 1) / SYCL_SAMPLER_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<1>(num_blocks * SYCL_SAMPLER_BLOCK_SIZE, SYCL_SAMPLER_BLOCK_SIZE),
                         [=](sycl::nd_item<1> item) { denoise_f32_kernel(x, c, u, iu, m, in, p, o, cfg, ne, item); });
}

void ggml_sycl_sample_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const float *  x    = sampler_src(dst, 0);
    const float *  den  = sampler_src(dst, 1);
    const float *  p    = sampler_src(dst, 2);
    float *        o    = (float *) dst->data;
    const int32_t  mode = ggml_get_op_params_i32(dst, 0);
    const uint32_t seed = (uint32_t) ggml_get_op_params_i32(dst, 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const int64_t      ne         = ggml_nelements(dst);
    const int64_t      num_blocks = (ne + SYCL_SAMPLER_BLOCK_SIZE - 1) / SYCL_SAMPLER_BLOCK_SIZE;
    const sycl::nd_range<1> range(num_blocks * SYCL_SAMPLER_BLOCK_SIZE, SYCL_SAMPLER_BLOCK_SIZE);
    if (mode == GGML_SAMPLE_STEP_EULER) {
        stream->parallel_for(range, [=](sycl::nd_item<1> item) { euler_f32_kernel(x, den, p, o, ne, item); });
    } else {
        stream->parallel_for(range,
                             [=](sycl::nd_item<1> item) { euler_ancestral_f32_kernel(x, den, p, o, seed, ne, item); });
    }
}
