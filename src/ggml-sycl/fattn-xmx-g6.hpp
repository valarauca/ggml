#ifndef GGML_SYCL_FATTN_XMX_G6_HPP
#define GGML_SYCL_FATTN_XMX_G6_HPP

#include "common.hpp"

// Q heads per KV head the g6 kernel is written for.
static constexpr int XMX_G6_R = 6;

// Inputs of the g6 prefill kernel. Q is the f16 copy fattn_xmx_g6_stage_q() writes, K and V are f16
// with the token, head and batch strides in elements, rows is fattn_xmx_g6_mask_rows() per query row
// (null without a mask), and dst is the contiguous [DV, H, nq, mb] output.
struct fattn_xmx_g6_params {
    const sycl::half * Q;
    const sycl::half * K;
    const sycl::half * V;
    const sycl::half * mask;
    const sycl::int2 * rows;
    float *            dst;
    int64_t            k_s1, k_s2, k_s3;
    int64_t            v_s1, v_s2, v_s3;
    int64_t            mask_s1;
    int                nq, nkv, H, Hkv, mb;
    float              scale;
    int                tq;
};

// Tokens per work-group for a KV length: 16 from 2048 keys on (each K/V tile serves 96 query rows),
// 8 below, where the shorter work-groups balance the causal triangle better.
int fattn_xmx_g6_tq(int nkv);

// Elements of the staged Q: nq rounded up to 8 rows, for every head and batch, at head_dim 256.
size_t fattn_xmx_g6_q_elems(int64_t nq, int64_t H, int64_t mb);

// Strided f32 Q -> f16 [mb][Hkv][nq / tq][6][tq][256], so the 16 queries one sub-group scores are 16
// consecutive rows. Rows past nq are zero.
void fattn_xmx_g6_stage_q(const char * src, sycl::half * dst, int64_t nq, int64_t H, int64_t Hkv, int64_t mb, size_t nb1,
                          size_t nb2, size_t nb3, int tq, dpct::queue_ptr stream);

// For each query row of an f16 mask: one past the last key that is not -inf, and the first key that is
// not 0 (nkv if none). A row is causal-shaped when the second is at least the first.
void fattn_xmx_g6_mask_rows(const sycl::half * mask, sycl::int2 * rows, int nq, int nkv, int64_t mask_s1,
                            dpct::queue_ptr stream);

// Head_dim 256 prefill attention for 6 Q heads per KV head: one work-group per p.tq tokens and KV head.
void fattn_xmx_g6_launch(dpct::queue_ptr stream, const fattn_xmx_g6_params & p);

#endif // GGML_SYCL_FATTN_XMX_G6_HPP
