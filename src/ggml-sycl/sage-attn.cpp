#include "sage-attn.hpp"

#include <climits>

#ifdef GGML_SYCL_FA_XMX

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>

namespace jm     = sycl::ext::oneapi::experimental::matrix;
namespace jmi    = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

static constexpr int SAGE_SG  = 16;
static constexpr int SAGE_TM  = 8;
static constexpr int SAGE_TN  = 16;
static constexpr int SAGE_NSG = 8;
static constexpr int SAGE_BC  = 64;
// Quantization granularity of the CUDA backend's Ampere path: Q per 32 tokens, K per 64 tokens.
static constexpr int SAGE_QB  = 32;
static constexpr int SAGE_KB  = 64;

// Scratch laid out after the output, each block 256-byte aligned like the CUDA SageWorkspace.
struct sage_workspace {
    size_t qi, kp, qs, ks, km, size;

    explicit sage_workspace(const ggml_tensor * dst) {
        const ggml_tensor * q      = dst->src[0];
        const ggml_tensor * k      = dst->src[1];
        const int64_t       D      = q->ne[0];
        const int64_t       lk_pad = GGML_PAD(k->ne[1], SAGE_KB);
        size                       = GGML_PAD(ggml_nbytes(dst), 256);
        qi                         = reserve(ggml_nelements(q));
        kp                         = reserve(D * lk_pad * k->ne[2] * k->ne[3]);
        qs                         = reserve((q->ne[1] + SAGE_QB - 1) / SAGE_QB * q->ne[2] * q->ne[3] * sizeof(float));
        ks                         = reserve(lk_pad / SAGE_KB * k->ne[2] * k->ne[3] * sizeof(float));
        km                         = reserve(D * k->ne[2] * k->ne[3] * sizeof(float));
    }

    size_t reserve(size_t bytes) {
        const size_t offset = size;
        size += GGML_PAD(bytes, 256);
        return offset;
    }
};

struct sage_params {
    const int8_t *     qi;
    const float *      qs;
    const int8_t *     kp;
    const float *      ks;
    const sycl::half * v;
    float *            dst;
    int                lq, lk, lk_pad, H, Hk, B;
    float              scale;
};

template <typename T>
static auto sage_global(const T * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(const_cast<T *>(p));
}

// Per-channel mean of K over all tokens of a KV head, accumulated from 256-token chunks.
template <int D>
static void sage_key_mean(const float * k, float * km, int lk, int heads, dpct::queue_ptr stream) {
    constexpr int CH  = 256;
    const int     nch = (lk + CH - 1) / CH;
    stream->memset(km, 0, sizeof(float) * D * heads);
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) heads * nch, D), sycl::range<2>(1, D)), [=](sycl::nd_item<2> it) {
        const int     hc  = it.get_group(0);
        const int     h   = hc / nch;
        const int     c   = hc % nch;
        const int     d   = it.get_local_id(1);
        const float * src = k + (size_t) h * lk * D;
        float         sum = 0.0f;
        for (int t = c * CH; t < sycl::min(lk, (c + 1) * CH); ++t) {
            sum += src[(size_t) t * D + d];
        }
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> acc(km[(size_t) h * D + d]);
        acc.fetch_add(sum / lk);
    });
}

// Q -> int8 per 32-token block of each head, scale = amax / 127, round to nearest even.
template <int D>
static void sage_quantize_q(const float * q, int8_t * qi, float * qs, int lq, int heads, dpct::queue_ptr stream) {
    constexpr int WG  = 128;
    constexpr int PER = SAGE_QB * D / WG / 4;
    const int     nqb = (lq + SAGE_QB - 1) / SAGE_QB;
    stream->parallel_for(sycl::nd_range<1>((size_t) heads * nqb * WG, WG), [=](sycl::nd_item<1> it) {
        const int     g    = it.get_group(0);
        const int     h    = g / nqb;
        const int     b    = g % nqb;
        const int     t    = it.get_local_id(0);
        const int     n    = sycl::min(SAGE_QB, lq - b * SAGE_QB) * D / 4;
        const float * src  = q + ((size_t) h * lq + (size_t) b * SAGE_QB) * D;
        sycl::float4  vals[PER];
        float         amax = 1e-7f;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = t + i * WG;
            vals[i]       = idx < n ? *(const sycl::float4 *) (src + idx * 4) : sycl::float4(0.0f);
            amax          = sycl::fmax(amax, sycl::fmax(sycl::fmax(sycl::fabs(vals[i].x()), sycl::fabs(vals[i].y())),
                                                        sycl::fmax(sycl::fabs(vals[i].z()), sycl::fabs(vals[i].w()))));
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        if (t == 0) {
            qs[(size_t) h * nqb + b] = amax / 127.0f;
        }
        const float inv = 127.0f / amax;
        int8_t *    dst = qi + ((size_t) h * lq + (size_t) b * SAGE_QB) * D;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = t + i * WG;
            if (idx < n) {
                const sycl::int4 r = (vals[i] * inv).template convert<int, sycl::rounding_mode::rte>();
                *(sycl::vec<int8_t, 4> *) (dst + idx * 4) = r.template convert<int8_t>();
            }
        }
    });
}

// K - mean -> int8 per 64-token block, written straight into the VNNI layout the int8 B operand
// loads: dword (d/4, token) holds channels d..d+3 of one token, tokens padded to lk_pad with zeros.
template <int D>
static void sage_quantize_k(const float * k, const float * km, int8_t * kp, float * ks, int lk, int lk_pad, int heads,
                            dpct::queue_ptr stream) {
    constexpr int WG    = 128;
    constexpr int WORDS = SAGE_KB * D / 4;
    const int     nkb   = lk_pad / SAGE_KB;
    stream->parallel_for(sycl::nd_range<1>((size_t) heads * nkb * WG, WG), [=](sycl::nd_item<1> it) {
        const int     g    = it.get_group(0);
        const int     h    = g / nkb;
        const int     b    = g % nkb;
        const int     t    = it.get_local_id(0);
        const float * src  = k + (size_t) h * lk * D;
        const float * mean = km + (size_t) h * D;
        float         amax = 1e-7f;
        for (int w = t; w < WORDS; w += WG) {
            const int token = b * SAGE_KB + w % SAGE_KB;
            const int d     = w / SAGE_KB * 4;
            if (token < lk) {
                const sycl::float4 v = *(const sycl::float4 *) (src + (size_t) token * D + d) - *(const sycl::float4 *) (mean + d);
                amax = sycl::fmax(amax, sycl::fmax(sycl::fmax(sycl::fabs(v.x()), sycl::fabs(v.y())), sycl::fmax(sycl::fabs(v.z()), sycl::fabs(v.w()))));
            }
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        if (t == 0) {
            ks[(size_t) h * nkb + b] = amax / 127.0f;
        }
        const float inv = 127.0f / amax;
        int32_t *   dst = (int32_t *) (kp + (size_t) h * D * lk_pad);
        for (int w = t; w < WORDS; w += WG) {
            const int    token = b * SAGE_KB + w % SAGE_KB;
            const int    d     = w / SAGE_KB * 4;
            sycl::float4 v(0.0f);
            if (token < lk) {
                v = (*(const sycl::float4 *) (src + (size_t) token * D + d) - *(const sycl::float4 *) (mean + d)) * inv;
            }
            const sycl::vec<int8_t, 4> r = v.template convert<int, sycl::rounding_mode::rte>().template convert<int8_t>();
            dst[(size_t) (d / 4) * lk_pad + token] = r.template as<sycl::vec<int32_t, 1>>()[0];
        }
    });
}

// Same structure as the XMX flash-attention kernel, with QK^T on int8 DPAS: each sub-group owns 8
// query rows (inside one 32-token Q scale block) and streams K/V in 64-key blocks (one K scale each),
// so both dequantization scales are uniform per score tile.
template <int D>
struct sage_kernel {
    static constexpr int DKT = D / 32;
    static constexpr int DVT = D / SAGE_TN;
    static constexpr int CT  = SAGE_BC / SAGE_TN;
    static constexpr int PR  = SAGE_BC / SAGE_NSG;
    static constexpr int KR  = D / 4 / SAGE_NSG;

    sage_params p;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SAGE_SG> };
    }

    void operator()(sycl::nd_item<1> it) const {
        using namespace sycl;
        auto      sg   = it.get_sub_group();
        const int sgid = sg.get_group_linear_id();
        const int lane = sg.get_local_linear_id();

        const int nqb = (p.lq + SAGE_TM * SAGE_NSG - 1) / (SAGE_TM * SAGE_NSG);
        const int g   = it.get_group(0);
        const int qb  = g % nqb;
        const int h   = (g / nqb) % p.H;
        const int b   = g / (nqb * p.H);
        const int q0  = (qb * SAGE_NSG + sgid) * SAGE_TM;
        if (q0 >= p.lq) {
            return;
        }
        const int hk  = b * p.Hk + h / (p.H / p.Hk);
        const int hq  = b * p.H + h;
        const int nqs = (p.lq + SAGE_QB - 1) / SAGE_QB;

        const int8_t * Qh     = p.qi + (size_t) hq * p.lq * D;
        const int8_t * Kh     = p.kp + (size_t) hk * D * p.lk_pad;
        const half *   Vh     = p.v + (size_t) hk * p.lk * D;
        const float *  ksh    = p.ks + (size_t) hk * (p.lk_pad / SAGE_KB);
        const float    qscale = p.qs[(size_t) hq * nqs + q0 / SAGE_QB] * p.scale * 1.4426950408889634f;
        const size_t   kpitch = (size_t) p.lk_pad * 4;

        jm::joint_matrix<sub_group, int8_t, jm::use::a, SAGE_TM, 32, jm::layout::row_major> qa[DKT];
#pragma unroll
        for (int t = 0; t < DKT; ++t) {
            jmi::joint_matrix_load_checked(sg, qa[t], sage_global(Qh), D, p.lq, D, q0, t * 32);
        }

        jm::joint_matrix<sub_group, float, jm::use::accumulator, SAGE_TM, SAGE_TN> o[DVT];
#pragma unroll
        for (int t = 0; t < DVT; ++t) {
            jm::joint_matrix_fill(sg, o[t], 0.0f);
        }
        float m[SAGE_TM];
        float l[SAGE_TM];
#pragma unroll
        for (int i = 0; i < SAGE_TM; ++i) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
        }

        for (int kb = 0; kb < p.lk; kb += SAGE_BC) {
            if (kb + 2 * SAGE_BC <= p.lk) {
                const int8_t * kn = Kh + (size_t) (sgid * KR) * kpitch + (size_t) (kb + SAGE_BC) * 4;
#pragma unroll
                for (int cc = 0; cc < SAGE_BC * 4; cc += 64) {
                    jm::joint_matrix_prefetch<KR, 64>(sg, const_cast<int8_t *>(kn) + cc, kpitch, jm::layout::row_major,
                                                      syclex::properties{ syclex::prefetch_hint_L1 });
                }
#pragma unroll
                for (int cc = 0; cc < D; cc += 32) {
                    jm::joint_matrix_prefetch<PR, 32>(sg, const_cast<half *>(Vh) + (size_t) (kb + SAGE_BC + sgid * PR) * D + cc,
                                                      (size_t) D, jm::layout::row_major,
                                                      syclex::properties{ syclex::prefetch_hint_L1 });
                }
            }

            const float sscale = qscale * ksh[kb / SAGE_KB];
            jm::joint_matrix<sub_group, float, jm::use::accumulator, SAGE_TM, SAGE_TN> s[CT];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                jm::joint_matrix<sub_group, int32_t, jm::use::accumulator, SAGE_TM, SAGE_TN> si;
                jm::joint_matrix_fill(sg, si, 0);
#pragma unroll
                for (int t = 0; t < DKT; ++t) {
                    jm::joint_matrix<sub_group, int8_t, jm::use::b, 32, SAGE_TN, jm::layout::ext_intel_packed> kt;
                    jm::joint_matrix_load(sg, kt, sage_global(Kh + (size_t) (t * 8) * kpitch + (size_t) (kb + c * SAGE_TN) * 4), kpitch);
                    jm::joint_matrix_mad(sg, si, qa[t], kt, si);
                }
                jm::joint_matrix_copy(sg, si, s[c]);
            }

            float bmax[SAGE_TM];
#pragma unroll
            for (int i = 0; i < SAGE_TM; ++i) {
                bmax[i] = -INFINITY;
            }
            if (kb + SAGE_BC <= p.lk) {
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    int i = 0;
                    jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                        x       = x * sscale;
                        bmax[i] = sycl::fmax(bmax[i], x);
                        ++i;
                    });
                }
            } else {
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    const int key = kb + c * SAGE_TN + lane;
                    int       i   = 0;
                    jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                        x       = key < p.lk ? x * sscale : -INFINITY;
                        bmax[i] = sycl::fmax(bmax[i], x);
                        ++i;
                    });
                }
            }

            float alpha[SAGE_TM];
            float mref[SAGE_TM];
#pragma unroll
            for (int i = 0; i < SAGE_TM; ++i) {
                const float mn = sycl::fmax(m[i], reduce_over_group(sg, bmax[i], maximum<float>()));
                mref[i]        = mn == -INFINITY ? 0.0f : mn;
                alpha[i]       = sycl::exp2(m[i] - mref[i]);
                m[i]           = mn;
                l[i] *= alpha[i];
            }

            jm::joint_matrix<sub_group, half, jm::use::a, SAGE_TM, 16, jm::layout::row_major> pa[CT];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                int i = 0;
                jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                    x = sycl::exp2(x - mref[i]);
                    l[i] += x;
                    ++i;
                });
                jm::joint_matrix_copy(sg, s[c], pa[c]);
            }

#pragma unroll
            for (int t = 0; t < DVT; ++t) {
                int i = 0;
                jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                    x *= alpha[i];
                    ++i;
                });
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    jm::joint_matrix<sub_group, half, jm::use::b, 16, SAGE_TN, jm::layout::row_major> vb;
                    jmi::joint_matrix_load_checked(sg, vb, sage_global(Vh), D, p.lk, D, kb + c * 16, t * SAGE_TN);
                    jm::joint_matrix_mad(sg, o[t], pa[c], vb, o[t]);
                }
            }
        }

        float inv[SAGE_TM];
#pragma unroll
        for (int i = 0; i < SAGE_TM; ++i) {
            const float ls = reduce_over_group(sg, l[i], plus<float>());
            inv[i]         = ls > 0.0f ? 1.0f / ls : 0.0f;
        }
        float * dh = p.dst + ((size_t) b * p.lq * p.H + h) * D;
#pragma unroll
        for (int t = 0; t < DVT; ++t) {
            int i = 0;
            jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                x *= inv[i];
                ++i;
            });
            jmi::joint_matrix_store_checked(sg, o[t], sage_global(dh), (size_t) p.H * D, jm::layout::row_major, p.lq, D, q0,
                                            t * SAGE_TN);
        }
    }
};

// Checks that element i of the 8x16 f32 and int32 accumulators and the 8x16 f16 A operand is
// (row i, column lane), which the kernel's per-row softmax state and its accumulator copies rely on.
// It shares the large-GRF property with sage_kernel so both land in the same device image.
struct sage_layout_probe {
    int * bad;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SAGE_SG> };
    }

    template <typename M>
    void check(sycl::sub_group sg, M & mat, size_t lane) const {
        size_t i = 0;
        jmi::joint_matrix_apply(sg, mat, [&](auto &, size_t r, size_t c) {
            if (r != i || c != lane) {
                *bad = 1;
            }
            ++i;
        });
    }

    void operator()(sycl::nd_item<1> it) const {
        auto         sg   = it.get_sub_group();
        const size_t lane = sg.get_local_linear_id();
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, SAGE_TM, SAGE_TN>                           cf;
        jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, SAGE_TM, SAGE_TN>                         ci;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, SAGE_TM, 16, jm::layout::row_major>            a;
        jm::joint_matrix_fill(sg, cf, 0.0f);
        jm::joint_matrix_fill(sg, ci, 0);
        jm::joint_matrix_fill(sg, a, 0.0f);
        check(sg, cf, lane);
        check(sg, ci, lane);
        check(sg, a, lane);
    }
};

static bool sage_layout_ok(int device) {
    static int8_t checked[GGML_SYCL_MAX_DEVICES] = {};
    if (checked[device] != 0) {
        return checked[device] > 0;
    }
    sycl::queue q(dpct::dev_mgr::instance().get_device(device), sycl::property::queue::in_order{});
    int *       bad = sycl::malloc_device<int>(1, q);
    q.memset(bad, 0, sizeof(int));
    q.parallel_for(sycl::nd_range<1>(SAGE_SG, SAGE_SG), sage_layout_probe{ bad });
    int host_bad = 1;
    q.memcpy(&host_bad, bad, sizeof(int)).wait();
    sycl::free(bad, q);
    checked[device] = host_bad ? -1 : 1;
    if (host_bad) {
        GGML_LOG_WARN("%s: unexpected joint_matrix element layout on device %d, SageAttention disabled\n", __func__, device);
    }
    return !host_bad;
}

template <int D>
static void sage_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const int lq = (int) q->ne[1], lk = (int) k->ne[1], lk_pad = (int) GGML_PAD(k->ne[1], SAGE_KB);
    const int H = (int) q->ne[2], Hk = (int) k->ne[2], B = (int) q->ne[3];

    const sage_workspace work(dst);
    char *               data   = (char *) dst->data;
    int8_t *             qi     = (int8_t *) (data + work.qi);
    int8_t *             kp     = (int8_t *) (data + work.kp);
    float *              qs     = (float *) (data + work.qs);
    float *              ks     = (float *) (data + work.ks);
    float *              km     = (float *) (data + work.km);
    dpct::queue_ptr      stream = ctx.stream();

    sage_key_mean<D>((const float *) k->data, km, lk, Hk * B, stream);
    sage_quantize_q<D>((const float *) q->data, qi, qs, lq, H * B, stream);
    sage_quantize_k<D>((const float *) k->data, km, kp, ks, lk, lk_pad, Hk * B, stream);

    sage_params p{};
    p.qi     = qi;
    p.qs     = qs;
    p.kp     = kp;
    p.ks     = ks;
    p.v      = (const sycl::half *) v->data;
    p.dst    = (float *) dst->data;
    p.lq     = lq;
    p.lk     = lk;
    p.lk_pad = lk_pad;
    p.H      = H;
    p.Hk     = Hk;
    p.B      = B;
    memcpy(&p.scale, dst->op_params, sizeof(float));

    const int    nqb    = (lq + SAGE_TM * SAGE_NSG - 1) / (SAGE_TM * SAGE_NSG);
    const size_t groups = (size_t) nqb * H * B;
    stream->parallel_for(sycl::nd_range<1>(groups * SAGE_NSG * SAGE_SG, SAGE_NSG * SAGE_SG), sage_kernel<D>{ p });
}

#endif // GGML_SYCL_FA_XMX

bool ggml_sycl_sage_attn_supported(int device, const ggml_tensor * op) {
#ifndef GGML_SYCL_FA_XMX
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
#else
    const sycl_xe_family_caps & caps = get_xe_family_caps(ggml_sycl_info().devices[device].hw_info.xe_family);
    if (caps.dpas_n != SAGE_SG || !caps.block_2d_io) {
        return false;
    }
    if (static_cast<ggml_sage_attn_mode>(op->op_params[1]) != GGML_SAGE_ATTN_AUTO) {
        return false;
    }
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    if (!q || !k || !v || op->src[3] || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F16 ||
        op->type != GGML_TYPE_F32 || (q->ne[0] != 64 && q->ne[0] != 128)) {
        return false;
    }
    for (const ggml_tensor * t : { q, k, v, op }) {
        if (!ggml_is_contiguous(t) || ggml_nelements(t) > INT_MAX) {
            return false;
        }
    }
    if (q->ne[0] * GGML_PAD(k->ne[1], SAGE_KB) * k->ne[2] * k->ne[3] > INT_MAX) {
        return false;
    }
    return q->ne[0] == k->ne[0] && ggml_are_same_shape(k, v) && q->ne[2] % k->ne[2] == 0 && q->ne[3] == k->ne[3] &&
           op->ne[0] == q->ne[0] && op->ne[1] == q->ne[2] && op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3] &&
           sage_layout_ok(device);
#endif
}

size_t ggml_sycl_sage_attn_get_alloc_size(const ggml_tensor * op) {
#ifndef GGML_SYCL_FA_XMX
    return ggml_nbytes(op);
#else
    return sage_workspace(op).size;
#endif
}

void ggml_sycl_sage_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifndef GGML_SYCL_FA_XMX
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("SageAttention was not compiled into this backend");
#else
    GGML_ASSERT(ggml_sycl_sage_attn_supported(ctx.device, dst));
    if (dst->src[0]->ne[0] == 64) {
        sage_attn<64>(ctx, dst);
    } else {
        sage_attn<128>(ctx, dst);
    }
#endif
}
