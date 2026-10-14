// Head_dim 256 prefill attention for 6 Q heads per KV head (Qwen3.5 / 3.8 27B).
//
// Derived from attn_prompt in zig/kernels/xpu/qwen27/decode.cl of TensorFold
// (https://github.com/ashhart/TensorFold, PR 802), Copyright its authors, licensed under the Apache
// License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified: ported from OpenCL C to
// SYCL, f16 instead of bf16 for K, V and P, causality from the mask tensor instead of positions, f32
// output in ggml's layout, batches, K/V prefetch, and 16-token work-groups.

#include "fattn-xmx-g6.hpp"

#ifdef GGML_SYCL_FA_XMX

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

static constexpr int G6_D  = 256;
static constexpr int G6_SG = 16;
static constexpr int G6_BC = 128;

#ifdef __SYCL_DEVICE_ONLY__
typedef unsigned short g6_us32 __attribute__((ext_vector_type(32)));
typedef unsigned int   g6_u8 __attribute__((ext_vector_type(8)));
typedef unsigned short g6_us8 __attribute__((ext_vector_type(8)));
typedef short          g6_s8 __attribute__((ext_vector_type(8)));
typedef int            g6_i8 __attribute__((ext_vector_type(8)));
typedef float          g6_f8 __attribute__((ext_vector_type(8)));
typedef int            g6_i2 __attribute__((ext_vector_type(2)));

SYCL_EXTERNAL extern "C" g6_us32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(intptr_t base, int width_minus_one,
                                                                                     int height_minus_one,
                                                                                     int pitch_minus_one, g6_i2 coord);
SYCL_EXTERNAL extern "C" g6_u8 __builtin_IB_subgroup_block_read_flat_transform_u16_k16(intptr_t base, int width_minus_one,
                                                                                         int height_minus_one,
                                                                                         int pitch_minus_one, g6_i2 coord);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(intptr_t base, int width_minus_one,
                                                                                    int height_minus_one,
                                                                                    int pitch_minus_one, g6_i2 coord,
                                                                                    int cache_control);
SYCL_EXTERNAL extern "C" void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(intptr_t base, int width_minus_one,
                                                                                   int height_minus_one,
                                                                                   int pitch_minus_one, g6_i2 coord,
                                                                                   int cache_control);
SYCL_EXTERNAL g6_f8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int k, g6_s8 a, g6_i8 b, g6_f8 acc, int operands);

static inline g6_f8 g6_dpas(g6_s8 a, g6_i8 b, g6_f8 acc) {
    return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, acc, 0x400 | 0x800);
}

template <int ROWS>
static inline void g6_prefetch(intptr_t base, int hm1, int pm1, int row) {
#pragma unroll
    for (int x = 0; x < G6_D; x += 32) {
        if constexpr (ROWS == 16) {
            __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(base, G6_D * 2 - 1, hm1, pm1, g6_i2{ x, row }, 4);
        } else {
            __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(base, G6_D * 2 - 1, hm1, pm1, g6_i2{ x, row }, 4);
        }
    }
}
#endif

int fattn_xmx_g6_tq(int nkv) {
    return nkv >= 2048 ? 16 : 8;
}

// One work-group covers TQ tokens and the 6 Q heads of one KV head (6 TQ query rows) with TQ
// sub-groups, so every K and V tile it loads serves all of them. S is computed transposed (K tile as
// A, 16 queries as B), which puts one query in each lane and keeps the softmax inside the lane. The
// first 6 TQ / 8 sub-groups score a block of 128 keys, each 16 queries (16 / TQ heads by TQ tokens)
// against one half of 64 keys, and the two halves meet in SLM. P goes to SLM as 8-row A tiles, and
// every sub-group then accumulates P V for 256 / TQ output columns. K and V come in through 2D block
// loads (32 key rows of K at a time, V in VNNI form), and each sub-group prefetches its share of the
// next block into L1. With QF the work-group first converts its 6 TQ rows of ggml's f32 Q into its block
// of the f16 copy, otherwise fattn_xmx_g6_stage_q() wrote the copy.
template <int TQ, bool QF>
struct fattn_xmx_g6_kernel {
    static constexpr int R   = XMX_G6_R;
    static constexpr int NSG = TQ;
    static constexpr int NQK = 2 * R * TQ / G6_SG;
    static constexpr int HPG = G6_SG / TQ;
    static constexpr int RT  = TQ / 8;
    static constexpr int OT  = G6_D / G6_SG / NSG;
    static constexpr int DT  = G6_D / G6_SG;
    static constexpr int PC  = G6_BC / 16;

    fattn_xmx_g6_params                     p;
    sycl::local_accessor<unsigned short, 1> pl;
    sycl::local_accessor<float, 1>          red;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<G6_SG> };
    }

    void operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
        auto      sg   = it.get_sub_group();
        const int sgid = sg.get_group_linear_id();
        const int lane = sg.get_local_linear_id();

        const int ntb = (p.nq + TQ - 1) / TQ;
        const int gid = it.get_group(0);
        const int tb  = gid % ntb;
        const int hk  = gid / ntb % p.Hkv;
        const int b   = gid / (ntb * p.Hkv);
        const int t0  = tb * TQ;

        const bool qk = sgid < NQK;
        const int  g  = sgid / 2;
        const int  kh = sgid % 2;
        const int  tq = lane % TQ;
        const int  hq = g * HPG + lane / TQ;
        const int  t  = t0 + tq;

        const sycl::int2 rw      = p.rows ? p.rows[sycl::min(t, p.nq - 1)] : sycl::int2(p.nkv, p.nkv);
        const int        e_t     = rw.x();
        const int        kv_end  = sycl::reduce_over_group(sg, e_t, sycl::maximum<int>());
        const int        kv_zero = sycl::reduce_over_group(sg, rw.y(), sycl::minimum<int>());
        const bool       causal  = sycl::all_of_group(sg, rw.y() >= e_t);

        sycl::half * const Qblk = p.Q + ((size_t) (b * p.Hkv + hk) * ntb + tb) * R * TQ * G6_D;
        const sycl::half * Qrow = Qblk + (g * 16 + lane) * G6_D;
        if constexpr (QF) {
            for (int e = it.get_local_linear_id(); e < R * TQ * G6_D / 4; e += TQ * G6_SG) {
                const int    d  = e % (G6_D / 4) * 4;
                const int    rr = e / (G6_D / 4);
                const int    tr = t0 + rr % TQ;
                sycl::float4 v(0.0f);
                if (tr < p.nq) {
                    v = *(const sycl::float4 *) (p.Qf + b * p.q_s3 + (int64_t) (hk * R + rr / TQ) * p.q_s2 +
                                                 (int64_t) tr * p.q_s1 + d);
                }
                *(sycl::vec<sycl::half, 4> *) (Qblk + rr * G6_D + d) = v.convert<sycl::half>();
            }
            sycl::group_barrier(it.get_group());
        }
        const intptr_t     Kb   = (intptr_t) (p.K + b * p.k_s3 + hk * p.k_s2);
        const intptr_t     Vb   = (intptr_t) (p.V + b * p.v_s3 + hk * p.v_s2);
        const int          kp   = (int) (p.k_s1 * 2) - 1;
        const int          vp   = (int) (p.v_s1 * 2) - 1;
        const int          hm1  = p.nkv - 1;
        const sycl::half * mr   = p.mask ? p.mask + (int64_t) sycl::min(t, p.nq - 1) * p.mask_s1 : nullptr;

        auto MX = red.get_multi_ptr<sycl::access::decorated::no>();
        auto SM = MX + NQK * 16;
        auto AL = MX + 2 * NQK * 16;
        auto P  = pl.get_multi_ptr<sycl::access::decorated::no>();
        auto Pd = pl.get_multi_ptr<sycl::access::decorated::yes>().get();

        const float log2e   = 1.4426950408889634f;
        const float s_scale = p.scale * log2e;
        const int   kh0     = kh * (G6_BC / 2);
        const int   pbase   = ((hq * RT + tq / 8) * PC) * 128 + tq % 8 * 16;

        g6_f8 o[R][RT][OT];
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
                for (int ot = 0; ot < OT; ++ot) {
                    o[i][rt][ot] = 0.0f;
                }
            }
        }
        float m = -INFINITY;
        float l = 0.0f;

        for (int kb = 0; kb < kv_end; kb += G6_BC) {
            if (kb + G6_BC < kv_end) {
                g6_prefetch<G6_BC / NSG>(Kb, hm1, kp, kb + G6_BC + sgid * (G6_BC / NSG));
                g6_prefetch<G6_BC / NSG>(Vb, hm1, vp, kb + G6_BC + sgid * (G6_BC / NSG));
            }
            const int k0 = kb + kh0;
            g6_f8     s[8];
            float     mx = -INFINITY;
            if (qk) {
#pragma unroll
                for (int bt = 0; bt < 8; ++bt) {
                    s[bt] = 0.0f;
                }
#pragma unroll 4
                for (int ds = 0; ds < DT; ++ds) {
                    const g6_i8 qb = *(const g6_i8 *) (Qrow + ds * 16);
#pragma unroll
                    for (int h2 = 0; h2 < 2; ++h2) {
                        const g6_us32 ka = __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
                            Kb, G6_D * 2 - 1, hm1, kp, g6_i2{ ds * 16, k0 + h2 * 32 });
#pragma unroll
                        for (int bt = 0; bt < 4; ++bt) {
                            g6_s8 a;
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                a[r] = (short) ka[bt * 8 + r];
                            }
                            s[h2 * 4 + bt] = g6_dpas(a, qb, s[h2 * 4 + bt]);
                        }
                    }
                }

                if (kb + G6_BC <= kv_zero) {
#pragma unroll
                    for (int bt = 0; bt < 8; ++bt) {
                        s[bt] *= s_scale;
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            mx = sycl::fmax(mx, s[bt][r]);
                        }
                    }
                } else if (causal) {
#pragma unroll
                    for (int bt = 0; bt < 8; ++bt) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float v = k0 + bt * 8 + r < e_t ? s[bt][r] * s_scale : -INFINITY;
                            s[bt][r]      = v;
                            mx            = sycl::fmax(mx, v);
                        }
                    }
                } else {
                    const bool full = kb + G6_BC <= p.nkv;
#pragma unroll
                    for (int bt = 0; bt < 8; ++bt) {
                        const g6_s8 mv = mr ? *(const g6_s8 *) (mr + k0 + bt * 8) : g6_s8(0);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            float v = s[bt][r] * s_scale + (float) sycl::bit_cast<sycl::half>(mv[r]) * log2e;
                            if (!full && k0 + bt * 8 + r >= p.nkv) {
                                v = -INFINITY;
                            }
                            s[bt][r] = v;
                            mx       = sycl::fmax(mx, v);
                        }
                    }
                }
                MX[sgid * 16 + lane] = mx;
            }
            sycl::group_barrier(it.get_group());

            float alpha = 1.0f;
            if (qk) {
                const float mn  = sycl::fmax(m, sycl::fmax(mx, MX[(sgid ^ 1) * 16 + lane]));
                float       sum = 0.0f;
                alpha           = mn == -INFINITY ? 1.0f : sycl::native::exp2(m - mn);
#pragma unroll
                for (int bt = 0; bt < 8; ++bt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float pv  = mn == -INFINITY ? 0.0f : sycl::native::exp2(s[bt][r] - mn);
                        const int   key = kh0 + bt * 8 + r;
                        sum += pv;
                        P[pbase + key / 16 * 128 + key % 16] = sycl::bit_cast<unsigned short>((sycl::half) pv);
                    }
                }
                SM[sgid * 16 + lane] = sum;
                if (kh == 0) {
                    AL[hq * TQ + tq] = alpha;
                }
                m = mn;
            }
            sycl::group_barrier(it.get_group());
            if (qk) {
                l = l * alpha + SM[(sgid & ~1) * 16 + lane] + SM[(sgid | 1) * 16 + lane];
            }

#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float al = AL[i * TQ + rt * 8 + r];
#pragma unroll
                        for (int ot = 0; ot < OT; ++ot) {
                            o[i][rt][ot][r] *= al;
                        }
                    }
                }
            }
            for (int c = 0; c < PC; ++c) {
                if (kb + c * 16 >= kv_end) {
                    break;
                }
                g6_s8 pa[R][RT];
#pragma unroll
                for (int i = 0; i < R; ++i) {
#pragma unroll
                    for (int rt = 0; rt < RT; ++rt) {
                        pa[i][rt] = (g6_s8) __spirv_SubgroupBlockReadINTEL<g6_us8>(Pd + ((i * RT + rt) * PC + c) * 128);
                    }
                }
#pragma unroll
                for (int ot = 0; ot < OT; ++ot) {
                    const g6_u8 vw = __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
                        Vb, G6_D * 2 - 1, hm1, vp, g6_i2{ (sgid * OT + ot) * 16, kb + c * 16 });
#pragma unroll
                    for (int i = 0; i < R; ++i) {
#pragma unroll
                        for (int rt = 0; rt < RT; ++rt) {
                            o[i][rt][ot] = g6_dpas(pa[i][rt], (g6_i8) vw, o[i][rt][ot]);
                        }
                    }
                }
            }
        }

        sycl::group_barrier(it.get_group());
        if (qk && kh == 0) {
            AL[hq * TQ + tq] = l > 0.0f ? 1.0f / l : 0.0f;
        }
        sycl::group_barrier(it.get_group());

#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int tr = rt * 8 + r;
                    if (t0 + tr < p.nq) {
                        const float inv = AL[i * TQ + tr];
                        float * dh = p.dst + (((size_t) b * p.nq + t0 + tr) * p.H + hk * R + i) * G6_D + sgid * OT * 16 + lane;
#pragma unroll
                        for (int ot = 0; ot < OT; ++ot) {
                            dh[ot * 16] = o[i][rt][ot][r] * inv;
                        }
                    }
                }
            }
        }
#else
        GGML_UNUSED(it);
#endif
    }
};

template <int TQ, bool QF>
static void fattn_xmx_g6_submit(dpct::queue_ptr stream, const fattn_xmx_g6_params & p) {
    constexpr int NQK    = 2 * XMX_G6_R * TQ / G6_SG;
    const size_t  groups = (size_t) (p.nq + TQ - 1) / TQ * p.Hkv * p.mb;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<unsigned short, 1> pl(sycl::range<1>(XMX_G6_R * G6_BC * TQ), h);
        sycl::local_accessor<float, 1>          red(sycl::range<1>(2 * NQK * 16 + XMX_G6_R * TQ), h);
        h.parallel_for(sycl::nd_range<1>(groups * TQ * G6_SG, TQ * G6_SG), fattn_xmx_g6_kernel<TQ, QF>{ p, pl, red });
    });
}

size_t fattn_xmx_g6_q_elems(int64_t nq, int64_t H, int64_t mb) {
    return (size_t) ((nq + 15) / 16 * 16) * H * mb * G6_D;
}

void fattn_xmx_g6_stage_q(const char * src, sycl::half * dst, int64_t nq, int64_t H, int64_t Hkv, int64_t mb, size_t nb1,
                          size_t nb2, size_t nb3, int tq, dpct::queue_ptr stream) {
    const int64_t TQ  = tq;
    const int64_t R   = H / Hkv;
    const int64_t ntb = (nq + TQ - 1) / TQ;
    const int64_t n4  = ntb * TQ * H * mb * G6_D / 4;
    const bool    v4  = (uintptr_t) src % 16 == 0 && nb1 % 16 == 0 && nb2 % 16 == 0 && nb3 % 16 == 0;
    stream->parallel_for(sycl::range<1>(n4), [=](sycl::id<1> id) {
        int64_t       i  = id[0];
        const int64_t d  = i % (G6_D / 4) * 4;
        i /= G6_D / 4;
        const int64_t tq = i % TQ;
        i /= TQ;
        const int64_t hq = i % R;
        i /= R;
        const int64_t tb = i % ntb;
        i /= ntb;
        const int64_t hk = i % Hkv;
        const int64_t b  = i / Hkv;
        const int64_t q  = tb * TQ + tq;
        sycl::float4  v(0.0f);
        if (q < nq) {
            const char * s = src + d * sizeof(float) + q * nb1 + (hk * R + hq) * nb2 + b * nb3;
            if (v4) {
                v = *(const sycl::float4 *) s;
            } else {
                v = sycl::float4(((const float *) s)[0], ((const float *) s)[1], ((const float *) s)[2], ((const float *) s)[3]);
            }
        }
        *(sycl::vec<sycl::half, 4> *) (dst + id[0] * 4) = v.convert<sycl::half>();
    });
}

void fattn_xmx_g6_mask_rows(const sycl::half * mask, sycl::int2 * rows, int nq, int nkv, int64_t mask_s1,
                            dpct::queue_ptr stream) {
    constexpr int WG = 256;
    const int     nv = (uintptr_t) mask % 16 == 0 && mask_s1 % 8 == 0 ? nkv / 8 : 0;
    stream->parallel_for(sycl::nd_range<1>((size_t) nq * WG, WG), [=](sycl::nd_item<1> it) {
        const int          r     = it.get_group(0);
        const int          li    = it.get_local_linear_id();
        const sycl::half * mr    = mask + (int64_t) r * mask_s1;
        int                last  = 0;
        int                first = nkv;
        for (int k8 = li; k8 < nv; k8 += WG) {
            const sycl::uint4 v = *(const sycl::uint4 *) (mr + k8 * 8);
            for (int e = 0; e < 8; ++e) {
                const uint32_t x = (v[e / 2] >> (e % 2 * 16)) & 0xFFFF;
                if (x != 0xFC00) {
                    last = sycl::max(last, k8 * 8 + e + 1);
                }
                if ((x & 0x7FFF) != 0) {
                    first = sycl::min(first, k8 * 8 + e);
                }
            }
        }
        for (int k = nv * 8 + li; k < nkv; k += WG) {
            const uint16_t x = sycl::bit_cast<uint16_t>(mr[k]);
            if (x != 0xFC00) {
                last = sycl::max(last, k + 1);
            }
            if ((x & 0x7FFF) != 0) {
                first = sycl::min(first, k);
            }
        }
        last  = sycl::reduce_over_group(it.get_group(), last, sycl::maximum<int>());
        first = sycl::reduce_over_group(it.get_group(), first, sycl::minimum<int>());
        if (li == 0) {
            rows[r] = sycl::int2(last, first);
        }
    });
}

void fattn_xmx_g6_launch(dpct::queue_ptr stream, const fattn_xmx_g6_params & p) {
    if (p.Qf) {
        if (p.tq == 16) {
            fattn_xmx_g6_submit<16, true>(stream, p);
        } else {
            fattn_xmx_g6_submit<8, true>(stream, p);
        }
    } else {
        if (p.tq == 16) {
            fattn_xmx_g6_submit<16, false>(stream, p);
        } else {
            fattn_xmx_g6_submit<8, false>(stream, p);
        }
    }
}

#endif // GGML_SYCL_FA_XMX
