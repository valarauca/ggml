// A/B harness for SageAttention across backends.
//
// Runs GGML_OP_SAGE_ATTN on identical inputs on every GPU backend that supports it (CUDA as the
// reference implementation, SYCL as the port) and compares each against an exact CPU flash
// attention, and the SYCL result against CUDA. SYCL's flash attention is timed alongside.
//
// usage: test-sage-ab [--gpu-only] [shape-name-filter]
//
// --gpu-only skips the CPU reference and CUDA, for profiling the SYCL kernels.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct sage_shape {
    const char * name;
    int64_t      D, lq, lk, H, Hk;
};

struct sage_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_tensor *         q   = nullptr;
    ggml_tensor *         k   = nullptr;
    ggml_tensor *         v   = nullptr;
    ggml_tensor *         out = nullptr;

    ~sage_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

enum class op_kind { sage, flash };

static bool build(sage_graph & g, const sage_shape & s, op_kind kind, ggml_backend_t backend, const std::vector<float> & hq,
                  const std::vector<float> & hk, const std::vector<ggml_fp16_t> & hv) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    g.q                 = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, s.D, s.lq, s.H, 1);
    g.k                 = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, s.D, s.lk, s.Hk, 1);
    g.v                 = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, s.D, s.lk, s.Hk, 1);
    const float scale   = 1.0f / std::sqrt((float) s.D);
    if (kind == op_kind::sage) {
        g.out = ggml_sage_attn(g.ctx, g.q, g.k, g.v, scale, GGML_SAGE_ATTN_AUTO);
    } else {
        ggml_tensor * k16 = ggml_cast(g.ctx, g.k, GGML_TYPE_F16);
        g.out             = ggml_flash_attn_ext(g.ctx, g.q, k16, g.v, nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(g.out, GGML_PREC_F32);
    }
    if (!ggml_backend_supports_op(backend, g.out)) {
        return false;
    }
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    ggml_backend_tensor_set(g.q, hq.data(), 0, ggml_nbytes(g.q));
    ggml_backend_tensor_set(g.k, hk.data(), 0, ggml_nbytes(g.k));
    ggml_backend_tensor_set(g.v, hv.data(), 0, ggml_nbytes(g.v));
    return true;
}

static std::vector<float> read_out(const sage_graph & g) {
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
    return out;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double err = 0.0, nrm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = (double) a[i] - (double) ref[i];
        err += e * e;
        nrm += (double) ref[i] * ref[i];
    }
    return nrm > 0.0 ? err / nrm : err;
}

static double time_ms(ggml_backend_t backend, const sage_graph & g) {
    ggml_backend_graph_compute(backend, g.gf);
    ggml_backend_synchronize(backend);
    const int  iters = 10;
    const auto t0    = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, g.gf);
    }
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main(int argc, char ** argv) {
    const char * filter   = nullptr;
    bool         gpu_only = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu-only") == 0) {
            gpu_only = true;
        } else {
            filter = argv[i];
        }
    }
    ggml_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads");
    if (set_threads) {
        set_threads(cpu, (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    }
    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t sycl_dev = ggml_backend_dev_by_name("SYCL0");
    ggml_backend_t     cuda     = cuda_dev && !gpu_only ? ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    ggml_backend_t     sycl     = sycl_dev ? ggml_backend_dev_init(sycl_dev, nullptr) : nullptr;
    if (!sycl) {
        fprintf(stderr, "no SYCL0 device\n");
        return 1;
    }

    const sage_shape shapes[] = {
        { "dit-4096x4096-h24",    128, 4096, 4096, 24, 24 },
        { "dit-4096x4121-h24",    128, 4096, 4121, 24, 24 },
        { "d64-4096x4096-h16",     64, 4096, 4096, 16, 16 },
        { "gqa-1024x1024-h32-8",  128, 1024, 1024, 32,  8 },
        { "tail-77x333-h8",       128,   77,  333,  8,  8 },
    };
    const double nmse_max = 1e-3;

    int failures = 0;
    printf("%-22s %12s %12s %12s %10s %10s %10s\n", "shape", "sycl/cpu", "cuda/cpu", "sycl/cuda", "sycl ms", "cuda ms",
           "sycl fa ms");
    for (const sage_shape & s : shapes) {
        if (filter && !strstr(s.name, filter)) {
            continue;
        }
        std::mt19937                    rng(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float>              hq(s.D * s.lq * s.H), hk(s.D * s.lk * s.Hk);
        std::vector<ggml_fp16_t>        hv(s.D * s.lk * s.Hk);
        for (auto & x : hq) {
            x = nd(rng);
        }
        for (auto & x : hk) {
            x = nd(rng) + 0.5f;
        }
        for (auto & x : hv) {
            x = ggml_fp32_to_fp16(nd(rng));
        }

        std::vector<float> ref;
        if (!gpu_only) {
            sage_graph ref_g;
            build(ref_g, s, op_kind::flash, cpu, hq, hk, hv);
            ggml_backend_graph_compute(cpu, ref_g.gf);
            ref = read_out(ref_g);
        }

        sage_graph sy;
        if (!build(sy, s, op_kind::sage, sycl, hq, hk, hv)) {
            printf("%-22s SYCL does not support this shape\n", s.name);
            ++failures;
            continue;
        }
        const double             sy_ms  = time_ms(sycl, sy);
        const std::vector<float> sy_out = read_out(sy);
        const double             sy_err = gpu_only ? NAN : nmse(sy_out, ref);

        double cu_err = NAN, cross = NAN, cu_ms = NAN;
        if (cuda) {
            sage_graph cu;
            if (build(cu, s, op_kind::sage, cuda, hq, hk, hv)) {
                cu_ms                          = time_ms(cuda, cu);
                const std::vector<float> cu_out = read_out(cu);
                cu_err                         = nmse(cu_out, ref);
                cross                          = nmse(sy_out, cu_out);
            }
        }

        sage_graph fa;
        build(fa, s, op_kind::flash, sycl, hq, hk, hv);
        const double fa_ms = time_ms(sycl, fa);

        const bool bad = !gpu_only && !(sy_err <= nmse_max);
        failures += bad;
        printf("%-22s %12.3e %12.3e %12.3e %10.3f %10.3f %10.3f%s\n", s.name, sy_err, cu_err, cross, sy_ms, cu_ms, fa_ms,
               bad ? "  FAIL" : "");
    }

    if (cuda) {
        ggml_backend_free(cuda);
    }
    ggml_backend_free(sycl);
    ggml_backend_free(cpu);
    printf("%s\n", failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
