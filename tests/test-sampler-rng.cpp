// Statistics and cross-backend parity of the stateless WangHash noise in GGML_OP_SAMPLE_STEP.
//
// With x = denoised = 0, alpha = 1 and sigma_up = 1 the Euler-ancestral step outputs pure noise,
// so the op itself is the generator under test. Checks moments, a 64-bin chi-square against the
// normal CDF, correlation across neighbouring elements, steps and seeds, determinism, and that
// every GPU backend matches the CPU reference. A small analytic case checks the Euler and
// Euler-ancestral formulas, including the hash, against an independent double-precision model.
//
// usage: test-sampler-rng [--gpu-only] [--n log2-elements]
//
// --gpu-only skips the CPU reference and the CPU-vs-GPU comparison, for profiling.

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct noise_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_tensor *         p   = nullptr;
    ggml_tensor *         out = nullptr;

    ~noise_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static bool build(noise_graph & g, ggml_backend_t backend, int64_t n, uint32_t seed) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * x     = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n);
    ggml_tensor * den   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n);
    g.p                 = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 6);
    g.out               = ggml_sample_step(g.ctx, x, den, g.p, GGML_SAMPLE_STEP_EULER_ANCESTRAL, seed);
    if (!ggml_backend_supports_op(backend, g.out)) {
        return false;
    }
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    ggml_backend_tensor_memset(x, 0, 0, ggml_nbytes(x));
    ggml_backend_tensor_memset(den, 0, 0, ggml_nbytes(den));
    return true;
}

static std::vector<float> run(ggml_backend_t backend, noise_graph & g, uint32_t step) {
    const float p[6] = { 1.0f, 0.5f, 0.5f, 1.0f, 1.0f, (float) step };
    ggml_backend_tensor_set(g.p, p, 0, sizeof(p));
    ggml_backend_graph_compute(backend, g.gf);
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
    return out;
}

static double corr(const std::vector<float> & a, const std::vector<float> & b, size_t lag) {
    const size_t n  = a.size() - lag;
    double       sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = a[i], y = b[i + lag];
        sa += x;
        sb += y;
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    const double ma = sa / n, mb = sb / n;
    return (sab / n - ma * mb) / std::sqrt((saa / n - ma * ma) * (sbb / n - mb * mb));
}

struct moments {
    double mean, var, skew, kurt, chi2;
    bool   finite;
};

static moments stats(const std::vector<float> & v) {
    moments m{};
    m.finite         = true;
    const double n   = (double) v.size();
    double       s1  = 0;
    for (float x : v) {
        m.finite = m.finite && std::isfinite(x);
        s1 += x;
    }
    m.mean = s1 / n;
    double s2 = 0, s3 = 0, s4 = 0;
    std::vector<int64_t> bins(64, 0);
    for (float x : v) {
        const double d = x - m.mean;
        s2 += d * d;
        s3 += d * d * d;
        s4 += d * d * d * d;
        const double u = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        bins[std::min(63, std::max(0, (int) (u * 64.0)))]++;
    }
    m.var            = s2 / n;
    m.skew           = (s3 / n) / std::pow(m.var, 1.5);
    m.kurt           = (s4 / n) / (m.var * m.var);
    const double exp = n / 64.0;
    for (int64_t c : bins) {
        m.chi2 += (c - exp) * (c - exp) / exp;
    }
    return m;
}

static double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, (double) std::fabs(a[i] - b[i]));
    }
    return e;
}

static uint32_t ref_wang(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s ^= s >> 4;
    s *= 0x27d4eb2du;
    s ^= s >> 15;
    return s;
}

static double ref_normal(uint32_t seed, uint32_t step, uint32_t i) {
    const uint32_t key = ref_wang(seed ^ ref_wang(step + 0x9e3779b9u));
    const uint32_t h1  = ref_wang(key ^ (2u * i));
    const uint32_t h2  = ref_wang(h1 ^ (2u * i + 1u));
    const double   u1  = ((double) (h1 >> 8) + 0.5) / 16777216.0;
    const double   u2  = (double) (h2 >> 8) / 16777216.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

static double analytic_err(ggml_backend_t backend, ggml_sample_step_mode mode) {
    const int64_t  n    = 1000;
    const uint32_t seed = 7u;
    const float    p[6] = { 0.9f, 0.7f, 0.6f, 0.4f, 1.25f, 5.0f };
    std::vector<float> hx(n), hd(n);
    for (int64_t i = 0; i < n; ++i) {
        hx[i] = std::sin(0.37 * i) * 2.0f;
        hd[i] = std::cos(0.11 * i);
    }
    ggml_init_params ip  = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    x   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor *    den = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor *    pt  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
    ggml_tensor *    out = ggml_sample_step(ctx, x, den, pt, mode, seed);
    ggml_cgraph *    gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(den, hd.data(), 0, ggml_nbytes(den));
    ggml_backend_tensor_set(pt, p, 0, sizeof(p));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> got(n);
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    double err = 0;
    for (int64_t i = 0; i < n; ++i) {
        double want;
        if (mode == GGML_SAMPLE_STEP_EULER) {
            want = hx[i] + (hx[i] - (double) hd[i]) / p[0] * ((double) p[1] - p[0]);
        } else {
            const double r = (double) p[2] / p[0];
            want = (r * hx[i] + (1.0 - r) * hd[i]) * p[4] + p[3] * ref_normal(seed, (uint32_t) p[5], (uint32_t) i);
        }
        err = std::max(err, std::fabs(want - got[i]));
    }
    return err;
}

static int check_analytic(const char * name, ggml_backend_t backend) {
    const double e_euler = analytic_err(backend, GGML_SAMPLE_STEP_EULER);
    const double e_anc   = analytic_err(backend, GGML_SAMPLE_STEP_EULER_ANCESTRAL);
    printf("%s: analytic max abs err euler %.2e euler_a %.2e\n", name, e_euler, e_anc);
    return (e_euler < 1e-6 && e_anc < 1e-6) ? 0 : (printf("  analytic                                         FAIL\n"), 1);
}

static int check(const char * what, bool ok) {
    printf("  %-48s %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

static int check_stats(const char * name, const std::vector<float> & s0, const std::vector<float> & s1,
                       const std::vector<float> & seed1) {
    const moments m = stats(s0);
    printf("%s: mean %.2e var-1 %.2e skew %.2e kurt-3 %.2e chi2(63) %.1f\n", name, m.mean, m.var - 1.0, m.skew,
           m.kurt - 3.0, m.chi2);
    const double c1   = corr(s0, s0, 1);
    const double c128 = corr(s0, s0, 128);
    const double cst  = corr(s0, s1, 0);
    const double csd  = corr(s0, seed1, 0);
    printf("%s: corr lag1 %.2e lag128 %.2e step %.2e seed %.2e\n", name, c1, c128, cst, csd);
    int fail = 0;
    fail += check("finite", m.finite);
    fail += check("|mean| < 1e-3", std::fabs(m.mean) < 1e-3);
    fail += check("|var - 1| < 2e-3", std::fabs(m.var - 1.0) < 2e-3);
    fail += check("|skew| < 5e-3", std::fabs(m.skew) < 5e-3);
    fail += check("|kurt - 3| < 1e-2", std::fabs(m.kurt - 3.0) < 1e-2);
    fail += check("chi2(63) < 103.4 (p > 1e-3)", m.chi2 < 103.4);
    fail += check("|corr| lag 1, lag 128, step, seed < 1e-3",
                  std::fabs(c1) < 1e-3 && std::fabs(c128) < 1e-3 && std::fabs(cst) < 1e-3 && std::fabs(csd) < 1e-3);
    return fail;
}

int main(int argc, char ** argv) {
    bool gpu_only = false;
    int  log2n    = 24;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu-only") == 0) {
            gpu_only = true;
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            log2n = atoi(argv[++i]);
        }
    }
    ggml_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);
    ggml_backend_load_all();

    const int64_t  n    = int64_t(1) << log2n;
    const uint32_t seed = 42u;
    int            fail = 0;

    ggml_backend_t cpu         = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto           set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads");

    std::vector<float> ref;
    if (!gpu_only) {
        noise_graph g0, g1;
        build(g0, cpu, n, seed);
        build(g1, cpu, n, seed + 1);
        if (set_threads) {
            set_threads(cpu, 1);
        }
        const std::vector<float> one_thread = run(cpu, g0, 0);
        if (set_threads) {
            set_threads(cpu, (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
        }
        ref                         = run(cpu, g0, 0);
        const std::vector<float> s1 = run(cpu, g0, 1);
        const std::vector<float> sd = run(cpu, g1, 0);
        fail += check_stats("CPU", ref, s1, sd);
        fail += check_analytic("CPU", cpu);
        fail += check("CPU bit-identical at 1 and 8 threads", memcmp(one_thread.data(), ref.data(), n * sizeof(float)) == 0);
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t gpu  = ggml_backend_dev_init(dev, nullptr);
        const char *   name = ggml_backend_dev_name(dev);
        noise_graph    g0, g1;
        if (!build(g0, gpu, n, seed) || !build(g1, gpu, n, seed + 1)) {
            printf("%s: GGML_OP_SAMPLE_STEP not supported\n", name);
            ggml_backend_free(gpu);
            continue;
        }
        const std::vector<float> a = run(gpu, g0, 0);
        const auto               t0 = std::chrono::high_resolution_clock::now();
        const std::vector<float> b  = run(gpu, g0, 0);
        const auto               t1 = std::chrono::high_resolution_clock::now();
        const std::vector<float> s1 = run(gpu, g0, 1);
        const std::vector<float> sd = run(gpu, g1, 0);
        printf("%s: %lld elements, compute + readback %.3f ms\n", name, (long long) n,
               std::chrono::duration<double, std::milli>(t1 - t0).count());
        fail += check_stats(name, a, s1, sd);
        fail += check_analytic(name, gpu);
        fail += check("bit-identical across two runs", memcmp(a.data(), b.data(), n * sizeof(float)) == 0);
        if (!gpu_only) {
            const double e = max_abs(a, ref);
            printf("%s: max |gpu - cpu| %.3e\n", name, e);
            fail += check("max |gpu - cpu| < 2e-6", e < 2e-6);
        }
        ggml_backend_free(gpu);
    }

    ggml_backend_free(cpu);
    printf("%s\n", fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}
