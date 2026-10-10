#pragma once

// Stateless counter-based noise for GGML_OP_SAMPLE_STEP. Plain C so CPU and device code share the
// integer part bit for bit; each backend turns the two uniforms into a normal with its own
// log/cos/sqrt (Box-Muller, first output only).

#include <stdint.h>

static inline uint32_t ggml_wang_hash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s ^= s >> 4;
    s *= 0x27d4eb2du;
    s ^= s >> 15;
    return s;
}

static inline uint32_t ggml_wang_key(uint32_t seed, uint32_t step) {
    return ggml_wang_hash(seed ^ ggml_wang_hash(step + 0x9e3779b9u));
}

// u1 in (0, 1) so log(u1) is finite, u2 in [0, 1). Requires i < 2^31.
static inline void ggml_wang_uniform2(uint32_t key, uint32_t i, float * u1, float * u2) {
    const uint32_t h1 = ggml_wang_hash(key ^ (2u * i));
    const uint32_t h2 = ggml_wang_hash(h1 ^ (2u * i + 1u));
    *u1 = ((float) (h1 >> 8) + 0.5f) * (1.0f / 16777216.0f);
    *u2 = (float) (h2 >> 8) * (1.0f / 16777216.0f);
}
