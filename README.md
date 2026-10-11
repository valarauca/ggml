# ggml

<div align="center">

<img src="https://raw.githubusercontent.com/ggml-org/media/refs/heads/master/logo/ggml-logo.jpg" width="256" height="256" alt="ggml logo" />

<b>Tensor library for machine learning</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/ggml?filter=v*)](https://github.com/ggml-org/ggml/releases)
[![CI](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml)

</div>

## About this fork

This is a fork of ggml centered on Intel GPU performance through the SYCL backend. The `feature/intel` branch is tuned and measured on an Intel Arc Pro B70 (Xe2 / Battlemage, 32 Xe cores). It is the ggml used by [valarauca/stable-diffusion.cpp](https://github.com/valarauca/stable-diffusion.cpp), and its SYCL changes are kept in step with [valarauca/llama.cpp](https://github.com/valarauca/llama.cpp) `feature/intel`.

### Changes on `feature/intel`

- **Build**. `GGML_SYCL_XE_FAMILIES` (for example `xe2-hpg`) compiles ahead of time for named Xe families from a family table, which also picks the fp16 GEMM path per device. AOT link options carry over when ggml-sycl is a static library.
- **Flash attention**. A joint_matrix (XMX / DPAS) kernel is the default for prefill at head dim 64, 128 and 256 and for MLA (576 / 512), with f16, q8_0 and q4_0 K/V. At head dim 256 one work-group fills an Xe core with four heads of one KV head, a last partial wave runs as a split-KV launch, and blocks inside the all-zero mask prefix skip the mask. Unmasked attention and short quantized KV go through oneDNN SDPA. `GGML_SYCL_FA_XMX=0` turns the XMX kernel off and `GGML_SYCL_FA_KERNEL=tile|onednn|xmx` forces one.
- **SageAttention**. `GGML_OP_SAGE_ATTN` with int8 QK^T and f16 PV on XMX, a port of the CUDA backend's path, alongside CUDA sage attention support.
- **Diffusion sampling**. `GGML_OP_DENOISE` and `GGML_OP_SAMPLE_STEP` (CPU and SYCL) keep a stable-diffusion sampling loop on the device, and SYCL graphs are replayed instead of re-recorded (including graphs with CONCAT).
- **Correctness**. Zero-point corrections in Q4_1 / Q5_0 / Q5_1 MMVQ matched to the CPU, an exact offset correction in the Q4_0 x q8_1 dot product, f32 sigmoid for half and bf16, the q2_0 quantizer, IM2COL index range, non-contiguous CONV_2D and ROLL, zero-size CONV_3D, the src1 precision hint, and padded KV views in the MKL and oneDNN flash-attention paths.
- **Other backends**. INT8 convrot (CPU, Vulkan, HIP), FP8 casts and a native FP8 matmul through cuBLASLt, and a precomputed RoPE apply op (CPU, CUDA, Vulkan).
- **Tools**. `test-fattn-ab` compares the SYCL flash-attention kernels with the CPU result (`--gpu-only` for profiling, `--kernel` to pick kernels).

On the B70, the head dim 256 XMX kernel runs 24 heads over 4 KV heads with 512 queries at 57 TFLOPS against 33,280 keys (oneDNN SDPA 49), measured with `test-fattn-ab`.

## Quick start

Build from source:

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

For a minimal, fully commented example (matrix multiplication), see [examples/simple](examples/simple).

## Description

The main goal of `ggml` is to be a simple, portable, and efficient tensor library for machine learning with minimal setup.

- Plain C/C++ implementation without any dependencies
- Cross-platform - x86, ARM, RISC-V, LoongArch, PowerPC, s390x, and WebAssembly
- SIMD-optimized kernels for x86, ARM, and RISC-V
- Broad backend support - CPU, GPU, NPU, and browser
- 2- to 8-bit integer quantization, plus MXFP4 and NVFP4 microscaling formats
- Zero memory allocations during runtime

## Documentation

- [The GGUF file format](docs/gguf.md)
- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Contributing

- For changes to the core `ggml` library (including to the CMake build system), please open a PR in [llama.cpp](https://github.com/ggml-org/llama.cpp) - doing so will make your PR more visible, better tested, and more likely to be reviewed
