# Patches

Four small source patches cover building TensorFlow 2.19's `libtensorflow`
with **CUDA 12.8 + `sm_120`** across the toolchains this project has used.
Not every build needs every patch — the table says which:

| Patch | Needed by | Notes |
|---|---|---|
| `01-gpu_prim-cub-const` | **all builds** | template mismatch, compiler-agnostic |
| `02-cutlass-set_slice_3x3` | **all observed builds** | upstream typo; lives in bazel cache |
| `03-boringssl-memchr-const` | Clang 22 host (Fedora) | NOT triggered by Alma 8's clang-21 container build — kept for newer host compilers |
| `04-gpu-device-functions-nvcc-laneid` | `--config=cuda_nvcc` only | harmless under `cuda_clang` (guard still picks the clang branch); applied unconditionally by `build-portable.sh` |

They fall into two groups by *where* the file lives, which determines *when*
you apply them. The container recipe (`scripts/build-portable.sh`) applies
01 and 04 automatically; 02 is applied to the bazel cache after the first
build attempt populates it.

> The other build fixes from this recipe are **not** source patches and live in
> the main [`README.md`](../README.md) build command instead: force-including
> `<cstdint>` (`--cxxopt=-include cstdint`), silencing Clang 22 warnings
> (`-Wno-c23-extensions`, `-Wno-gnu-offsetof-extensions`, `-Wno-macro-redefined`),
> and installing the `lld` linker (`sudo dnf install lld`).

---

## 1. `01-gpu_prim-cub-const.patch` — TensorFlow source tree

**What:** CUDA 12.8's `cub` changed `ThreadLoadVolatilePointer` to take a
`const T*`. TF 2.19's `Eigen::half` / `Eigen::bfloat16` specializations in
`tensorflow/core/kernels/gpu_prim.h` still use a non-`const` `T*`, so they no
longer match the template (38 compile errors across the GPU kernels). This
const-qualifies the two load specializations and their `reinterpret_cast`.

**Why it's compiler-agnostic:** a specialization that doesn't match its template
is rejected by *any* compiler — switching Clang versions does not help.

**Apply BEFORE the build** (it's in the normal source tree):
```bash
cd ~/tensorflow-2.19
patch -p0 < /path/to/patches/01-gpu_prim-cub-const.patch
```

---

## 2 & 3 — Bazel-fetched externals (apply AFTER the first build attempt)

`cutlass` and `boringssl` are downloaded by Bazel into its cache, under a path
like `~/.cache/bazel/_bazel_<user>/<hash>/external/...` (the `<hash>` is
machine/workspace-specific — that's why the patch headers here show a clean
`external/...` path rather than an absolute one). So:

1. Start the build once so Bazel populates `external/`.
2. Apply patches 2 and 3 to the fetched files (find them with
   `find ~/.cache/bazel -path '*/external/cutlass_archive/include/cutlass/matrix.h'`
   and the boringssl equivalent).
3. **Re-run the build — do NOT `bazel clean`.** Cleaning re-extracts the
   archives and reverts these edits.

### `02-cutlass-set_slice_3x3.patch`
**What:** CUTLASS `matrix.h` *calls* `set_slice3x3(...)` at four sites, but the
method is *defined* as `set_slice_3x3` (a missing underscore — an upstream
typo). Fixes the four calls.

### `03-boringssl-memchr-const.patch`
**What:** Newer glibc makes `memchr` return `const`-generic, so three helper
functions returning `void *` hit
`-Werror,-Wincompatible-pointer-types-discards-qualifiers`. Adds an explicit
`(void *)` cast at all three sites. (Surfaced by Clang 22 on Fedora 44; the
AlmaLinux 8 / clang-21 container build does not trip it.)

---

## 4. `04-gpu-device-functions-nvcc-laneid.patch` — TensorFlow source tree

**What:** `GpuLaneId()` in `tensorflow/core/util/gpu_device_functions.h` guards
with `#if __clang__` to select clang's `__nvvm_read_ptx_sreg_laneid()` builtin.
Under `--config=cuda_nvcc` (nvcc device compiler + clang *host* compiler),
`__clang__` is defined during preprocessing but nvcc's device compiler (cicc)
doesn't have clang builtins → `identifier "__nvvm_read_ptx_sreg_laneid" is
undefined` in every kernel that includes the header. The guard becomes
`#if defined(__clang__) && !defined(__NVCC__)`, routing nvcc to the inline-PTX
branch it expects.

**When:** only the nvcc+clang-host path needs it; it's a no-op for pure
`cuda_clang` builds (clang still takes the builtin branch), so the container
recipe applies it unconditionally. Apply BEFORE the build, like patch 01:
```bash
patch -p0 -d ~/tensorflow-2.19 < /path/to/patches/04-gpu-device-functions-nvcc-laneid.patch
```

---

These patches are unified diffs against TensorFlow (Apache-2.0), NVIDIA CUTLASS
(BSD-3-Clause), and BoringSSL — distributed here as diffs only; no upstream
source is included. See [`../LICENSE`](../LICENSE).
