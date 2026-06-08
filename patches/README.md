# Patches

Three small source patches are needed to build TensorFlow 2.19's
`libtensorflow` with **CUDA 12.8 + `sm_120`** on a **current toolchain**
(Clang 22 / GCC 16 / Fedora 44). They fall into two groups by *where* the file
lives, which determines *when* you apply them.

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
`(void *)` cast at all three sites.

---

These patches are unified diffs against TensorFlow (Apache-2.0), NVIDIA CUTLASS
(BSD-3-Clause), and BoringSSL — distributed here as diffs only; no upstream
source is included. See [`../LICENSE`](../LICENSE).
