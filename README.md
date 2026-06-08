# Blackwell-native TensorFlow for PixInsight on Linux (RTX 50-series, sm_120)

Build a GPU-accelerated `libtensorflow` C library that works with PixInsight's
RC Astro tools (**StarXTerminator, NoiseXTerminator, BlurXTerminator**) and
StarNet on **NVIDIA Blackwell GPUs** (RTX 5070/5080/5090, compute capability
`sm_120`) under Linux.

**Status:** confirmed working — RTX 5080 + Fedora 44, all three RC Astro tools
GPU-accelerated, no `CUDA_ERROR_INVALID_PTX`.

---

## Why this exists

Every existing Linux path for PixInsight GPU acceleration breaks on Blackwell:

- **Google's prebuilt `libtensorflow`** stops at **2.18** and is built against
  CUDA 12.5 — it has **no `sm_120` kernels**. On a 50-series card it tries to
  JIT-compile from PTX and fails with `CUDA_ERROR_INVALID_PTX`. (TF issues
  [#89272](https://github.com/tensorflow/tensorflow/issues/89272),
  [#103531](https://github.com/tensorflow/tensorflow/issues/103531).)
- Community Blackwell builds exist only as **Python wheels** — PixInsight needs
  the **C API library** (`libtensorflow.so`), a different artifact.
- The popular Kubuntu guides install that prebuilt 2.15/2.18 — which JIT-fails
  on Blackwell.

So the only working option is to **build `libtensorflow` from source with CUDA
12.8 and native `sm_120` kernels**. This repo is the reproducible recipe:
the exact versions, the source patches a current toolchain needs, and the
PixInsight install steps.

> This repo contains **only** documentation, our scripts, and small **patches
> (unified diffs)** against the upstream projects. It bundles no TensorFlow
> source, no NVIDIA libraries, and no compiled binaries — you build those
> yourself from the official sources. See [Licensing](#licensing).

---

## Tested environment

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 5080 (Blackwell, `sm_120`) |
| OS | Fedora 44 (KDE), kernel 7.0.x |
| NVIDIA driver | 595.80 (RPM Fusion `akmod-nvidia`) |
| Host compiler | Clang 22 (Fedora default) |
| TensorFlow | **v2.19.0** (source) |
| CUDA (hermetic + runtime) | **12.8** |
| cuDNN | **9.7.0** |
| Bazel | 7.7.0 (via bazelisk) |
| Build Python | 3.12 (hermetic) |
| PixInsight | 1.9.3 |

### Why TensorFlow 2.19 specifically
- **2.20 / 2.21** switched to "pywrap" build rules and **removed the C-library
  target** (`//tensorflow/tools/lib_package:libtensorflow`) — you can't easily
  build `libtensorflow.so` from them.
- **2.18 and older** predate CUDA 12.8 in their hermetic set (top out at 12.6),
  so they can't fetch the CUDA that `sm_120` requires.
- **2.19 is the only version with *both*** the clean C-library target **and**
  hermetic CUDA 12.8 support.

---

## Prerequisites

1. Working NVIDIA driver (RPM Fusion recommended on Fedora; must support
   Blackwell — 570+). Do **not** install NVIDIA's `cuda` metapackage over an
   RPM Fusion driver; if you add NVIDIA's CUDA repo, exclude its driver packages
   (see RPM Fusion's CUDA howto).
2. **CUDA 12.8 toolkit** (runtime libs) + **cuDNN 9 for CUDA 12** installed
   (e.g. CUDA 12.8 runfile `--toolkit` → `/usr/local/cuda-12.8`, then drop the
   `cudnn-*_cuda12` libs into its `lib64`). Needed at *runtime* by the library.
3. **lld** linker: `sudo dnf install lld` (TF links with `-fuse-ld=lld`).
4. `git`, and **bazelisk** (it auto-fetches the Bazel version TF pins):
   ```
   mkdir -p ~/bin && curl -fsSL -o ~/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && chmod +x ~/bin/bazel
   ```

---

## Build

```bash
git clone --depth 1 --branch v2.19.0 https://github.com/tensorflow/tensorflow.git ~/tensorflow-2.19
cd ~/tensorflow-2.19

# Configure for hermetic CUDA 12.8 + cuDNN 9.7 + native sm_120, host compiler Clang.
export TF_NEED_CUDA=1 TF_NEED_CLANG=1 TF_NEED_ROCM=0 \
       CLANG_COMPILER_PATH=/usr/bin/clang CC=/usr/bin/clang \
       HERMETIC_CUDA_VERSION=12.8.0 HERMETIC_CUDNN_VERSION=9.7.0 \
       HERMETIC_PYTHON_VERSION=3.12 HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_120 \
       PYTHON_BIN_PATH=/usr/bin/python3
yes "" | ./configure
```

### Apply the patches (see [`patches/`](patches/))
A current toolchain (Clang 22 / GCC 16 / glibc on Fedora 44) is newer than TF
2.19 expects, and CUDA 12.8 ships newer `cub`. Apply:

```bash
# 1. TF gpu_prim.h: CUDA 12.8 cub made ThreadLoadVolatilePointer take `const T*`;
#    const-correct the half/bfloat16 load specializations. (source tree)
patch -p0 -d ~/tensorflow-2.19 < patches/01-gpu_prim-cub-const.patch
```
The **cutlass** (`set_slice3x3` typo) and **boringssl** (`const`-generic
`memchr`) patches apply to Bazel-fetched externals, so they're applied **after
the first build attempt populates the cache**, then you rebuild **without
`bazel clean`** (cleaning re-extracts and reverts them). See
[`patches/README.md`](patches/) for the exact in-cache paths, or just run the
build once, apply `02-` and `03-` to the printed `external/...` paths, and rerun.

### Build the C library (memory-bounded so it can't wedge the machine)
```bash
systemd-run --user --scope --collect -p MemoryMax=32G -p MemoryHigh=28G \
  -p MemorySwapMax=0 -p TasksMax=400 \
  ~/bin/bazel build -c opt \
    --copt=-Wno-error --copt=-Wno-c23-extensions \
    --copt=-Wno-gnu-offsetof-extensions --copt=-Wno-macro-redefined \
    --host_copt=-Wno-c23-extensions --host_copt=-Wno-gnu-offsetof-extensions \
    --host_copt=-Wno-macro-redefined \
    --cxxopt=-include --cxxopt=cstdint --host_cxxopt=-include --host_cxxopt=cstdint \
    --jobs=16 \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    --repo_env=HERMETIC_CUDA_VERSION=12.8.0 \
    --repo_env=HERMETIC_CUDNN_VERSION=9.7.0 \
    --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_120 \
    //tensorflow/tools/lib_package:libtensorflow
```
Output: `bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz`.

> Pin the hermetic versions and compute capability **on the command line**
> (`--repo_env=...`). TF's rc files set CUDA 12.5.1 / cuDNN 9.3.0 and a default
> compute-capability list; command-line `--repo_env` is applied last and wins
> unambiguously.

### Verify it's actually Blackwell
```bash
/usr/local/cuda-12.8/bin/cuobjdump --list-elf libtensorflow.so.2.19.0 | grep -oE 'sm_[0-9]+' | sort -u
# expect: sm_120
```

---

## Install into PixInsight

1. **Replace PixInsight's TensorFlow** with the build (back up first):
   ```bash
   tar -xzf libtensorflow.tar.gz -C /tmp/libtf && \
   sudo mv /usr/local/libtensorflow /usr/local/libtensorflow.bak ; \
   sudo cp -aP /tmp/libtf /usr/local/libtensorflow
   ```
   (Also remove any `libtensorflow*` bundled in `/opt/PixInsight/bin/lib` so the
   external build is used.)
2. **Provide the two runtime libs the build links** that the CUDA runfile
   toolkit doesn't put on the path — `libnccl.so.2` (from the build's hermetic
   cache, or `dnf`) and `libcupti.so.12` (from `cuda-12.8/extras/CUPTI/lib64`) —
   into `/usr/local/cuda-12.8/lib64`.
3. **Point PixInsight's launcher at the libraries.** PixInsight's launcher
   (`/opt/PixInsight/bin/PixInsight.sh`) hard-overwrites `LD_LIBRARY_PATH`;
   prepend the CUDA + TensorFlow dirs to its line 7. The idempotent
   [`scripts/pixinsight-gpu-fix.sh`](scripts/pixinsight-gpu-fix.sh) does this
   (and re-removes the bundled TF) — **re-run it after every PixInsight update**,
   which reverts both.
4. **Re-register the modules** if PixInsight dropped them after earlier failed
   loads: *Process → Modules → Install Modules*, search `/opt/PixInsight/bin`
   (Recursive), install StarX/NoiseX/BlurX, restart.

### Verify the GPU is used
Launch PixInsight, run NoiseXTerminator on an image while watching
`watch -n1 nvidia-smi`. Success = PixInsight holds VRAM, GPU-Util climbs, and
there is **no `INVALID_PTX`** and **no multi-minute JIT freeze** (native kernels
need no JIT).

`scripts/pixinsight-cpu-mode.sh` is a break-glass fallback that forces the tools
onto the CPU if the GPU library ever breaks.

---

## Licensing

This repository is documentation, original scripts, and small patches:

- **Patches** (`patches/`) are unified diffs against TensorFlow (Apache-2.0),
  NVIDIA CUTLASS (BSD-3-Clause), and BoringSSL (Apache-2.0/OpenSSL/ISC).
  Distributing diffs against these is permitted by their licenses.
- **Scripts** and this guide are released under the MIT License (see `LICENSE`).
- This repo bundles **no** upstream source, **no** NVIDIA libraries, and **no**
  compiled binaries. You obtain TensorFlow from Google and CUDA/cuDNN/NCCL from
  NVIDIA under their respective licenses.

A compiled `libtensorflow.so` you build is itself redistributable under
Apache-2.0 (with TF's `LICENSE`/`NOTICE`) — but it is intentionally **not**
distributed here, since it is pinned to one exact CUDA/glibc/`sm_120` toolchain
and carries attribution obligations the recipe does not.

## Credits / prior art
Built on the trail blazed by the PixInsight + Cloudy Nights community —
**lblock** (the Kubuntu GPU-acceleration guides), **Leonard** (first to build
`libtensorflow` from source for a 5090), and **maludwig** (the TF 2.x Blackwell
wheel build guide on TF #89272). This repo's contribution is a reproducible,
**C-library**, **Fedora**, from-source recipe with the toolchain patches a
current distro needs.
