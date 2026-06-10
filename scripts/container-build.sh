#!/bin/bash
# container-build.sh — runs INSIDE quay.io/pypa/manylinux_2_28_x86_64 (AlmaLinux
# 8, glibc 2.28). Builds the fat-binary libtensorflow C library with TF 2.19's
# cuda_nvcc config: hermetic nvcc 12.8 generates ALL device SASS (sm_80..sm_120)
# so the container's clang only compiles HOST code and does not need Blackwell
# support. This is why approach B works without a bleeding-edge LLVM.
set -euo pipefail

JOBS="${JOBS:-14}"

# Host toolchain: AlmaLinux 8 clang (whatever AppStream ships — version printed
# below for the log) + lld (REQUIRED: TF link opts force -fuse-ld=lld).
dnf install -y clang lld git >/dev/null
clang --version | head -1

# Bazelisk fetches the bazel version TF pins in .bazelversion (7.x) itself.
curl -fL -o /usr/local/bin/bazel \
	https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel

# manylinux convention: CPython interpreters under /opt/python. Hermetic python
# 3.12 is fetched by bazel itself; this is only the bootstrap interpreter.
export PATH=/opt/python/cp312-cp312/bin:$PATH
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export HOME=/work   # keeps every cache bazel makes on the bind mount

cd /work/tensorflow

# CUDA driver stub for build-time tool execution: TF's op-wrapper generators
# (training_ops_gen_cc etc.) run DURING the build and link libcuda.so.1 — the
# DRIVER library, which a GPU-less container doesn't have. NVIDIA ships a stub
# in the toolkit for exactly this no-driver case; the hermetic cuda_cudart
# package carries it. Symlink it as libcuda.so.1 inside the container (this is
# the container's own /usr/lib64, discarded with it — never the host's).
# On a FRESH cache the stub appears only after bazel's fetch phase — so this
# also re-checks after a failed first attempt: re-running the script finds it.
STUB="$(ls /work/bazel-cache/*/external/cuda_cudart/lib/stubs/libcuda.so 2>/dev/null | head -1 || true)"
if [ -n "$STUB" ]; then
	ln -sf "$STUB" /usr/lib64/libcuda.so.1
	ldconfig
	echo "driver stub linked: $STUB"
else
	echo "driver stub not in cache yet (fresh build) — genrules may fail once; re-run after fetch"
fi

# Flag provenance:
#  --config=cuda_clang       SWITCHED from cuda_nvcc 2026-06-10 after TWO
#                            nvcc-only error classes (GpuLaneId clang-builtin
#                            guard; Eigen-half alignas(4)-vs-(2) strictness in
#                            split/concat kernels). Alma 8's clang-21 knows
#                            sm_120, so clang can compile device code directly —
#                            this replicates the PROVEN Fedora host config
#                            (cuda_clang + clang-22) one compiler version off.
#  CLANG_CUDA_COMPILER_PATH  how cuda_clang finds the device compiler
#                            (verified .bazelrc:260, cuda_clang_official)
#  --repo_env on the CLI     proven gotcha: rc-layered defaults (CUDA 12.5.1 /
#                            cuDNN 9.3.0) silently win otherwise
#  arch list                 spec decision: fat binary; syntax verified
#                            cuda_configure.bzl:169 (sm_/compute_ prefixes)
#  cstdint force-include +   proven fixes #2/#3 from the Fedora build; harmless
#  warning suppressions      if this clang doesn't need them
#  -Wno-error                survive new-compiler warnings (proven)
#  -Qunused-arguments        crosstool passes --cuda-path to host clang even for
#                            plain C; clang-21 hard-errors on the unused arg.
#                            Same fix cuda_clang itself uses (.bazelrc:239).
# include_cuda_libs=false (overrides --config=cuda's =true; later flag wins):
# do NOT hard-link the CUDA libraries. TF then loads them lazily by soname at
# first GPU use (dso_loader), so the SAME artifact runs everywhere: full GPU
# when driver+CUDA+cuDNN are present, announced CPU fallback when not — the
# user's directive 2026-06-10: "never break functionality, only improve it."
# This matches Google's own distribution configuration (.bazelrc cuda_wheel).
bazel --output_user_root=/work/bazel-cache build -c opt \
	--config=cuda_clang \
	--@local_config_cuda//cuda:include_cuda_libs=false \
	--action_env=CLANG_CUDA_COMPILER_PATH=/usr/bin/clang \
	--copt=-Wno-error --keep_going --jobs="$JOBS" \
	--copt=-Qunused-arguments --host_copt=-Qunused-arguments \
	--repo_env=HERMETIC_PYTHON_VERSION=3.12 \
	--repo_env=HERMETIC_CUDA_VERSION=12.8.0 \
	--repo_env=HERMETIC_CUDNN_VERSION=9.7.0 \
	--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_80,sm_86,sm_89,sm_90,sm_120,compute_120" \
	--cxxopt=-include --cxxopt=cstdint --host_cxxopt=-include --host_cxxopt=cstdint \
	--copt=-Wno-c23-extensions --copt=-Wno-gnu-offsetof-extensions --copt=-Wno-macro-redefined \
	--host_copt=-Wno-c23-extensions --host_copt=-Wno-gnu-offsetof-extensions --host_copt=-Wno-macro-redefined \
	//tensorflow/tools/lib_package:libtensorflow

cp bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz /work/out/
echo "BUILD_OK"
