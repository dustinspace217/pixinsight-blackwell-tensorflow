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

# Flag provenance:
#  --config=cuda_nvcc        verified .bazelrc:265 — implies --config=cuda,
#                            nvcc for device, clang for host (TF_NVCC_CLANG=1)
#  --repo_env on the CLI     proven gotcha: rc-layered defaults (CUDA 12.5.1 /
#                            cuDNN 9.3.0) silently win otherwise
#  arch list                 spec decision: fat binary; syntax verified
#                            cuda_configure.bzl:169 (sm_/compute_ prefixes)
#  cstdint force-include +   proven fixes #2/#3 from the Fedora build; harmless
#  warning suppressions      if this clang doesn't need them
#  -Wno-error                survive new-compiler warnings (proven)
bazel --output_user_root=/work/bazel-cache build -c opt \
	--config=cuda_nvcc \
	--copt=-Wno-error --keep_going --jobs="$JOBS" \
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
