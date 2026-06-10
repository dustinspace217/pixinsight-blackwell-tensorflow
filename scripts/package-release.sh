#!/bin/bash
# package-release.sh — turns the raw bazel lib_package tarball into the named
# release artifact: recompress as xz (smaller; gz→xz typically 25-35% off),
# verify LICENSE presence (Apache-2.0 requires it), embed build provenance
# including the load-time dependency contract (this build hard-links its CUDA
# deps — no lazy-dlopen CPU fallback like Google's 2.18 build).
#   usage: bash scripts/package-release.sh <libtensorflow.tar.gz> <out-dir>
set -euo pipefail
TAR="${1:?usage: package-release.sh <tarball> <outdir>}"
OUT="${2:?usage: package-release.sh <tarball> <outdir>}"
NAME="libtensorflow-2.19.0-gpu-cuda12.8-sm80_120-linux-x86_64"
D="$(mktemp -d -p "${TMPDIR:-/tmp}" tfpkg.XXXX)"
trap 'rm -rf "$D"' EXIT

mkdir -p "$D/$NAME"
tar xzf "$TAR" -C "$D/$NAME"
ls "$D/$NAME"/LICENSE* >/dev/null || { echo "FAIL: no LICENSE in lib_package output"; exit 1; }
{
	echo "Built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "Source: tensorflow v2.19.0 (github.com/tensorflow/tensorflow) + patches/ in"
	echo "  github.com/dustinspace217/pixinsight-blackwell-tensorflow"
	echo "Build env: quay.io/pypa/manylinux_2_28_x86_64 container (glibc 2.28 floor),"
	echo "  --config=cuda_clang with AlmaLinux 8 clang-21, lld"
	echo "Hermetic: CUDA 12.8.0, cuDNN 9.7.0, python 3.12"
	echo "GPU arches: SASS sm_80,sm_86,sm_89,sm_90,sm_120 + cc-12.0 PTX (forward-JIT)"
	echo ""
	echo "Load-time requirements (DT_NEEDED — NO CPU fallback if missing):"
	echo "  NVIDIA driver (libcuda.so.1), CUDA 12.x runtime (cudart, cublas/Lt,"
	echo "  cufft, cusolver, cusparse, nvrtc + builtins 12.8, nvJitLink, cupti),"
	echo "  cuDNN 9 family, NCCL 2 (libnccl.so.2 — NOT part of the CUDA toolkit)."
	echo "  All must be on the loader path (see README: PixInsight.sh edit)."
	echo ""
	echo "Verified: glibc symbol audit <=2.28; loads on Debian 13 (trixie) and"
	echo "  AlmaLinux 8 containers; GPU end-to-end in PixInsight on RTX 5080/Fedora 44."
} > "$D/$NAME/PROVENANCE.txt"

mkdir -p "$OUT"
tar -C "$D" -cJf "$OUT/$NAME.tar.xz" "$NAME"   # xz; XZ_OPT="-9 -T0" via env if size-critical
( cd "$OUT" && sha256sum "$NAME.tar.xz" > "$NAME.tar.xz.sha256" )
SIZE=$(stat -c%s "$OUT/$NAME.tar.xz")
echo "artifact: $OUT/$NAME.tar.xz ($((SIZE/1024/1024)) MB)"
if [ "$SIZE" -ge 2000000000 ]; then
	echo "WARN: >= ~2GB GitHub per-file cap — split: split -b 1900m $NAME.tar.xz $NAME.tar.xz.part"
fi
