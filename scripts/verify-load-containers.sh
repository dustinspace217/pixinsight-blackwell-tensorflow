#!/bin/bash
# verify-load-containers.sh — loads the built libtensorflow inside (a) Debian 13
# (the requested target) and (b) AlmaLinux 8 (the glibc-2.28 FLOOR — if it loads
# here, it loads on everything newer). ctypes + TF_Version() proves symbol
# resolution and basic C-API function, no GPU needed (kernels are distro-blind).
#
# The GPU build hard-links its CUDA dependencies (DT_NEEDED: cudart, cublas,
# cufft, cusolver, cusparse, nvrtc, nvJitLink, cupti, the cuDNN 9 family, nccl,
# and the libcuda.so.1 driver) — there is NO lazy-dlopen CPU fallback like
# Google's old 2.18 build. So the load test stages those runtime libs from the
# hermetic build cache; that's a feature: it validates the lib against exactly
# the dependency set the README documents for users.
#
# SELinux note: everything is COPIED into a throwaway :Z-labeled stage dir —
# never bind-mount + relabel the original artifact's directory.
#   usage: bash scripts/verify-load-containers.sh <libtensorflow.tar.gz> [bazel-cache-dir]
set -euo pipefail
TAR="${1:?usage: verify-load-containers.sh <tarball> [bazel-cache-dir]}"
CACHE="${2:-$HOME/tf-portable-build/bazel-cache}"

STAGE="$(mktemp -d -p "${TMPDIR:-/tmp}" tfload.XXXX)"
trap 'rm -rf "$STAGE"' EXIT
cp "$TAR" "$STAGE/tf.tar.gz"

# Gather the CUDA runtime set out of the hermetic cache, staged under their
# exact SONAMEs — what the dynamic linker actually resolves. NVIDIA repos ship
# fully-versioned real files (libcufft.so.11.3.x) behind .so.NN symlinks, and
# bazel's _solib dirs add dangling-symlink copies; so: search external/<repo>/lib
# only, match the soname prefix, take the first hit (symlink or file), and
# cp -L (dereference) it INTO the soname filename.
# SONAME list = the artifact's actual DT_NEEDED CUDA set (objdump-verified).
mkdir -p "$STAGE/cudalibs"
for so in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcufft.so.11 \
          libcusolver.so.11 libcusparse.so.12 libnvrtc.so.12 \
          libnvrtc-builtins.so.12.8 libnvJitLink.so.12 libcupti.so.12 \
          libnccl.so.2 libcudnn.so.9 libcudnn_ops.so.9 libcudnn_cnn.so.9 \
          libcudnn_adv.so.9 libcudnn_graph.so.9 libcudnn_engines_precompiled.so.9 \
          libcudnn_engines_runtime_compiled.so.9 libcudnn_heuristic.so.9; do
	# -print -quit = first match without a pipe (`find | head` SIGPIPEs under
	# pipefail). No -type filter: the .so.NN entry is often a symlink; cp -L
	# dereferences it to the real file at the staged soname path.
	hit="$(find "$CACHE" -path '*/external/*/lib/*' -name "$so*" -not -path '*stubs*' -print -quit 2>/dev/null)"
	if [ -n "$hit" ]; then
		cp -L "$hit" "$STAGE/cudalibs/$so"
	else
		echo "WARN: $so not found in cache — load test may fail on it"
	fi
done
# The driver stub satisfies libcuda.so.1 in driverless containers (same trick
# the build itself uses for build-time op generators).
stub="$(find "$CACHE" -path '*/external/*/stubs/libcuda.so' -type f -print -quit 2>/dev/null)"
[ -n "$stub" ] && cp "$stub" "$STAGE/cudalibs/libcuda.so.1"
echo "staged $(ls "$STAGE/cudalibs" | wc -l) CUDA runtime libs"

for IMG in docker.io/library/debian:trixie docker.io/library/almalinux:8; do
	echo "== $IMG =="
	podman run --rm -v "$STAGE":/stage:Z "$IMG" bash -c '
		set -e
		if command -v apt-get >/dev/null; then
			export DEBIAN_FRONTEND=noninteractive
			apt-get update -qq >/dev/null && apt-get install -y -qq python3 >/dev/null 2>&1
		else
			dnf install -y -q python3 >/dev/null 2>&1
		fi
		mkdir -p /tf && tar xzf /stage/tf.tar.gz -C /tf
		export LD_LIBRARY_PATH=/stage/cudalibs:/tf/lib
		python3 -c "
import ctypes
lib = ctypes.CDLL(\"/tf/lib/libtensorflow.so.2\")
lib.TF_Version.restype = ctypes.c_char_p
v = lib.TF_Version().decode()
print(\"TF_Version:\", v)
assert v.startswith(\"2.19\"), v
print(\"VERDICT: LOADS_OK\")
"'
done
