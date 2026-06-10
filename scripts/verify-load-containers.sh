#!/bin/bash
# verify-load-containers.sh — validates the lazy-CUDA artifact contract inside
# (a) Debian 13 (the requested target) and (b) AlmaLinux 8 (the glibc-2.28
# FLOOR — if it loads here, it loads on everything newer). Each distro runs
# TWO modes:
#
#   BARE mode      — no CUDA anywhere: the lib must STILL load, create a
#                    working CPU session, and ANNOUNCE the missing GPU stack
#                    on stderr (the user-directed fallback contract:
#                    "never break functionality, only improve upon it").
#   GPU-LIBS mode  — CUDA runtime staged from the hermetic build cache: the
#                    lib must load with the full GPU dependency set present
#                    (driver stub stands in for libcuda.so.1; real GPU
#                    engagement is verified on the host in PixInsight).
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

# The in-container test program. TF_Version() proves symbol resolution;
# TF_NewSession proves the runtime actually FUNCTIONS — and, decisively for
# the fallback contract, session creation triggers device enumeration, which
# is the moment lazy CUDA loading runs and the GPU-skip announcement is
# emitted on stderr when the stack is absent.
cat > "$STAGE/tf_load_test.py" <<'PYEOF'
import ctypes, sys

lib = ctypes.CDLL("/tf/lib/libtensorflow.so.2")
lib.TF_Version.restype = ctypes.c_char_p
v = lib.TF_Version().decode()
print("TF_Version:", v)
assert v.startswith("2.19"), v

# Minimal C-API session: graph + options -> session, status must be TF_OK (0).
for fn in ("TF_NewGraph", "TF_NewSessionOptions", "TF_NewStatus", "TF_NewSession"):
    getattr(lib, fn).restype = ctypes.c_void_p
graph = lib.TF_NewGraph()
opts = lib.TF_NewSessionOptions()
status = lib.TF_NewStatus()
sess = lib.TF_NewSession(ctypes.c_void_p(graph), ctypes.c_void_p(opts), ctypes.c_void_p(status))
code = lib.TF_GetCode(ctypes.c_void_p(status))
assert code == 0, f"TF_NewSession status code {code}"
assert sess, "TF_NewSession returned NULL"
print("SESSION_OK")

# Session creation alone is lazier than expected — it does NOT probe CUDA
# (verified 2026-06-10: bare-mode stderr stayed empty). Device ENUMERATION is
# what forces the lazy dlopen and emits the missing-CUDA announcement, and
# it's also what PixInsight tools effectively do when they ask for a GPU.
lib.TF_SessionListDevices.restype = ctypes.c_void_p
lib.TF_DeviceListCount.restype = ctypes.c_int
devs = lib.TF_SessionListDevices(ctypes.c_void_p(sess), ctypes.c_void_p(status))
assert lib.TF_GetCode(ctypes.c_void_p(status)) == 0, "TF_SessionListDevices failed"
n = lib.TF_DeviceListCount(ctypes.c_void_p(devs))
print("DEVICE_COUNT:", n)
assert n >= 1, "no devices at all — CPU device missing"
print("VERDICT: LOADS_OK")
PYEOF

# Gather the CUDA runtime set out of the hermetic cache, staged under their
# exact SONAMEs — what the dynamic linker actually resolves. NVIDIA repos ship
# fully-versioned real files (libcufft.so.11.3.x) behind .so.NN symlinks, and
# bazel's _solib dirs add dangling-symlink copies; so: search external/<repo>/lib
# only, match the soname prefix, take the first hit (symlink or file), and
# cp -L (dereference) it INTO the soname filename.
# SONAME list = the GPU runtime set TF dlopens (objdump-derived 2026-06-10).
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
		echo "WARN: $so not found in cache — GPU-libs mode may be incomplete"
	fi
done
# The driver stub satisfies libcuda.so.1 in driverless containers (same trick
# the build itself uses for build-time op generators).
stub="$(find "$CACHE" -path '*/external/*/stubs/libcuda.so' -type f -print -quit 2>/dev/null)"
[ -n "$stub" ] && cp "$stub" "$STAGE/cudalibs/libcuda.so.1"
echo "staged $(ls "$STAGE/cudalibs" | wc -l) CUDA runtime libs (for GPU-libs mode)"

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
		export TF_CPP_MIN_LOG_LEVEL=0

		echo "-- BARE mode (no CUDA: must fall back to CPU, announced) --"
		export LD_LIBRARY_PATH=/tf/lib
		python3 /stage/tf_load_test.py 2> /tmp/bare_stderr.txt
		echo "-- fallback announcement on stderr: --"
		grep -iE "could not load|cuinit|skipping.*gpu|cuda" /tmp/bare_stderr.txt || { echo "(no announcement captured — full stderr follows)"; cat /tmp/bare_stderr.txt; }

		echo "-- GPU-LIBS mode (CUDA staged: must also load) --"
		export LD_LIBRARY_PATH=/stage/cudalibs:/tf/lib
		python3 /stage/tf_load_test.py 2> /tmp/gpu_stderr.txt
		echo "-- device-probe lines: --"
		grep -iE "cuinit|cuda|gpu" /tmp/gpu_stderr.txt || echo "(no CUDA lines)"
	'
done
