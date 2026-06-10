#!/bin/bash
# build-portable.sh — host-side orchestrator for the PORTABLE (manylinux_2_28,
# glibc>=2.28) fat-binary libtensorflow build. Clones TF v2.19.0 into a work
# dir, applies the TF-source patch, then runs the in-container build under a
# kernel-enforced memory scope (the careful-math-only approach has wedged this
# machine twice — see workspace CLAUDE.md "Testing Compute-Expensive Resources").
#
# Usage:  bash scripts/build-portable.sh            # full run
#         JOBS=16 bash scripts/build-portable.sh    # override parallelism
# Output: ~/tf-portable-build/out/libtensorflow.tar.gz (raw bazel lib_package)
set -euo pipefail

WORK="${WORK:-$HOME/tf-portable-build}"
# Resolve the repo root from this script's location so patches/ resolves
# regardless of the caller's CWD.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG=quay.io/pypa/manylinux_2_28_x86_64

mkdir -p "$WORK/out" "$WORK/bazel-cache"

# Fresh shallow clone pinned to the release tag. The existing host clone at
# ~/tensorflow-2.19 is NOT reused: it carries .tf_configure.bazelrc residue and
# in-tree edits from the Fedora build; reproducibility wants a clean tree.
if [ ! -d "$WORK/tensorflow" ]; then
	git clone --depth 1 --branch v2.19.0 https://github.com/tensorflow/tensorflow.git "$WORK/tensorflow"
fi

# Patch 01 (gpu_prim cub-const) targets TF source, so it applies pre-build.
# Idempotency guard: the patched line contains "const volatile uint16_t".
if ! grep -q "const volatile uint16_t" "$WORK/tensorflow/tensorflow/core/kernels/gpu_prim.h"; then
	patch -d "$WORK/tensorflow" -p1 < "$REPO/patches/01-gpu_prim-cub-const.patch"
fi

# Memory scope values are the PROVEN no-OOM set from the 2026-06-07 host build.
# TasksMax=400 (not the generic 200): bazel's JVM + workers legitimately exceed
# 200 threads; MemoryMax stays the hard guard.
exec systemd-run --user --scope --collect \
	-p MemoryMax=32G -p MemoryHigh=28G -p MemorySwapMax=0 -p TasksMax=400 \
	podman run --rm \
		-v "$WORK:/work" \
		-v "$REPO:/repo:ro" \
		-e JOBS="${JOBS:-14}" \
		"$IMG" bash /repo/scripts/container-build.sh
