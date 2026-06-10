# Portable GPU Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a glibc-2.28-floor fat-binary `libtensorflow` (sm_80→sm_120) and a portable, pip-free DeepSNR/ONNX delivery, verified on Debian 13, released on GitHub for broad sharing.

**Architecture:** TF is built inside a `manylinux_2_28` Podman container using TF 2.19's `--config=cuda_nvcc` (nvcc compiles device code → container clang version doesn't need Blackwell awareness), inheriting the proven 2026-06-07 host-build config. DeepSNR work is delivery-only: the official ORT 1.26.0 wheel is already manylinux_2_28; we replace pip with a stdlib PyPI fetch and mirror the libs as a release. Both artifacts gate on container load tests (debian:trixie + almalinux:8) before draft releases; QA review runs before anything publishes.

**Tech Stack:** Podman, Bazel(isk), TF 2.19 hermetic CUDA 12.8.0/cuDNN 9.7.0, python3 stdlib, gh CLI.

**Spec:** `docs/superpowers/specs/2026-06-10-portable-artifacts-design.md` (decisions there are settled — do not re-ask).

## Status (updated 2026-06-10 ~05:45)
Phase: A+B COMPLETE; C gates 1-2 PASSED; artifact PACKAGED (190 MB tar.xz)
Done: fat build SUCCEEDED under cuda_clang (BUILD_OK, 23,700 actions; boringssl
  patch never needed under clang-21); gate 1 glibc=2.28 + 5 SASS arches +
  cc-12.0 PTX; gate 2 LOADS_OK on trixie AND alma8 (20 CUDA libs staged by
  soname); ORT side fully done (see ~03:00 entry)
Next: Task 12 (DUSTIN: backup lib → install fat lib → PixInsight StarX/NXT run)
  → Task 15 READMEs → Task 14 draft releases → Phase E QA workflow → publish
Blocked: Task 12 needs Dustin (sudo + GUI)

## Verified facts the plan builds on (do not re-derive)
- TF 2.19 `.bazelrc:265` — `build:cuda_nvcc --config=cuda` + `TF_NVCC_CLANG=1` + `cuda_compiler=nvcc`. Host clang need NOT know sm_120; hermetic nvcc 12.8 does device SASS.
- `cuda_configure.bzl:169` — arch list entries must start `sm_` or `compute_`, comma-separated. Official builds use the same shape (`.bazelrc:248`).
- `.bazelrc:252-255` — lld required (`-fuse-ld=lld` in link opts).
- Proven host values (PLAN.md status, 2.19 build): hermetic CUDA **12.8.0**, cuDNN **9.7.0**, python 3.12, target `//tensorflow/tools/lib_package:libtensorflow`, repo_env on CLI (rc gotcha), no `bazel clean` between fixes, `MemoryMax=32G/TasksMax=400/jobs≤16` no-OOM.
- manylinux_2_28 image pythons live under `/opt/python/cp3XX-*/bin`.
- ORT wheel `onnxruntime_gpu-1.26.0-*-manylinux_2_27…2_28_x86_64.whl`; the three capi `.so`s do not link libpython (python-tag-independent).

---

## Phase A — TF container build scaffolding + kickoff

### Task 1: Pre-flight checks

**Files:** none (read-only)

- [ ] **Step 1: Disk + tools**

Run (separate simple calls — H.replace rule): `df -h /home | tail -1`, `podman --version`, `ls /usr/local/cuda-12.8/bin/cuobjdump`
Expected: ≥150G free; podman ≥4.x; cuobjdump exists.

- [ ] **Step 2: Pull images**

Run: `podman pull quay.io/pypa/manylinux_2_28_x86_64` then `podman pull docker.io/library/debian:trixie` then `podman pull docker.io/library/almalinux:8`
Expected: three image digests print.

### Task 2: Host orchestrator script

**Files:**
- Create: `~/Claude/pixinsight-blackwell-tf/scripts/build-portable.sh`

- [ ] **Step 1: Write the script**

```bash
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
```

- [ ] **Step 2: Syntax check**

Run: `bash -n ~/Claude/pixinsight-blackwell-tf/scripts/build-portable.sh`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

`git -C ~/Claude/pixinsight-blackwell-tf add scripts/build-portable.sh` then commit `feat: host orchestrator for portable fat-binary build`.

### Task 3: In-container build script

**Files:**
- Create: `~/Claude/pixinsight-blackwell-tf/scripts/container-build.sh`

- [ ] **Step 1: Write the script**

```bash
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
```

- [ ] **Step 2: Syntax check** — `bash -n .../scripts/container-build.sh`, expect exit 0.

- [ ] **Step 3: Commit** — `feat: in-container manylinux_2_28 build script (cuda_nvcc, fat arch list)`.

### Task 4: Kickoff + iteration loop (LONG — run in background, do Phase B meanwhile)

**Files:** none new (build.log under `~/tf-portable-build/`)

- [ ] **Step 1: Launch**

Run (background): `bash ~/Claude/pixinsight-blackwell-tf/scripts/build-portable.sh > ~/tf-portable-build/build.log 2>&1`
Expected: hours. Check via `tail -5 ~/tf-portable-build/build.log` (simple call shape).

- [ ] **Step 2: On failure — classify against the knowledge base, fix, re-run (NO `bazel clean`)**

| Error signature (in build.log) | Fix (exact) |
|---|---|
| `set_slice3x3` undeclared / no member | `sed -i 's/set_slice3x3/set_slice_3x3/g' /home/dustin/tf-portable-build/bazel-cache/*/external/cutlass_archive/include/cutlass/matrix.h` (host-side path of the bind mount; keep `.orig` first via `cp`) |
| `ThreadLoadVolatilePointer` const errors in `gpu_prim.h` | already pre-patched by Task 2; if it still fires, the patch didn't apply — re-check Step "Patch 01" guard |
| `memchr` discards-qualifiers in boringssl `internal.h` | `patch -d /home/dustin/tf-portable-build/bazel-cache/*/external/boringssl -p1 < ~/Claude/pixinsight-blackwell-tf/patches/03-boringssl-memchr-const.patch` |
| `unknown type int64_t/uint32_t` (tf_runtime) | already in flags (cstdint force-include) — should not fire |
| `invalid linker -fuse-ld=lld` | lld didn't install — check dnf step in container log |
| `clang: error: unsupported CUDA gpu architecture` | means device compile went through clang not nvcc — verify `--config=cuda_nvcc` reached bazel (grep the bazel invocation line in log) |
| anything new | read error, patch source/cache in-place (`.orig` copy first), re-run same command; record the fix in patches/README.md — this is the PROVEN workflow |

- [ ] **Step 3: On success** — `ls -la ~/tf-portable-build/out/libtensorflow.tar.gz` (non-empty, expect 1.5-2.5 GB).

---

## Phase B — DeepSNR delivery portability (runs while Task 4 churns)

### Task 5: stdlib wheel fetcher (replaces pip)

**Files:**
- Create: `~/Claude/deepsnr-gpu-linux/scripts/fetch_ort_wheel.py`

- [ ] **Step 1: Write the fetcher**

```python
#!/usr/bin/env python3
"""Fetch the official onnxruntime-gpu wheel from PyPI — stdlib only, no pip.

Why not pip: Debian 12+/Ubuntu 23.04+ mark the system Python externally
managed (PEP 668), which can refuse pip operations outside a venv. This
fetcher needs nothing beyond python3 itself, so the installer works the same
on every distro.

    usage: fetch_ort_wheel.py <version> <dest_dir>
Prints the downloaded wheel's path on stdout (the installer captures it).
"""
import hashlib
import json
import sys
import urllib.request

MAX_WHEEL_BYTES = 1_500_000_000  # sanity cap (~1.5 GB) — bounds the download loop

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: fetch_ort_wheel.py <version> <dest_dir>")
    ver, dest = sys.argv[1], sys.argv[2]

    # PyPI's JSON API lists every file of a release with URL + sha256.
    meta_url = f"https://pypi.org/pypi/onnxruntime-gpu/{ver}/json"
    with urllib.request.urlopen(meta_url, timeout=30) as r:
        meta = json.load(r)

    # The capi .so files are identical across cp3XX tags (they don't link
    # libpython), so any manylinux x86_64 wheel works; sort for determinism.
    cands = sorted(
        u for u in meta["urls"]
        if u["filename"].endswith(".whl")
        and "manylinux" in u["filename"]
        and "x86_64" in u["filename"]
    , key=lambda u: u["filename"])
    if not cands:
        sys.exit(f"ERROR: no manylinux x86_64 wheel for onnxruntime-gpu {ver}")
    pick = cands[-1]

    out = f"{dest}/{pick['filename']}"
    h = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(pick["url"], timeout=60) as r, open(out, "wb") as f:
        while True:  # bounded by MAX_WHEEL_BYTES below
            chunk = r.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_WHEEL_BYTES:
                sys.exit("ERROR: wheel exceeds sanity cap")
            h.update(chunk)
            f.write(chunk)

    want = pick["digests"]["sha256"]
    if h.hexdigest() != want:
        sys.exit(f"ERROR: sha256 mismatch (got {h.hexdigest()}, want {want})")
    print(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it for real**

Run: `python3 ~/Claude/deepsnr-gpu-linux/scripts/fetch_ort_wheel.py 1.26.0 /tmp/claude/fetch-test`  (mkdir first)
Expected: prints `/tmp/claude/fetch-test/onnxruntime_gpu-1.26.0-...whl`; file ~277 MB; exit 0.

- [ ] **Step 3: Negative test** — run with version `0.0.0`; expect `ERROR:`-prefixed exit, nonzero status.

- [ ] **Step 4: Commit** — `feat: stdlib PyPI wheel fetcher (drops pip; PEP 668-proof)`.

### Task 6: Installer integration

**Files:**
- Modify: `~/Claude/deepsnr-gpu-linux/install-deepsnr-gpu.sh` (phase-1 fetch block + header requirements note)

- [ ] **Step 1: Replace the pip block.** Old lines:

```bash
	echo "[*] Downloading onnxruntime-gpu==$VER from PyPI ..."
	python3 -m pip download "onnxruntime-gpu==$VER" --no-deps -d "$STAGE" >/dev/null
	echo "[*] Extracting ..."
	unzip -o -q "$STAGE"/onnxruntime_gpu-*.whl -d "$STAGE/x"
```

New lines (SCRIPT_DIR resolved once near the top of the file, after `set -euo pipefail`):

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

```bash
	echo "[*] Downloading onnxruntime-gpu==$VER from PyPI (stdlib fetch, no pip) ..."
	WHEEL="$(python3 "$SCRIPT_DIR/scripts/fetch_ort_wheel.py" "$VER" "$STAGE")"
	echo "[*] Extracting ..."
	unzip -o -q "$WHEEL" -d "$STAGE/x"
```

Also update the header comment: requirements are now `python3` + `unzip` (pip no longer used).

- [ ] **Step 2: Re-run phase 1 end-to-end** — `bash ~/Claude/deepsnr-gpu-linux/install-deepsnr-gpu.sh`; expect `[*] Staged to /var/tmp/deepsnr-gpu-stage` and `Next: sudo bash ...`.

- [ ] **Step 3: Commit** — `feat: pip-free installer (PEP 668-safe on Debian/Ubuntu)`.

### Task 7: CPU flag for test_infer.py (needed by container verify)

**Files:**
- Modify: `~/Claude/deepsnr-gpu-linux/scripts/test_infer.py`

- [ ] **Step 1: Minimal diff** — accept optional 2nd arg `cpu`:

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
if len(sys.argv) > 2 and sys.argv[2] == "cpu":
    providers = ["CPUExecutionProvider"]
```
and use `providers=providers` in the InferenceSession call (update the docstring usage line too).

- [ ] **Step 2: Verify locally** — `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 /tmp/claude/ort/venv/bin/python ~/Claude/deepsnr-gpu-linux/scripts/test_infer.py /opt/PixInsight/bin/deepsnr/DeepSNR_weights_v2.onnx cpu`
Expected: `session_providers: ['CPUExecutionProvider']` … `RUN_OK` … `VERDICT: RAN_ON_CPU_ONLY`.

- [ ] **Step 3: Commit** — `feat: optional cpu-only mode for test_infer (container verification)`.

### Task 8: Debian 13 verification script (ORT)

**Files:**
- Create: `~/Claude/deepsnr-gpu-linux/scripts/verify-trixie-ort.sh`

- [ ] **Step 1: Write it**

```bash
#!/bin/bash
# verify-trixie-ort.sh — proves the ORT artifact's portability claim on Debian 13.
# Runs a REAL CPU inference of the DeepSNR model inside a debian:trixie container
# using the exact official wheel the installer fetches (GPU behavior is distro-
# independent and already proven on the host; the container proves glibc/ABI).
# Also audits the CUDA provider lib: its only unresolved deps may be CUDA/cuDNN
# (absent in the container by design) — any glibc version error = FAIL.
#
#   usage: bash scripts/verify-trixie-ort.sh <wheel-path> <model.onnx>
set -euo pipefail
WHEEL="${1:?usage: verify-trixie-ort.sh <wheel> <model.onnx>}"
MODEL="${2:?usage: verify-trixie-ort.sh <wheel> <model.onnx>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

podman run --rm \
	-v "$(realpath "$WHEEL")":/w.whl:ro \
	-v "$(realpath "$MODEL")":/model.onnx:ro \
	-v "$SCRIPT_DIR/test_infer.py":/test_infer.py:ro \
	docker.io/library/debian:trixie bash -c '
	set -e
	apt-get update -qq >/dev/null
	apt-get install -y -qq python3 python3-venv unzip binutils >/dev/null
	python3 -m venv /v
	/v/bin/pip install -q /w.whl numpy
	echo "== CPU inference on trixie =="
	/v/bin/python /test_infer.py /model.onnx cpu
	echo "== CUDA provider ABI audit =="
	unzip -o -q /w.whl -d /x "onnxruntime/capi/*"
	objdump -T /x/onnxruntime/capi/libonnxruntime_providers_cuda.so | grep -oE "GLIBC_[0-9.]+" | sort -uV | tail -3
	ldd /x/onnxruntime/capi/libonnxruntime_providers_cuda.so 2>&1 | grep "not found" | grep -vE "libcu|libnv" \
		&& { echo "FAIL: non-CUDA unresolved deps"; exit 1; } || true
	echo "VERDICT: TRIXIE_OK"
'
```

- [ ] **Step 2: Run it** — `bash scripts/verify-trixie-ort.sh /tmp/claude/fetch-test/onnxruntime_gpu-1.26.0-*.whl /opt/PixInsight/bin/deepsnr/DeepSNR_weights_v2.onnx`
Expected: `RUN_OK` + `VERDICT: RAN_ON_CPU_ONLY` from test_infer, glibc max ≤ 2.28 printed, final `VERDICT: TRIXIE_OK`.

- [ ] **Step 3: Commit** — `feat: Debian 13 verification (CPU inference + ABI audit in container)`.

### Task 9: Release bundle script (ORT mirror)

**Files:**
- Create: `~/Claude/deepsnr-gpu-linux/scripts/make-release-bundle.sh`

- [ ] **Step 1: Write it**

```bash
#!/bin/bash
# make-release-bundle.sh — packages the three GPU ONNX Runtime libraries from
# the OFFICIAL PyPI wheel into a tar.gz for the GitHub Release, for users
# without python3. Provenance (wheel name + sha256) is embedded so anyone can
# reproduce the bundle from PyPI and diff it. MIT license text ships alongside.
#   usage: bash scripts/make-release-bundle.sh <wheel-path> <out-dir>
set -euo pipefail
WHEEL="${1:?usage: make-release-bundle.sh <wheel> <outdir>}"
OUT="${2:?usage: make-release-bundle.sh <wheel> <outdir>}"
VER="$(basename "$WHEEL" | sed -E 's/onnxruntime_gpu-([0-9.]+)-.*/\1/')"
D="$(mktemp -d -p /tmp/claude bundle.XXXX)"
NAME="ort-gpu-$VER-linux-x86_64"

unzip -o -q "$WHEEL" -d "$D/w" "onnxruntime/capi/*" "onnxruntime_gpu-*.dist-info/*"
mkdir -p "$D/$NAME"
cp "$D/w/onnxruntime/capi/libonnxruntime.so."*       "$D/$NAME/"
cp "$D/w/onnxruntime/capi/libonnxruntime_providers_shared.so" "$D/$NAME/"
cp "$D/w/onnxruntime/capi/libonnxruntime_providers_cuda.so"   "$D/$NAME/"
# License: ship whatever license file the wheel's dist-info carries.
cp "$D/w/"onnxruntime_gpu-*.dist-info/licenses/* "$D/$NAME/" 2>/dev/null \
	|| cp "$D/w/"onnxruntime_gpu-*.dist-info/LICENSE* "$D/$NAME/" 2>/dev/null \
	|| echo "WARN: no license file found in dist-info — add ORT MIT text manually"
{
	echo "Provenance: extracted unmodified from the official PyPI wheel"
	echo "wheel: $(basename "$WHEEL")"
	echo "sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
	echo "source: https://pypi.org/project/onnxruntime-gpu/$VER/"
} > "$D/$NAME/PROVENANCE.txt"

mkdir -p "$OUT"
tar -C "$D" -czf "$OUT/$NAME.tar.gz" "$NAME"
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
ls -la "$OUT/$NAME.tar.gz" "$OUT/$NAME.tar.gz.sha256"
```

- [ ] **Step 2: Run it** — `bash scripts/make-release-bundle.sh /tmp/claude/fetch-test/onnxruntime_gpu-1.26.0-*.whl /tmp/claude/ort-release`
Expected: tar.gz ~390 MB + .sha256; `tar -tzf` shows 3 .so + license + PROVENANCE.txt.

- [ ] **Step 3: Commit** — `feat: ORT release-bundle packaging with provenance`.

---

## Phase C — TF verification gates (after Task 4 succeeds)

### Task 10: Portability + arch audit

**Files:**
- Create: `~/Claude/pixinsight-blackwell-tf/scripts/audit-portability.sh`

- [ ] **Step 1: Write it**

```bash
#!/bin/bash
# audit-portability.sh — release gates for a built libtensorflow tarball:
#  (1) glibc floor: no symbol version above the manylinux_2_28 contract (2.28)
#  (2) GPU coverage: SASS for every promised arch + compute_120 PTX present
# GLIBCXX/CXXABI maxima are REPORTED (informational): the manylinux toolchain
# static-links newer libstdc++ pieces, so the load tests are the real C++ gate.
#   usage: bash scripts/audit-portability.sh <libtensorflow.tar.gz> [max_glibc]
set -euo pipefail
TAR="${1:?usage: audit-portability.sh <tarball> [max_glibc]}"
MAXG="${2:-2.28}"
CUOBJ="${CUOBJ:-/usr/local/cuda-12.8/bin/cuobjdump}"
D="$(mktemp -d -p /tmp/claude tfaudit.XXXX)"
tar xzf "$TAR" -C "$D"
SO="$(ls "$D"/lib/libtensorflow.so.2.* | head -1)"

echo "== glibc requirement =="
worst="$(objdump -T "$D"/lib/*.so* | grep -oE 'GLIBC_[0-9.]+' | sed 's/GLIBC_//' | sort -uV | tail -1)"
echo "worst GLIBC symbol: $worst (max allowed: $MAXG)"
if [ "$(printf '%s\n%s\n' "$worst" "$MAXG" | sort -V | tail -1)" != "$MAXG" ]; then
	echo "FAIL: requires glibc $worst > $MAXG"; exit 1
fi
echo "== libstdc++ (informational) =="
objdump -T "$D"/lib/*.so* | grep -oE 'GLIBCXX_[0-9.]+|CXXABI_[0-9.]+' | sort -uV | tail -2
echo "== GPU arches =="
"$CUOBJ" --list-elf "$SO" | grep -oE 'sm_[0-9]+' | sort -u
"$CUOBJ" --list-ptx "$SO" | grep -oE 'compute_[0-9]+' | sort -u
for a in 80 86 89 90 120; do
	"$CUOBJ" --list-elf "$SO" | grep -q "sm_$a" || { echo "FAIL: sm_$a SASS missing"; exit 1; }
done
"$CUOBJ" --list-ptx "$SO" | grep -q "compute_120" || { echo "FAIL: compute_120 PTX missing"; exit 1; }
echo "PASS: glibc<=$MAXG, all 5 SASS arches + compute_120 PTX present"
```

- [ ] **Step 2: Run** — `bash scripts/audit-portability.sh ~/tf-portable-build/out/libtensorflow.tar.gz`; expect final `PASS:` line.
- [ ] **Step 3: Commit** the script.

### Task 11: Container load tests (Debian 13 + AlmaLinux 8 floor)

**Files:**
- Create: `~/Claude/pixinsight-blackwell-tf/scripts/verify-load-containers.sh`

- [ ] **Step 1: Write it**

```bash
#!/bin/bash
# verify-load-containers.sh — loads the built libtensorflow inside (a) Debian 13
# (the requested target) and (b) AlmaLinux 8 (the glibc-2.28 FLOOR — if it loads
# here, it loads on everything newer). ctypes + TF_Version() proves symbol
# resolution and basic C-API function, no GPU needed (kernels are distro-blind).
#   usage: bash scripts/verify-load-containers.sh <libtensorflow.tar.gz>
set -euo pipefail
TAR="${1:?usage: verify-load-containers.sh <tarball>}"

for IMG in docker.io/library/debian:trixie docker.io/library/almalinux:8; do
	echo "== $IMG =="
	podman run --rm -v "$(realpath "$TAR")":/tf.tar.gz:ro "$IMG" bash -c '
		set -e
		if command -v apt-get >/dev/null; then
			apt-get update -qq >/dev/null && apt-get install -y -qq python3 >/dev/null
		else
			dnf install -y -q python3 >/dev/null
		fi
		mkdir -p /tf && tar xzf /tf.tar.gz -C /tf
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
```

- [ ] **Step 2: Run** — expect `TF_Version: 2.19.0` + `VERDICT: LOADS_OK` under BOTH images.
- [ ] **Step 3: Commit** the script.

### Task 12: Host PixInsight GPU run (USER-HANDOFF — needs sudo + GUI)

**Files:** none

- [ ] **Step 1: Stage commands for Dustin** (single-line, paste-safe; data-preserving backup FIRST — the current lib is the only working sm_120 build in existence):

Backup: `sudo cp -a /usr/local/libtensorflow /usr/local/libtensorflow.bak-sm120only`
Install: `sudo tar xzf /home/dustin/tf-portable-build/out/libtensorflow.tar.gz -C /usr/local/libtensorflow --strip-components=0`
(then `sudo bash ~/pixinsight-gpu-fix.sh` to re-assert launcher state)

- [ ] **Step 2: Dustin runs PixInsight** → StarX or NoiseX on a real image; `nvidia-smi` shows VRAM; console clean of INVALID_PTX. Paste console output back.
- [ ] **Step 3 (only if broken): revert** — `sudo rm -rf /usr/local/libtensorflow && sudo mv /usr/local/libtensorflow.bak-sm120only /usr/local/libtensorflow`

---

## Phase D — Packaging, releases (DRAFTS), READMEs, memory

### Task 13: TF release packaging

**Files:**
- Create: `~/Claude/pixinsight-blackwell-tf/scripts/package-release.sh`

- [ ] **Step 1: Write it**

```bash
#!/bin/bash
# package-release.sh — turns the raw bazel lib_package tarball into the named
# release artifact: recompress as xz (smaller; gz→xz typically 25-35% off),
# verify LICENSE presence (Apache-2.0 requires it), embed build provenance.
#   usage: bash scripts/package-release.sh <libtensorflow.tar.gz> <out-dir>
set -euo pipefail
TAR="${1:?usage: package-release.sh <tarball> <outdir>}"
OUT="${2:?usage: package-release.sh <tarball> <outdir>}"
NAME="libtensorflow-2.19.0-gpu-cuda12.8-sm80_120-linux-x86_64"
D="$(mktemp -d -p /tmp/claude tfpkg.XXXX)"

mkdir -p "$D/$NAME"
tar xzf "$TAR" -C "$D/$NAME"
ls "$D/$NAME"/LICENSE* >/dev/null || { echo "FAIL: no LICENSE in lib_package output"; exit 1; }
{
	echo "Built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "Source: tensorflow v2.19.0 (github.com/tensorflow/tensorflow)"
	echo "Build env: quay.io/pypa/manylinux_2_28_x86_64 (glibc 2.28 floor), --config=cuda_nvcc"
	echo "Hermetic: CUDA 12.8.0, cuDNN 9.7.0, python 3.12"
	echo "Arches: sm_80,sm_86,sm_89,sm_90,sm_120 SASS + compute_120 PTX"
	echo "Patches: see patches/ in github.com/dustinspace217/pixinsight-blackwell-tensorflow"
} > "$D/$NAME/PROVENANCE.txt"

mkdir -p "$OUT"
tar -C "$D" -cJf "$OUT/$NAME.tar.xz" "$NAME"   # -J = xz; add XZ_OPT="-9 -T0" via env if size-critical
( cd "$OUT" && sha256sum "$NAME.tar.xz" > "$NAME.tar.xz.sha256" )
SIZE=$(stat -c%s "$OUT/$NAME.tar.xz")
echo "artifact: $OUT/$NAME.tar.xz ($((SIZE/1024/1024)) MB)"
if [ "$SIZE" -ge 2000000000 ]; then
	echo "WARN: >= ~2GB GitHub per-file cap — split: split -b 1900m $NAME.tar.xz $NAME.tar.xz.part"
fi
```

- [ ] **Step 2: Run** — expect artifact + sha256, size printed; split warning only if over cap.
- [ ] **Step 3: Commit** the script.

### Task 14: Draft releases (NOT published until QA passes)

- [ ] **Step 1: TF draft** — `gh release create tf2.19.0-cuda12.8-fat-r1 --repo dustinspace217/pixinsight-blackwell-tensorflow --draft --title "Portable fat-binary libtensorflow 2.19.0 (CUDA 12.8, sm_80-sm_120)" --notes-file <notes>` then `gh release upload` the .tar.xz + .sha256. Notes content: compatibility contract (glibc ≥ 2.28 distro list), arch list, verification summary (audit + trixie/alma loads + PixInsight GPU run), install pointer to README.
- [ ] **Step 2: ORT draft** — same shape on deepsnr-gpu-linux: tag `ort1.26.0-r1`, upload bundle + sha256; notes include provenance (official wheel sha256) and "installer fetches this for you — the bundle is for no-python systems."

### Task 15: README updates (both repos)

**Files:**
- Modify: `~/Claude/pixinsight-blackwell-tf/README.md`
- Modify: `~/Claude/deepsnr-gpu-linux/README.md`

- [ ] **Step 1: TF README** — add sections: **Downloads** (release link, sha256 verify command, "or build it yourself" pointer to scripts/build-portable.sh); **Supported distros** (glibc ≥ 2.28 table: Debian 10+, Ubuntu 20.04+, RHEL/Alma/Rocky 8+, Fedora, Arch, openSUSE; explicitly verified: Debian 13, AlmaLinux 8, Fedora 44; musl/Alpine excluded); **Requirements: NVIDIA GPU** + AMD one-liner ("AMD/ROCm is out of scope: this artifact is CUDA machine code; a ROCm libtensorflow is a separate build we can't test"); reword the "contains no binaries" paragraph → "the repo contains only docs/patches/scripts; prebuilt binaries are published under Releases." Adjust the Fedora-specific LD_LIBRARY_PATH guidance with a Debian/Ubuntu note (CUDA at /usr/local/cuda-12.x via NVIDIA's official installers; cuDNN from NVIDIA's apt repo lands under /usr/lib/x86_64-linux-gnu — point PixInsight.sh at whichever directory holds libcudart.so.12/libcudnn.so.9; link NVIDIA's official install docs rather than asserting package names).
- [ ] **Step 2: DeepSNR README** — same Requirements/AMD lines (plus the vendor-blocked explanation: module hard-codes the CUDA EP); **Downloads** (release bundle for no-python users; installer auto-fetches otherwise); requirements line drops pip ("python3 + unzip"); Debian/Ubuntu path note as above; verified-on list (Fedora 44 host GPU; Debian 13 container CPU inference + ABI audit).
- [ ] **Step 3: Commit** both repos — `docs: multi-distro support, downloads, NVIDIA-only scope`.

### Task 16: Memory updates

- [ ] **Step 1:** Update `reference_pixinsight_gpu_fedora.md` (STATUS: portable fat build exists; release link; container recipe location) and `reference_deepsnr_gpu_onnx.md` (release link; pip-free installer; trixie verification). Add MCP observations on the relevant entities (per mid-session triggers).

---

## Phase E — QA review, then publish

### Task 17: QA review (BEFORE anything publishes)

- [ ] **Step 1:** Dispatch in ONE parallel batch: **code-reviewer**, **test-analyzer**, **security-auditor** (public install scripts that download binaries and write into /opt as root — security review is mandatory). Scope: all new/modified files in both repos. Opus floor enforced by hook.
- [ ] **Step 2:** Three-phase review per CLAUDE.md (independent findings → cross-exam → Head-Agent synthesis). Post to GitHub Discussions if the repos have it enabled; otherwise capture Phase C synthesis in THIS plan doc (the artifact layer is the Discussions post, not a precondition).
- [ ] **Step 3:** Fix P0/P1 findings; defer others via the deferment register format; commit fixes.

### Task 18: Publish

- [ ] **Step 1:** `git push` both repos (main).
- [ ] **Step 2:** Publish the two draft releases (`gh release edit <tag> --draft=false` on each repo).
- [ ] **Step 3:** Verify public URLs render; update plan Status block to DONE; deviation summary in chat + plan doc.

## Deviations log
(append at each commit boundary)

**2026-06-10 ~03:00 (Tasks 1-11):**
1. *Behavioral-fix*: gpu_prim patch applies with `-p0` not `-p1` (bare repo-relative
   header paths) + `--batch` (2c8da29). Plan-as-written had -p1; failed on first run.
2. *Behavioral-fix*: SELinux (enforcing, Fedora) denied the repo bind-mount — Dustin
   diagnosed it from the AVC alert. Reworked to single `:Z`-labeled work-dir mount,
   script copied in (5a01a85). Same stage-dir pattern then applied proactively to
   BOTH container verify scripts (never relabel /opt/PixInsight or the git repos).
3. *Behavioral-fix*: host clang-21 (Alma 8 ships 21, newer than assumed) hard-errors
   on unused `--cuda-path` → added `-Qunused-arguments`, the same fix cuda_clang
   itself carries (9d2e2e8).
4. *Behavioral-fix*: fetch_ort_wheel.py excludes free-threaded (cp3XXt) wheels —
   deterministic sort had picked cp314t, which the trixie pip can't install (1574949).
5. *Behavioral-fix*: trixie verify keeps original wheel filename (PEP 427 parsing)
   and pip-resolves the container-matching build for the inference half; ABI audit
   stays on the exact shipped wheel (1574949).
6. *Behavioral-fix*: ORT bundle takes LICENSE from `onnxruntime/LICENSE` (wheel has
   no dist-info license file) (2f0132c).
7. *Tooling note*: Claude Code H.replace renderer bug killed several podman
   stop/kill/launch invocations outright tonight — all container ops now route
   through script files. No effect on shipped artifacts.
None of these change WHAT ships; all are how-it-runs fixes discovered by execution.

**2026-06-10 ~05:45 (Tasks 4 iteration loop + 10-11 + 13):**
8. *Behavioral-change (build config)*: cuda_nvcc → **cuda_clang** (61445b8) after TWO
   nvcc-only error classes (GpuLaneId clang-builtin guard → patch 04, kept since it's
   correct under both compilers; Eigen-half alignas(4)-vs-(2) in split/concat kernels).
   The nvcc rationale (old container clang) was invalidated by Alma 8 shipping
   clang-21 (sm_120-aware). Final config = the PROVEN Fedora host config.
9. *New fix (container-only)*: CUDA driver stub symlinked in-container for build-time
   op generators (3967f5d) — they DT_NEED libcuda.so.1; no driver in container.
10. *Spec-assumption corrected*: clang names embedded PTX by target arch
   (sm_120.ptx), not nvcc's compute_120 — audit gate updated (bf0c9d3). PTX IS present.
11. *Script-bug class fixed twice*: pipefail+SIGPIPE — `cuobjdump|grep -q` and
   `find|head -1` both die at first match (bf0c9d3, 6ceca51). Pattern: file-route
   listings; find uses -print -quit.
12. *Discovery (documentation-relevant)*: the hermetic GPU build hard-links its CUDA
   deps (DT_NEEDED incl. libnccl.so.2 — NOT in the CUDA toolkit — and libcupti.so.12)
   and libcuda.so.1: NO lazy-dlopen CPU fallback unlike Google's 2.18. Matches the
   working host build's profile (host cuda dir already carries nccl+cupti). MUST be
   in README requirements + is embedded in PROVENANCE.txt (5da396a).
13. *Deferment* DEF-PORT-01 (LOW): boringssl memchr patch (03) unused under clang-21 —
   keep in patches/ for clang-22+ host builds; note added to patches/README in Task 15.
