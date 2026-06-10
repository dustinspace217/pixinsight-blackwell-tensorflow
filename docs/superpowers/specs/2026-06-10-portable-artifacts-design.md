# Portable GPU Artifacts — Design Spec
Cross-repo spec covering **pixinsight-blackwell-tensorflow** (this repo) and
**deepsnr-gpu-linux**. Goal: artifacts and installers usable on Debian 13 and
as many modern (last-5-years) glibc distros as possible, for broad public
sharing.

## Status (updated 2026-06-10)
Phase: 0 of 5 (design approved, spec written)
Done: design discussion; approach B (manylinux_2_28) approved; fat-binary GPU
  coverage approved; AMD declared out of scope; spec drafted
Next: user reviews this spec → writing-plans → Phase A kickoff (container build)
Blocked: nothing

## Decisions already made (by Dustin, 2026-06-10 session — do not re-ask)
- **Fat GPU binary**: SASS for sm_80, sm_86, sm_89, sm_90, sm_120 + compute_120
  PTX. One artifact serves Ampere → Blackwell. (Chosen over Blackwell-only,
  knowing build time ~doubles and artifact may approach GitHub's 2 GB file cap.)
- **Build environment: approach B** — manylinux_2_28 container (AlmaLinux 8,
  glibc 2.28 floor) with a pinned Blackwell-aware LLVM (≥20). Rejected: A
  (TF official manylinux2014 image — older clang, extra 5 yrs of compat buys
  nothing for PixInsight+RTX users) and C (Debian 11 base — loses RHEL 8 family).
- **Publish prebuilt binaries as GitHub Releases** on both repos. This is a
  deliberate policy change to the READMEs' "contains no binaries" stance →
  reword to "repo contains only docs/patches/scripts; binaries live in Releases."
  Licensing permits it: TF = Apache-2.0 (ship LICENSE + NOTICE in tarball),
  ONNX Runtime = MIT (ship license text in bundle).
- **AMD is out of scope**, stated explicitly in both READMEs:
  - TF artifact is CUDA/SASS by construction; a ROCm libtensorflow is possible
    in principle but is a separate untestable artifact (no AMD hardware here).
  - DeepSNR cannot use AMD at all at our layer: the signed module calls
    AppendExecutionProvider_CUDA_V2 (verified via strings 2026-06-09); provider
    choice is compiled into StarNet's binary. Vendor-level limitation.
  - README scope line: "Requirements: NVIDIA GPU." One-sentence AMD pointer.
- **musl distros (Alpine) excluded** — glibc artifacts; note in README, no work.

## Artifact 1 — portable fat libtensorflow (this repo)

### What
`libtensorflow` C library, TF **v2.19.0** (only version with BOTH the C-lib
target and hermetic CUDA 12.8 — see PLAN.md version decision), hermetic CUDA
**12.8.0** + cuDNN **9.7.0** (the values the PROVEN 2.19 host build used per
PLAN.md status + repo README; the 12.8.1/9.8.0 pair in PLAN.md's command block
was the abandoned 2.21 attempt — confirm exact values from build.log at
kickoff), built INSIDE a `quay.io/pypa/manylinux_2_28_x86_64`
Podman container so the binary links **glibc ≤ 2.28** symbols.

Compatibility contract: any x86_64 glibc-distro with glibc ≥ 2.28 — Debian 10+
(13 = 2.41 ✓), Ubuntu 20.04+, RHEL/Alma/Rocky 8+, Fedora, Arch, openSUSE.
Same floor as the official onnxruntime-gpu wheel (deliberate symmetry).

### Build mechanics (inherit the PROVEN host build, PLAN.md 2026-06-07)
- Target: `//tensorflow/tools/lib_package:libtensorflow` (tarball: lib + include
  + LICENSE assembly).
- Pin ALL hermetic values via command-line `--repo_env` (NOT .tf_configure rc —
  proven gotcha: rc-layered defaults win otherwise and cuDNN stays 9.3.0):
  `HERMETIC_PYTHON_VERSION=3.12, HERMETIC_CUDA_VERSION=12.8.0,
  HERMETIC_CUDNN_VERSION=9.7.0,
  HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_80,sm_86,sm_89,sm_90,sm_120,compute_120`
  (exact list syntax = verify at build time; `sm_120` alone is proven).
- Keep proven extra flags: `--copt=-Wno-error`, cstdint force-include
  (`--cxxopt=-include --cxxopt=cstdint` + host variants), c23/offsetof warning
  suppressions (harmless if the container clang doesn't need them).
- Toolchain inside container: bazelisk; LLVM pinned ≥20 (Blackwell-aware) from
  llvm.org release tarballs (AlmaLinux 8 repos likely top out too old — verify);
  lld REQUIRED (proven fix #4).
- Patches: start from the repo's three (`gpu_prim` cub-const, `cutlass
  set_slice_3x3`, `boringssl memchr`). gpu_prim = TF source patch; cutlass +
  boringssl = bazel external cache patches applied after first fetch/failure,
  then re-run WITHOUT `bazel clean` (proven workflow). Container clang (20/21)
  may need a DIFFERENT subset than host clang 22 — expect iteration ("debugging
  usually required" inherits). Update patches/README.md with which compiler
  surfaces which.
- Resources (proven safe attempt 1, no OOM): `systemd-run --user --scope
  --collect -p MemoryMax=32G -p MemoryHigh=28G -p MemorySwapMax=0 -p
  TasksMax=400` wrapping the podman run; `--jobs=14` (workstation core-cap rule;
  16 was proven — Dustin may bump). Bind-mount a host work dir (~/tf-portable-build)
  for bazel cache + outputs: survives container removal, keeps container storage
  small, /tmp(-tmpfs) never used. Disk budget ~100 GB (1.7 TB free — verified
  2026-06-07; re-verify at kickoff).
- Fat-binary wall-clock: single-arch was ~28.5k actions; expect roughly 2-3×
  kernel-compile time. Overnight async run.

### Verification gates (ALL must pass before release)
1. `cuobjdump --list-elf` → sm_80, sm_86, sm_89, sm_90, sm_120 all present;
   `--list-ptx` → compute_120.
2. glibc symbol audit: `objdump -T libtensorflow.so* | grep GLIBC_` max
   version ≤ 2.28.
3. Load test in `debian:trixie` podman container: python3 ctypes dlopen +
   `TF_Version()` (proves ABI/glibc claim on the actual requested distro).
4. Host end-to-end: BACK UP /usr/local/libtensorflow (data-preserving rule —
   it is currently the only working sm_120 build in existence), install fat
   lib, launch PixInsight, StarX/NoiseX GPU run, nvidia-smi VRAM check, no
   INVALID_PTX. Revert path documented.

### Packaging / release
- `libtensorflow-2.19.0-gpu-cuda12.8-sm80_120-linux-x86_64.tar.xz`
  (lib/ + include/ + LICENSE + NOTICE + THIRD_PARTY from lib_package) + .sha256.
- If xz > 2 GB (GitHub per-file cap): `split` into .partNN + cat instructions.
- GitHub Release on pixinsight-blackwell-tensorflow, tag e.g.
  `tf2.19.0-cuda12.8-fat-r1`. README "Downloads" section.

## Artifact 2 — DeepSNR/ONNX delivery portability (deepsnr-gpu-linux)

The binary is ALREADY portable: official onnxruntime-gpu 1.26.0 wheel is tagged
`manylinux_2_27`/`2_28` (verified 2026-06-09). No build. Work = delivery:

1. **Installer: drop pip entirely.** `pip download` can hit PEP 668
   ("externally-managed-environment") on Debian 12+/Ubuntu 23.04+. Replace with
   stdlib-only fetch: python3 urllib → PyPI JSON API
   (`https://pypi.org/pypi/onnxruntime-gpu/<ver>/json`) → select the
   cp3xx-manylinux_2_28_x86_64 wheel URL → download → unzip. New deps: just
   python3 + unzip. Version auto-detect from bundled lib (strings) unchanged.
2. **Verify on trixie:** podman `debian:trixie` + the three libs + CPU inference
   of /opt/PixInsight/bin/deepsnr/DeepSNR_weights_v2.onnx (CPUExecutionProvider;
   GPU-in-container not needed — GPU path proven on host, kernels are
   distro-independent; the container proves glibc/ABI).
3. **Release mirror:** `ort-gpu-1.26.0-linux-x86_64.tar.gz` (3 .so + ORT MIT
   license + provenance note "extracted unmodified from official PyPI wheel
   <name> sha256 <hash>") + .sha256, for users without python3/pip.
4. **README:** Debian/Ubuntu section (apt CUDA 12 + cuDNN 9 paths — cuDNN via
   apt lands in /usr/lib/x86_64-linux-gnu, CUDA in /usr/local/cuda-12.x; adjust
   the PixInsight.sh LD_LIBRARY_PATH line accordingly), NVIDIA-only scope line,
   Downloads section.

## Sequencing
- **Phase A**: TF container build scaffolding + kickoff (long pole, runs async).
- **Phase B** (parallel, while A churns): DeepSNR installer rewrite + trixie
  verification + release bundle prep.
- **Phase C**: TF verification gates 1-4.
- **Phase D**: Releases + README updates (both repos) + memory updates.
- **Phase E**: QA review per Post-Coding Process — code-reviewer +
  test-analyzer minimum (stabilization-pass slim set applies only later;
  this is first-pass public portfolio work, so walk the roster: add
  security-auditor [public install scripts piping downloads to root-level
  install] — likely set: code-reviewer, test-analyzer, security-auditor).
  Then share.

## Open empirical questions (resolve during build, not by assertion)
- LLVM 20 vs 21 availability/behavior in manylinux_2_28; which proven patches
  still fire under it.
- Exact HERMETIC_CUDA_COMPUTE_CAPABILITIES multi-arch syntax.
- Final artifact size vs 2 GB cap.
- Whether lib_package tarball needs include/ pruning for size.

## Out of scope
- AMD/ROCm (vendor-blocked for DeepSNR; untestable separate artifact for TF).
- macOS/Windows; musl/Alpine; non-x86_64 arches (aarch64 possible future ask —
  would be a separate build lane, do not fold in now).
- Changing what the artifacts DO — this is portability + delivery only.
