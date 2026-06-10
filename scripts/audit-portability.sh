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
D="$(mktemp -d -p "${TMPDIR:-/tmp}" tfaudit.XXXX)"
trap 'rm -rf "$D"' EXIT
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
echo "== load-time dependency contract =="
# The artifact is built with include_cuda_libs=false: CUDA must be loaded
# LAZILY (dlopen at first GPU use → announced CPU fallback), never via
# DT_NEEDED — a hard-linked CUDA soname here would reintroduce the
# fails-to-load-without-GPU behavior the 2026-06-10 user directive rejected.
objdump -p "$D"/lib/*.so* | grep NEEDED > "$D/needed.txt" || true
if grep -E "libcu|libnv|libnccl" "$D/needed.txt"; then
	echo "FAIL: CUDA/NVIDIA soname in DT_NEEDED — lazy-loading contract broken"
	exit 1
fi
echo "PASS: no CUDA sonames in DT_NEEDED (lazy dlopen contract)"
# Run cuobjdump ONCE per listing into files, grep the files. Piping cuobjdump
# straight into `grep -q` is a pipefail trap: grep exits at first match,
# cuobjdump dies on SIGPIPE, and the pipeline "fails" despite a successful
# match. Files also avoid 7 redundant scans of a ~1 GB shared object.
"$CUOBJ" --list-elf "$SO" > "$D/elf.txt"
"$CUOBJ" --list-ptx "$SO" > "$D/ptx.txt"
echo "== GPU arches (SASS) =="
grep -oE 'sm_[0-9]+' "$D/elf.txt" | sort -u
echo "== embedded PTX modules =="
# clang names embedded PTX by TARGET arch (foo.sm_120.ptx), unlike nvcc's
# compute_XY labels — match either; informational grep guarded for pipefail.
grep -oE 'compute_[0-9]+|sm_[0-9]+' "$D/ptx.txt" | sort -u || true
for a in 80 86 89 90 120; do
	grep -q "sm_$a" "$D/elf.txt" || { echo "FAIL: sm_$a SASS missing"; exit 1; }
done
grep -qE "compute_120|sm_120" "$D/ptx.txt" || { echo "FAIL: cc-12.0 PTX missing (no forward-JIT path)"; exit 1; }
echo "PASS: glibc<=$MAXG, all 5 SASS arches + cc-12.0 PTX present"
