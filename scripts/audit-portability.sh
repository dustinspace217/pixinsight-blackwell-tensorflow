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
echo "== GPU arches =="
"$CUOBJ" --list-elf "$SO" | grep -oE 'sm_[0-9]+' | sort -u
"$CUOBJ" --list-ptx "$SO" | grep -oE 'compute_[0-9]+' | sort -u
for a in 80 86 89 90 120; do
	"$CUOBJ" --list-elf "$SO" | grep -q "sm_$a" || { echo "FAIL: sm_$a SASS missing"; exit 1; }
done
"$CUOBJ" --list-ptx "$SO" | grep -q "compute_120" || { echo "FAIL: compute_120 PTX missing"; exit 1; }
echo "PASS: glibc<=$MAXG, all 5 SASS arches + compute_120 PTX present"
