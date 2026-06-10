#!/bin/bash
# verify-load-containers.sh — loads the built libtensorflow inside (a) Debian 13
# (the requested target) and (b) AlmaLinux 8 (the glibc-2.28 FLOOR — if it loads
# here, it loads on everything newer). ctypes + TF_Version() proves symbol
# resolution and basic C-API function, no GPU needed (kernels are distro-blind).
#
# SELinux note: the tarball is COPIED into a throwaway :Z-labeled stage dir —
# never bind-mount + relabel the original artifact's directory.
#   usage: bash scripts/verify-load-containers.sh <libtensorflow.tar.gz>
set -euo pipefail
TAR="${1:?usage: verify-load-containers.sh <tarball>}"

STAGE="$(mktemp -d -p "${TMPDIR:-/tmp}" tfload.XXXX)"
trap 'rm -rf "$STAGE"' EXIT
cp "$TAR" "$STAGE/tf.tar.gz"

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
