#!/bin/bash
# pixinsight-gpu-fix.sh
# ---------------------------------------------------------------------------
# Applies (and re-applies) the CUDA 12.8 + TensorFlow GPU configuration for
# PixInsight on Linux. Run it ONCE to set up, and AGAIN after every PixInsight
# update -- updates revert both changes this script makes.
#
# WHY this is needed (two separate reverts a PI update performs):
#   1. PI updates RESTORE the old bundled libtensorflow into
#      /opt/PixInsight/bin/lib. That bundled build is from the CUDA-11.7 era and
#      cannot drive a Blackwell (RTX 50-series, sm_120) GPU. While present, it
#      also SHADOWS the external GPU build in /usr/local/libtensorflow, because
#      the launcher searches PI's own lib dir first. So it must be removed.
#   2. PI updates OVERWRITE /opt/PixInsight/bin/PixInsight.sh, reverting the
#      LD_LIBRARY_PATH line so it no longer points at our CUDA 12.8 + TensorFlow
#      libraries -> PixInsight silently falls back to CPU.
#
# WHY edit the launcher instead of using ldconfig: scoping the CUDA paths to
# PixInsight's own process (via its launcher) keeps the CUDA 12.8 libraries from
# leaking into every other program (which previously broke wget by loading PI's
# old OpenSSL), and it lets us put cuda-12.8 FIRST so libcudnn.so.9 resolves to
# the CUDA-12 build instead of the CUDA-13 one also present on this system.
#
# Idempotent: safe to run any number of times (it checks before changing).
# Requires root (touches /opt): run with  sudo bash ~/pixinsight-gpu-fix.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PI_LIB="/opt/PixInsight/bin/lib"
PI_SH="/opt/PixInsight/bin/PixInsight.sh"
# Trailing colon matters: this string is PREPENDED in front of the launcher's
# existing "$dirname/lib:$dirname", so CUDA + TensorFlow are searched first.
CUDA_PREPEND="/usr/local/cuda-12.8/lib64:/usr/local/libtensorflow/lib:"

# --- 1. Remove PI's bundled TensorFlow so the external GPU build is used ---
if ls "$PI_LIB"/libtensorflow* >/dev/null 2>&1; then
	rm -f "$PI_LIB"/libtensorflow*
	echo "[fix] Removed bundled libtensorflow from $PI_LIB"
else
	echo "[fix] No bundled libtensorflow present (good)"
fi

# --- 2. Prepend CUDA 12.8 + TensorFlow to the launcher's LD_LIBRARY_PATH ---
# grep guard keeps this idempotent: if "cuda-12.8" already appears in the
# launcher we leave it alone (avoids stacking duplicate path entries).
if grep -q "cuda-12.8" "$PI_SH"; then
	echo "[fix] PixInsight.sh already carries the CUDA 12.8 path (good)"
else
	# Keep a pristine copy of whatever the (possibly newly-updated) vendor
	# launcher looked like, the first time we touch it. -n = never clobber.
	cp -n "$PI_SH" "$PI_SH.orig" 2>/dev/null || true
	# '#' is the sed delimiter so the '/' characters in the paths need no escaping.
	# We anchor on the start of the assignment line, so PI changing anything else
	# in the script in a future version does not affect this edit.
	sed -i "s#^LD_LIBRARY_PATH=#LD_LIBRARY_PATH=${CUDA_PREPEND}#" "$PI_SH"
	echo "[fix] Prepended CUDA 12.8 + TensorFlow path to $PI_SH"
fi

echo "[fix] Done. Launch PixInsight normally -- GPU acceleration is active."
