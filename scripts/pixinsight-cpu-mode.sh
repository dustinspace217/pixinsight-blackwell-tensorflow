#!/bin/bash
# pixinsight-cpu-mode.sh
# ===========================================================================
# BREAK-GLASS FALLBACK: make PixInsight's AI tools (StarXTerminator,
# NoiseXTerminator, BlurXTerminator, StarNet) WORK by running them on the CPU.
#
# WHY YOU MIGHT NEED THIS
#   Your RTX 5080 is an NVIDIA "Blackwell" GPU (compute capability sm_120).
#   The only TensorFlow library PixInsight can use on Linux (libtensorflow
#   2.18 -- the last one Google ever published) has NO Blackwell GPU kernels.
#   So while GPU acceleration is active, the RC Astro tools crash with:
#       CUDA_ERROR_INVALID_PTX  /  CUDA_ERROR_INVALID_HANDLE
#   Until a Blackwell-capable libtensorflow.so is built from source, this
#   script forces PixInsight onto the CPU so the tools simply WORK (just
#   unaccelerated). Your Ryzen 9 9950X3D handles them fine -- slower than a
#   working GPU, but reliable.
#
# WHAT IT DOES (all reversible; nothing system-critical, no driver/kernel/boot)
#   1. Removes PixInsight's bundled libtensorflow so the external build is used.
#   2. Ensures the launcher can find the TensorFlow + CUDA libraries.
#   3. Adds  export CUDA_VISIBLE_DEVICES=-1  to the launcher. That hides the GPU
#      from TensorFlow, so TensorFlow runs everything on the CPU -> no Blackwell
#      JIT crash.
#
# USAGE
#   sudo bash ~/pixinsight-cpu-mode.sh          # enable CPU mode (default)
#   sudo bash ~/pixinsight-cpu-mode.sh gpu      # undo: re-enable GPU attempts
#                                               # (only useful once a Blackwell
#                                               #  libtensorflow.so is built)
#
# IMPORTANT: a PixInsight UPDATE overwrites the launcher and restores the
#   bundled libtensorflow, undoing this. Just re-run this script after any
#   PixInsight update.
#
# MANUAL EQUIVALENT (if you can't run the script at all)
#   Edit  /opt/PixInsight/bin/PixInsight.sh  and add this single line just
#   before the last line (the one starting with  eval ):
#       export CUDA_VISIBLE_DEVICES=-1
#   To undo later, delete that line. (sudo nano /opt/PixInsight/bin/PixInsight.sh)
# ===========================================================================
set -euo pipefail

PI_LIB="/opt/PixInsight/bin/lib"
PI_SH="/opt/PixInsight/bin/PixInsight.sh"
# Library dirs the launcher needs so TensorFlow + its CUDA deps resolve.
CUDA_PREPEND="/usr/local/cuda-12.8/lib64:/usr/local/libtensorflow/lib:"
MODE="${1:-cpu}"

if [ ! -f "$PI_SH" ]; then
	echo "[cpu-mode] ERROR: $PI_SH not found -- is PixInsight installed at /opt/PixInsight?"
	exit 1
fi

# ---- UNDO MODE: re-enable GPU attempts -------------------------------------
if [ "$MODE" = "gpu" ] || [ "$MODE" = "--undo" ]; then
	if grep -q "CUDA_VISIBLE_DEVICES=-1" "$PI_SH"; then
		sed -i '/^export CUDA_VISIBLE_DEVICES=-1$/d' "$PI_SH"
		echo "[cpu-mode] Removed CUDA_VISIBLE_DEVICES=-1 -> GPU attempts re-enabled."
		echo "[cpu-mode] (Only do this once a Blackwell-capable libtensorflow.so is installed.)"
	else
		echo "[cpu-mode] No CPU lock present; nothing to undo."
	fi
	exit 0
fi

# ---- ENABLE CPU MODE -------------------------------------------------------

# 1. Use the external libtensorflow, not PI's bundled (Blackwell-incompatible) one.
if ls "$PI_LIB"/libtensorflow* >/dev/null 2>&1; then
	rm -f "$PI_LIB"/libtensorflow*
	echo "[cpu-mode] Removed bundled libtensorflow (external build will be used)."
fi

# 2. Make sure the launcher can find TensorFlow + CUDA libs (idempotent).
if ! grep -q "cuda-12.8" "$PI_SH"; then
	cp -n "$PI_SH" "$PI_SH.orig" 2>/dev/null || true
	sed -i "s#^LD_LIBRARY_PATH=#LD_LIBRARY_PATH=${CUDA_PREPEND}#" "$PI_SH"
	echo "[cpu-mode] Added TensorFlow/CUDA library path to launcher."
fi

# 3. Force CPU by hiding the GPU from TensorFlow (idempotent).
if grep -q "CUDA_VISIBLE_DEVICES=-1" "$PI_SH"; then
	echo "[cpu-mode] CPU mode already enabled (good)."
else
	# Insert right after a stable existing export line in the vendor launcher...
	sed -i '/^export MKL_ENABLE_INSTRUCTIONS=AVX2/a export CUDA_VISIBLE_DEVICES=-1' "$PI_SH"
	# ...and if that anchor wasn't there, fall back to inserting before the eval.
	grep -q "CUDA_VISIBLE_DEVICES=-1" "$PI_SH" || sed -i '/^eval /i export CUDA_VISIBLE_DEVICES=-1' "$PI_SH"
	echo "[cpu-mode] Enabled: PixInsight will run the AI tools on the CPU."
fi

echo "[cpu-mode] Done. Launch PixInsight normally; StarX/NoiseX/BlurX will use the CPU."
