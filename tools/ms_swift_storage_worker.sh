#!/usr/bin/env bash
# Run INSIDE a new single-node Slurm allocation. The driver owns stage publication.
set -euo pipefail
: "${MSW_PYTHON:?set an absolute Python interpreter}"
: "${MSW_EVIDENCE_ROOT:?set a fresh durable run path}"
: "${MSW_RETENTION_PLAN:?set the checkpoint contract JSON path}"
: "${MSW_SCRATCH_PEAK_BYTES:?include raw tensors, models and save staging}"
: "${MSW_DURABLE_PEAK_BYTES:?include retained checkpoints and publication staging}"
: "${MSW_ESTIMATED_FILES:?declare inode budget}"
: "${MSW_RAM_RESERVE_BYTES:?declare resident training memory peak for RAM-backed scratch}"
: "${SLURM_JOB_ID:?run only inside a Slurm worker}"
test "$#" -ge 1 || { echo 'usage: ms_swift_storage_worker.sh driver.py [args...]' >&2; exit 64; }
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export MSW_STORAGE_TOOL="$TOOLS/ms_swift_artifact_storage.py"
"$MSW_PYTHON" "$MSW_STORAGE_TOOL" init --evidence "$MSW_EVIDENCE_ROOT" \
  --plan "$MSW_RETENTION_PLAN" --scratch-peak "$MSW_SCRATCH_PEAK_BYTES" \
  --durable-peak "$MSW_DURABLE_PEAK_BYTES" --files "$MSW_ESTIMATED_FILES" --ram-reserve "$MSW_RAM_RESERVE_BYTES"
export MSW_WORK_ROOT="/tmp/msw-$(id -u)-${SLURM_JOB_ID}"
export TMPDIR="$MSW_WORK_ROOT/t" PYTHONPYCACHEPREFIX="$MSW_WORK_ROOT/pycache" XDG_CACHE_HOME="$MSW_WORK_ROOT/xdg"
export MSW_RAW_ROOT="$MSW_WORK_ROOT/raw" MSW_TRAINER_ROOT="$MSW_WORK_ROOT/trainer" MSW_MODEL_ROOT="$MSW_WORK_ROOT/models"
# JITTOR_HOME/JITTOR_TORCH_CACHE_ROOT and read-only model inputs stay caller-pinned.
trap 'rc=$?; printf "exit_code=%s job=%s\n" "$rc" "$SLURM_JOB_ID" > "$MSW_EVIDENCE_ROOT/logs/worker-exit.txt"' EXIT
"$MSW_PYTHON" "$@" > >(tee -a "$MSW_EVIDENCE_ROOT/logs/driver.log") 2>&1
