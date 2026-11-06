#!/usr/bin/env bash
#SBATCH --job-name=msw-c06-amp-diag
#SBATCH --partition=s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/xinshen/_state/jittor-ms-swift-cuda-20261102/tasks/CUDA-06/output/pretrain-amp-diag-%j.out
#SBATCH --error=/home/xinshen/_state/jittor-ms-swift-cuda-20261102/tasks/CUDA-06/output/pretrain-amp-diag-%j.err
set -euo pipefail
TASK_ROOT=/home/xinshen/_state/jittor-ms-swift-cuda-20261102/tasks/CUDA-06
SOURCE_ROOT=/home/xinshen/_state/jittor-ms-swift-cuda-20261102/source-audit/CUDA-06/pretrain
BASE_PYTHON=/home/xinshen/_state/jittor-ms-swift-cuda-20261102/envs/formal-candidate/bin/python
REPO="$SOURCE_ROOT/repository"
CASE_ROOT="$TASK_ROOT/pretrain/grad-diagnostic-amp-$SLURM_JOB_ID"
SITE="$TASK_ROOT/export-merge/attempt-157176/site"
DIAG_SITE="$TASK_ROOT/pretrain/grad-diagnostic-amp-159277/site-amp5776"
PATCH_ROOT="$TASK_ROOT/pretrain/grad-diagnostic-amp-5776/source"
CACHE="$TASK_ROOT/cache/cuda06-core507-cp311-cuda122-sm89"
BASE_MODEL=/home/xinshen/.cache/modelscope/hub/models/Qwen/Qwen2-0___5B
DATASET="$SOURCE_ROOT/inputs/cuda16-existing-train-fixture.jsonl"
OUTPUT_DIR="$CASE_ROOT/output"
RECEIPT="$CASE_ROOT/receipt.json"
mkdir -p "$CASE_ROOT/tmp"
test -d "$SITE/jittor" && test -d "$SITE/torch" && test -f "$DATASET"
test -f "$BASE_MODEL/model.safetensors" && test -f "$PATCH_ROOT/amp-overlay-manifest.json"
test -d "$DIAG_SITE/jittor" && test -d "$DIAG_SITE/torch"
test ! -e "$OUTPUT_DIR"
export PATH="$(dirname "$BASE_PYTHON"):/home/xinshen/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin:$PATH"
export nvcc_path=/home/xinshen/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc
export JT_BUILD_NVCC_PATH="$nvcc_path"
export PYTHONPATH="$DIAG_SITE:$REPO/tools" CUDA_VISIBLE_DEVICES=0
export HOME="$CASE_ROOT/home" TMPDIR="$CASE_ROOT/tmp"
export JITTOR_HOME="$CACHE/jittor" JITTOR_TORCH_CACHE_ROOT="$CACHE/torch-shim" XDG_CACHE_HOME="$CACHE/xdg"
export CUDA_CACHE_PATH="$CACHE/cuda" TORCH_HOME="$CACHE/torch" TORCH_EXTENSIONS_DIR="$CACHE/torch-extensions"
export JITTOR_TORCH_EXTENSIONS_DIR="$CACHE/jittor-torch-extensions" TRITON_HOME="$CACHE/triton" TRITON_CACHE_DIR="$CACHE/triton/cache"
export HF_HOME="$TASK_ROOT/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export JITTOR_TORCH_SHIM=1 backend_fallback=error use_cuda=1 use_parallel_op_compiler=0
export JT_USE_CUDA=1 JT_BUILD_CUDA_ARCHS=89 JT_USE_PARALLEL_OP_COMPILER=0 JT_CUDA_KERNEL_MATH=strict
export CUDA06_BASE_MODEL="$BASE_MODEL" CUDA06_DATASET="$DATASET" CUDA06_OUTPUT_DIR="$OUTPUT_DIR" CUDA06_RECEIPT="$RECEIPT"
export CUDA06_DIAGNOSTIC_SITE="$DIAG_SITE" CUDA06_AMP_OVERLAY_MANIFEST="$PATCH_ROOT/amp-overlay-manifest.json"
export CUDA06_EXPECTED_MODEL_SHA256=9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad
export CUDA06_EXPECTED_DATASET_SHA256=a187929eb0f6d1249ff3bda8f3ecd61ebf1b90405631855fc4f82c1861399092
export CUDA06_SOURCE_HEAD=compat-amp-rmsnorm-5776eca
export CUDA06_TRAIN_FP16=false
cd "$REPO"
"$BASE_PYTHON" -m pytest -q -s /home/xinshen/projects/ms-swift-cuda/jittor-lab/worktrees/msw-c06-export-merge/tools/task_cuda06_pretrain/test_grad_diagnostic_amp_fix.py
