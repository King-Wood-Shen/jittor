# ms-swift / JTorch CUDA integration

**Status:** bounded CUDA integration accepted for the listed cases; full structure gate has one known path-dependent failure. **Date:** 2026-10-21. **Official base:** `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e` (`origin/2.0-refactor`). **Maintainer:** ms-swift CUDA/JTorch integration. **Review when:** the upstream base advances or any recorded CUDA contract changes.

## Source selection

Integrated selected, non-equivalent changes from these fetched personal branches:

- `fix/msw-cuda-inference` — `33d686e419f5f07107d08bf0c2018843cfdc8265`
- `fix/msw-cuda-parallel` — `4c15324d218acc5bbc7ccc663223236a27bb7a3b`
- `fix/ms-swift-cuda-adaptation-20261020-sync` — `bdc4bc6dd2f3a3b698f76748375785c609843628`

The selected changes cover CUDA cross-entropy target placement, lazy CUDA driver initialization for the Triton bridge, child-process Torch shim setup, TorchScript eager annotation, iterable DataLoader batching, artifact storage isolation/publication, and JTorch CUDA/shared semantics for gradients, indexing, reductions, factories, optimizers, DDP hooks, FSDP2 tied parameters/checkpoint state, and distributed object collectives. The `priority-optimizations.md` reference was included. CPU RNG and unrelated MPS/ROCm changes were excluded. Upstream history and per-branch commit differences were checked before selection; no equivalent implementation was present in the fetched official base.

## Verification

CUDA verification ran serially in Slurm job `111576` and follow-up jobs `111577`/`111578`, using Python 3.11.15, CUDA 12.2.140, SM89 RTX 4090, and an isolated Jittor cache under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261021-integration-cuda/`. No timings are treated as acceptance evidence.

The CUDA probe exercised cross-entropy with CUDA logits and host labels, `eye(out)`, `fill_diagonal` gradients, `fliplr` gradients, `triu` index assignment, float16 logical reductions, out-of-bounds CUDA slicing and zero gradients, second derivatives through `backward(create_graph=True)`, and CUDA parameter updates with StepLR. All reported `use_cuda=true`; fallback delta was zero. A follow-up probe covered iterable batching, map-style `batch_size=None`, and `torch.jit.annotate` with CUDA operations, also with zero fallback. The new CUDA iterable DataLoader regression passed (1 test).

Artifact storage tests passed (10 tests). The lazy Triton driver test passed (1 test). Affected structural/package/API checks passed (59 tests, 50 subtests). `bash tools/check_repo_layout.sh` and `git diff --check` passed.

The broad `tests/structure` run produced 1513 passed, 6 skipped, 1066 passed subtests, and one failure: `tests/structure/backends/rocm/test_rocm_library_provider.py::test_specs_resolve_only_native_source_and_quote_paths`. That assertion rejects any source path containing `cuda`; the checkout parent directory is named `ms-swift-cuda`, so it classifies the checkout path as a CUDA source. This is a path-dependent test false positive and remains unresolved here because changing the unrelated ROCm structure test is outside this integration scope. The affected structural subset listed above passed.

The first iterable DataLoader probe exposed a real bug: iterable batches ignored `batch_size` and `drop_last`. The implementation was corrected to accumulate and emit batches, and the CUDA regression plus follow-up CUDA probe passed.

Previously recorded ms-swift inference and two-card training/FSDP2 experiments were not repeated. Their claims remain limited to their original source revisions and evidence: `docs/results/2026-10-19-ms-swift-cuda-inference-axis.md`, `docs/results/2026-10-20-swift-two-rank-sft-lora-resume.md`, and the FSDP2 evidence captured at source `4c881347a6385f9b88f50768e48f9221fb781f12` under the corresponding untracked lab state. They are not promoted to evidence for broader model/device coverage.

## Remaining scope

No CPU adaptation, CPU tensor validation, timing/performance acceptance, full ms-swift matrix, or new inference/two-card run is claimed. CUDA validation is limited to the cases above and reused evidence remains bound to its historical source SHA. Raw logs and build caches remain outside the repository in the lab state directory.
