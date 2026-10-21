# ms-swift / JTorch CUDA integration

**Status:** the source-scoped CUDA mechanisms and public reentrant two-rank
profile below are verified; the complete ms-swift adaptation matrix remains
unfinished. **Date:** 2026-10-22. **Official base:**
`67cf9af564cdaf378d6292fa856714ad3de42628` (`origin/2.0-refactor`).
**Source PR node:** `66c62ed4a0a4603bf6126332c09c341f93fc1d36`.
**Maintainer:** ms-swift CUDA/JTorch integration. **Review when:** the official
base or a recorded gradient, storage, checkpoint, device or RNG contract changes.

## Source and scope

Repairs belong to Jittor core, Torch compatibility and the CUDA backend. They
cover exported allocation lifetime, asynchronous fetch completion on a nondefault
GPU, legacy RNG decoding and device-correct random_, half extrema, stable SDPA
gradients, generic module state hooks, real reentrant checkpointing and the
standard FSDP checkpoint facade. No Swift/TRL source was patched and no
algorithm-specific workaround was added. Worker functional units were selected
individually from committed revisions; dirty worker source was not copied.

The source PR also contains separately selected shared optimizer/storage/capture
repairs (`f562b73d3`, `d68b8973f`), CUDA level-zero launch width (`d79938e54`)
and pending-producer full-reduction fusion (`c56b4c2bc`). Each functional unit
has its own commit and retained validation scope. The earlier `15d4455f8`
metadata-only attention classification change preserves strict test bodies,
numerical budgets and NPU entries. Literal CPU-versus-CUDA numerical cases were
not rerun or claimed as passing.

CUDA execution and compilation ran on Slurm compute nodes with RTX 4090 / SM89,
Python 3.11.15 and CUDA 12.2.140. First compilation was serialized, rank caches
were separate, source/ABI snapshots were retained, and cache configuration
fingerprints were recorded. No timing or throughput is an acceptance criterion.

## Independent mechanism validation

| Scope | Source and result | Actual allocated peak |
|---|---|---|
| Maintained CUDA core, two GPUs | `7e0551708`, Slurm 113667: native134 + Torch225 = 359 passed, 7 skipped, 2 xfailed; original multi-graph gradient sequence and storage guards pass | Mechanism/model memory evidence is recorded separately below |
| Attention, scalar casts and stack | `7e0551708`, 113600: 30 passed, 170 subtests; zero fallback | 84934656 bytes since the suite's last explicit reset, not a whole-suite peak |
| Original small-gradient SDPA samples | `7e0551708`: native5 + public Torch5 satisfy unchanged analytic net/per-element budgets, exact 2D generic / 4D fused routes, retained views and clean subprocess exit | 10752 / 11264 bytes; zero fallback |
| Checkpoint scheduling, reentrant contracts and FSDP facade | `7e0551708`, 113667: 30 direct passed, no skips; includes captured parameters, mutation guards, AMP, state hooks and real updates | 18944 bytes; zero fallback |
| Checkpoint with causal/dropout SDPA | `7e0551708`: two SGD-momentum updates change 192 elements; forward calls 2 off / 4 on; output, input/parameter gradients, momentum and subsequent RNG match | 13312 bytes; zero fallback |
| Two-rank DDP/FSDP persistence | `7e0551708`, 113667: both ranks restore model/SGD state and CPU/CUDA/Python/NumPy RNG after effective perturbation, then complete steps 2-to-4; loss 0.0510→0.0408; DDP 20 or FSDP 10 local elements updated per rank | 7168–11264 bytes across the labelled cases; device-wide NVML samples also retained |

At `d68b8973f`, independent113974 passes the divergent-input two-rank oracle:
checkpoint off/on each makes three SGD-momentum updates; global weight and local
input gradients, momentum, updated parameters, hooks and retained aliases/pointers
match an independent analytic reference. Actual allocated peak is7168 bytes per
rank, fallback0. The unchanged baseline15 fails the alias/pointer check at step0
in113944 after its gradient and numeric update checks pass. New storage/capture9,
checkpoint30, dropout-SDPA two-update interaction and two-GPU core359 pass.

The initial view/optimizer harness accidentally ran explicit CPU parameterizations.
Raw334/101 mixed passing counts and the selection error remain visible;
collection-only113989 confirms170/48 passing CUDA nodes, with no test rerun.
Their mixed-process allocator peaks are not presented as CUDA-only subset peaks.

At `c56b4c2bc`, independent114001 passes19 CUDA launch/reduction methods and18
subtests, plus one separately labelled host dispatch-decline metadata check.
The CPU reduction and three CPU-classified legacy codegen nodes are deselected.
The original allocation upper bound, unchanged analytic gradient budgets and
repeatability checks pass. Fallback0; native process allocated peak67158016 bytes
is the suite high-water, not the optimized broadcast case's isolated peak.
Storage/capture9, CUDAoptimizer48, checkpoint30, dropout-SDPA two real updates and
single-GPU core356 pass;10 policy skips and2 pre-existing xfails remain.
Worker isolated37376-byte allocation evidence keeps its original worker SHA.

The FSDP hook reference was checked with genuine Torch2.6 CUDA: sharded
parameter hooks are absent, while module pre/post and input hooks fire. The
independent checkpoint comparison requires six of each module/input hook and
real local gradients; an earlier incorrect parameter-hook expectation is kept
in raw evidence, not hidden by a tolerance change.

Half extrema evidence remains bound to `67df86f3a`: five native methods plus
four subtests, public float16/bfloat16 extrema/dtype/backward and checkpoint AMP
checks pass, with zero fallback and measured peaks 2048/5632 bytes. RNG evidence
remains bound to `8820c2a11`: six CUDA-ambient methods and seven legacy subtests
pass with zero fallback and peak 3072 bytes. Their historical results are not
relabelled as a new full-matrix run.

## Official sync and object-scoped execution continuation

The official `67cf9af56` merge is independently checked at `8af2342ee` in
Slurm114148: 21 official CUDA methods and44subtests, class predicates across four
floating dtypes, public any/all across seven dtypes and retained integer truth
all pass. Private16, GPU1 Copy0 fetch, storage9, optimizer48, checkpoint30,
dropout-SDPA/CUB real SGD and all six two-rank DDP/FSDP hooks/persistence phases
pass with zero fallback and actual native/NVML peaks. Two-GPU core359 passes.
The original reentrant model profile remains evidence at `15d4455f8`.

Private materialization prerequisites selected at `af60eb185` have16 direct
contracts; their union with checkpoint30 covers the worker's41 plus five FSDP
contracts. An initial count observer incorrectly expected41 from the16-case
file; raw failure is retained and no passing tests were repeated. A separate
real public sum/CUB optimizer interaction makes two updates to66049 elements.
These prerequisites do not implement defaultFalse checkpointing.

At `7617793b3`, Torch serialization snapshots execute only the saved object's
tensor dependencies and drain pending fetch copies without adding global roots.
The unchanged CUDA counterfactual fails both metadata and good-tensor saves at
`8af2342ee` (114246), then passes at the repair (114273). Five direct guards
(114274) preserve GPU0/1 residency, float64/int64 and lazy/nested values; explicit
failed tensors and their real dependencies still raise. Actual parent peaks
are2560bytes per GPU, with2048bytes in the isolated failure child, fallback0.
Checkpoint30, dropout-SDPA SGD, core359 and all six DDP/FSDP distributed phases
(114275) pass with allocator/NVML evidence. The original wrong top-level pytest
path and omitted experiment helper failures are retained; only unexecuted
stages were corrected and run. No source/tolerance adjustment was made.

`66c62ed4a` selects only worker7f1's two optional-root failure filters: unrelated
failed producers are excluded from implicit weak sync/auto_flush, while explicit
roots, real dependencies and sync_all retain their failures. Its independent
scope is51 direct CUDA contracts (private16, scheduling11, activation16, FSDP5,
serialization3), single-GPU core356/10skips/2pre-existingxfails, and measured
native peaks. Slurm114281/114283 final audit passes; actual parent peak18944bytes and the two
implicit-failure child peaks2048bytes each, zero fallback. An incorrect count-hook
argument prevented collection in114281; only that never-executed direct group was
run after the harness correction. Original failure logs remain available.

All recent structure gates retain the same pre-existing ROCm path assertion:
1516passed,6skipped,1066subtests,1failure. Layout/manifest/diff checks pass.
Historical eleven-mode/variant results are not relabelled as a new-SHA full pass.
Compiled external-backward admission, live hidden trainability and gradient
materialization side effects remain algorithms-owned. In retained af60 CUDA
probes114091/114102, numeric reads of published gradients execute extra optimizer
updates with no intervening step call; getter-only reads and eager controls do
not. Actual peaks18432/11264bytes, fallback0; failures remain open. DefaultFalse
owner/lease/RNG/frontend implementation remains options-owned.

## Public model checkpoint and recovery

At `15d4455f8`, the unmodified Swift worktree
`88d727951203256baa564c643c651b6f8d90fd7e` ran Qwen2-0.5B LoRA, FP32 SDPA,
two ranks, per-device batch 2, AdamW, seed 731 and explicit
`gradient_checkpointing_kwargs={"use_reentrant":true}`. The prefix trains two
steps and saves checkpoint2; fresh processes load it and continue to step4.
The matched off branch starts fresh and trains two steps with the same model,
data, precision, optimizer and auditor.

The four reported losses are 0.23444393, 0.14808717, 0.07887644 and 0.03934895.
Every rank consumes two batches and makes two finite, nonzero parameter updates
in each prefix/resume/off phase. Resume gradient audits see global steps 2 and 3.
The on phases record 48 no-grad plus 48 grad decoder calls per rank; off records
48 grad calls and no no-grad decoder calls. Fallback increment is zero.

Allocator peaks are reset once, at the same first-training-step boundary in
every actual rank process; live allocations remain the baseline. Whole-process,
device-wide NVML sampling is recorded separately without resetting it.

| Memory metric | Checkpoint off | Checkpoint on, prefix2 |
|---|---:|---:|
| Allocated peak, each rank | 6266783232 bytes | 3789196800 bytes |
| Reserved peak, rank 0 / rank 1 | 6547308544 / 6529482752 bytes | 3948937216 / 3931111424 bytes |
| Sampled whole-device NVML, same two GPU UUIDs | 8082 / 7239 MiB | 5604 / 4761 MiB |

The measured training allocator peak falls 39.535% on each rank. These are real
measured counters and samples, not a capacity estimate or a throughput result.

Slurm 113773 completed every training child stage, but its parent summary failed
because the monitor selector omitted the new off stage. Original outputs,
checkpoints and allocator peaks remain intact. Only that incomplete off branch
was measured again in 113847 (COMPLETED, exit 0), with both allocator and NVML evidence;
its allocator peaks exactly match the original off run. The final independent
audit binds prefix/resume to 113773 and completed off to 113847, rather than
pretending the parent job exited successfully or mixing measurements from
unlabelled executions. Uniform peak-reset markers, real updates and all rank
exits were checked. No accepted DPO or prior on/resume experiment was rerun.

A separate genuine Torch2.6.0+cu124 CUDA public Swift first-step diagnostic,
113831 (COMPLETED, exit 0), uses exactly the same 2×512 int64 input bytes. Its loss is
0.23444396 versus current JTorch 0.2344439328, with no Jittor import and a real
allocated peak 3954747904 bytes. This supports the current loss scale; it is not
a requirement for strict stepwise native-training parity or a cross-framework
memory comparison. Historical official-base loss differences are not attributed
to checkpointing without evidence.

Historical GKD new-process recovery at `a34679ee9`, Slurm23642, has a retained
unrounded public Swift `memory=4.1201171875`GiB. The committed chain from
`state.max_memory` through `torch.cuda.max_memory_reserved` to the native
per-device reserved high-water counter binds4423942144 bytes to the six
resumed updates2-to-8. This is reserved pool high-water observed during training;
its reset origin is not reconstructed and it is not live-allocated or NVML
occupancy. The same-source forced-RNG single-update probe23646 remains separate.
Source and log hashes are retained; no GKD training was repeated and no current
source result is inferred from this historical record.

## Gates and reproduction

At `7e0551708` and at the final metadata candidate, structure checks report
1516 passed, 6 skipped and 1066 passed subtests, with one pre-existing failure:
`tests/structure/backends/rocm/test_rocm_library_provider.py::test_specs_resolve_only_native_source_and_quote_paths`.
That assertion rejects an absolute source path containing `cuda`; this checkout
parent contains that word. Slurm113667,113760,113974 and114001 therefore retain exit1. The latter two
independently report the same1516/6/1066 structure counts and known failure.
Repository layout, generated manifests and diff checks pass. The unrelated ROCm
check was not modified or skipped, and no blanket green gate is claimed.

Within a real CUDA Slurm allocation, the maintained core entry is:

```bash
python tools/run_test_suite.py --tier core --backend cuda --serial-compile
```

The minimal checkpoint contracts are in
`compat/tests/torch/test_activation_checkpoint.py`,
`test_activation_rematerialization.py` and `test_fsdp_activation_checkpoint.py`.
Use their canonical `python/jittor/compat/tests/torch/` paths in Torch-mode pytest
with `JITTOR_TEST_DEVICES=cuda` and `JITTOR_TORCH_SHIM=1`; importing the top-level
`compat` package is not the same entry. Native attention, half and multi-device
regressions live in `tests/backends/cuda/` and `tests/ops/`.

Raw public argv, dataset/model/input hashes, sampler data, failed harness
observations, source afterimage bindings and build caches remain in unversioned
batch state `sol61-20261021`. They are not shipped in the source PR. Prior
inference, DPO and other training-mode conclusions keep their original committed
source and acceptance scope; this report does not promote all historical modes
to the current PR node.

## Remaining contracts

`use_reentrant=False` remains unsupported. The public DDP default and FSDP's
NO_REENTRANT activation wrapper therefore remain open paths. Native bulk
materialization, region ownership, callback/lifetime isolation, per-producer RNG
and FSDP/Python-hook behavior need separate verification. A warning, direct call
or a True-mode result is not False-mode support.

Compiled forward callables needing external backward also remain open under
KI-COMPAT-018. At `c56b4c2bc`, independent114011 proves that no-grad capture
replays detached output after grad mode is enabled: input/parameter gradients
are None, fallback0 and actual allocated peak3072 bytes. The expected FAILED1
and initial import-only harness failure114008 are retained. Algorithms owns the
shared capture-admission/grad-context repair; successful codegen or checkpoint
contracts do not close this defect.

The complete eleven-mode training and variant matrix remains unfinished. CPU
training, other accelerator backends and new inference coverage are not claimed.
Current defects and review conditions are indexed in
[known issues](../../agent/manuals/known-issues.md), including KI-COMPAT-016,
KI-COMPAT-017, KI-COMPAT-018 and KI-EXEC-016.
