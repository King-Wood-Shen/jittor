# Fused AdamW mixed-dtype CUDA validation

This record covers the installed `fa124604ffbca11ec2236b647d0725621d5b1298` Jittor primitive.

- Parent/step: `188171.44`, `cscg-qh13`, one GPU, eight CPUs, 40 GB; Slurm `COMPLETED 0:0` in 3 seconds; pool PID `3696484`.
- Approved request SHA-256: `8732efe557d0e0676c3284077a767cdd13ddcee589c025c371b86577f2fab728`.
- Source test: `tests/optim/test_fused_adamw_device_contract.py`; Git blob `4607a6fc1cade2d714a98c7e222614a8828d9f28`; SHA-256 `119ecc81d47b1e080773c734b33f11a82a8c482e8fbdf4fb36eb18a723337e00`.
- Exact node: `test_fused_adamw_device_contract.py::test_fused_adamw_double_bias_correction_and_physical_writes`.
- JUnit: three passed (FP32, FP16, BF16), zero failures/errors/skips. Pytest, final gate and overall step exited zero; fallback counter remained zero.
- The test asserts three updates per dtype against independent golden values: FP32 `rtol=1e-6, atol=1e-7`, FP16/BF16 exact equality. It checks physical CUDA storage, pointer ordinal, dtype and pointer identity. Inputs are deterministic inline tensors; no model or dataset is used.
- Installed runtime: Python 3.11.15; Jittor 2.0.2 from `fa124604`; jittor-compat 2.0.2 from `f82913e38fbf0487e64a898449a75e1021e520c4`; jittor-ms-swift 2.0.0.dev1 from the local `formal-downstream` checkout; ms-swift 4.5.2 from `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`; NumPy 1.26.4; pytest 7.4.4.
- The hot JITTOR_HOME full-root baseline matched before import: 1113 entries/1093 files, tree SHA-256 `23ada69eeb60461acceee679dc1707225f85d0842749b82e5fb41614334a5c1c`. After the run, only native `jittor.lock` content changed (517 to 522 bytes); the writer gate found no active holder. No cold-build slot was claimed.
- Original terminal receipt SHA-256: `acb14362052e618c4e5fd6c4737f372428a414ee7098e292242b8e91e0ac3866`; JUnit: `48a157f169dc02da762637cfb6a5a9ccc5729d67416537961b04943ae727920b`; pytest log: `80a07ca20d7b0885691b2687148ccdbea5fe52a49f77e0c23b5b3a73b2096d40`; complete run checksum list: `c90b8e118e267a3d1ed486f382437565b57ad662a9f58989116dd2453216e15b`.

The earlier `.42` runner failure remains preserved. This run does not cover merged commit `140ad5c52`, checkpoint or serialization behavior, Swift training, distributed execution, or complete current-main runtime acceptance.
