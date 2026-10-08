# Qwen2-0.5B ms-swift 私有 Swift LoRA 固定状态前向

- 状态：当前基线下直接构造 Swift 私有 LoRA 的 L1 固定配置前向通过数值门槛；L0 partial；L2-L5 未通过/未运行。本结论不代表公开 `swift sft`、Swift LoRA 训练或整个 ms-swift 兼容。
- 日期：2026-10-08。
- Jittor 基线：集成 HEAD `28f227e563daca3fc08bebbf814154c2e15c95b3`；`origin/2.0-refactor` 已刷新并确认为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，与当前 HEAD 的基线祖先一致。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：进入公开 Trainer/CLI 三步训练、恢复或性能验证时，或 ms-swift tuner/compat 实现变化时。

运行键 `20261008-qwen2-swiftlora-forward-v2`。Slurm 13196 在 cscg-qh06 的 RTX 4090（UUID `GPU-c7472e09-955b-7640-6cfd-b308890d9721`，driver `580.178.04`）上依次执行独立原生 PyTorch CUDA 进程和严格 shim 进程。模型为本地 Qwen2-0.5B，FP32/eager；使用当前 ms-swift 的 `swift.tuners.LoRAConfig` 与 `Swift.prepare_model`，rank 8、alpha 32、target `q_proj/v_proj`、dropout 0。两条固定提示经原生 tokenizer 生成同一 `[2,7]` input IDs 和 attention mask。原生构造的 96 个 trainable adapter 张量按键保存，再逐键复制到 shim 模型；双方完整参数均为 386 个，名称键相同，adapter 初态由拷贝保证一致。

模型参数、adapter 参数和输入均在 cuda:0。输出 logits shape `[2,7,151936]`，25 个 hidden state shape 均为 `[2,7,896]`，全部有限。Slurm 13200 在 GPU worker 对保存结果比较：logits 最大绝对差 `6.96182e-5`、相对 L2 `2.88510e-6`；hidden 最大绝对差 `3.58582e-4`（h24）、最大相对 L2 `2.66048e-6`（h23）。候选从导入到保存 `use_cuda=1`，`fallback_count=0`。原始脚本、日志、NPZ、元数据与逐张量比较位于未版本化目录 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-swiftlora-forward-v2/`。

同一探针的首版运行键 `20261008-qwen2-swiftlora-forward-v1`（Slurm 13194）在原生模型前向时将 tokenizer 的 CPU 张量传给 CUDA 模型，触发设备不匹配；shim 未运行。该探针错误已在 v2 显式将输入置于 cuda:0 后修正，不构成运行时兼容性故障。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 原生与 shim 均构造实际 Qwen2 模型、tokenizer 和 Swift 私有 LoRA；386 参数键、dtype/device 与 96 个 trainable 参数键一致。未构造 Swift template、dataset、optimizer、trainer，故不满足完整 L0。 |
| L1 | PASS（固定配置直接 API） | 同一 base checkpoint、显式相同 adapter 状态和 tokenizer 输入；logits 与全部 25 层 hidden 的结构、有限值和误差均通过门槛。不是公开 CLI/Trainer 结果。 |
| L2 | not-run | 无 backward、optimizer 或更新。 |
| L3 | not-run | 无 checkpoint 恢复。 |
| L4 | not-run | 未运行 `swift sft` 或其他公开 ms-swift 入口。 |
| L5 | blocked | 前序 L0、L2-L4 未通过；无性能协议。 |
