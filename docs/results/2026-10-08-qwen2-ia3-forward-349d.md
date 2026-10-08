# Qwen2-0.5B IA3：Transformers/PEFT 直接前向张量级复验

- 状态：固定 IA3 模型配置的 L0/L1 通过；ms-swift CLI、恢复和性能层仍未验收。
- 日期：2026-10-08。
- Jittor 基线：`349d38732b9f826f250a616deeabc5cbe88860f0`（合入 `origin/2.0-refactor` 的 `25700b208fe58680169312e214c9ee82f125a094`）。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`；此项使用 Transformers/PEFT API，不调用 ms-swift CLI。
- 运行环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1；Qwen2-0.5B FP32/eager，PEFT IA3 adapter，batch 2，`Hello` 与 `What is 1+1?`，序列长度 7。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch shim、Transformers/PEFT 模型加载、Qwen2 前向或 IA3 注入变更；补做公开 `swift infer` 同基线验证时。

原生 PyTorch 与严格 shim 先后在 Slurm 12600 同一 cscg-qh06 RTX 4090（UUID `GPU-e0c8764d-4b51-62e5-f0d3-4aa27a8ddb68`）上运行，使用相同 Qwen2 `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad` 和 IA3 `adapter_model.safetensors` SHA256 `e7de90795307c4ff2d8d928d5fa43c11bd6ff07174c174d9bb549ff9b85e4348`。独立设备元数据补采 Slurm 12602 复用已完成的隔离 JIT 缓存；两次均为单进程单卡，不涉及 rank。

两侧均构造 362 个参数。比较器逐键核对参数名、shape、dtype 与 `requires_grad` 一致；补采记录确认原生与候选的全部参数、输入 IDs、attention mask、logits 和 25 组 hidden states 均为 `cuda:0`。所有捕获值均有限，输入与输出 shape 一致。Logits 最大绝对差为 `4.67301e-5`、相对 L2 为 `8.32954e-7`；25 组 hidden states 中最大绝对差为 `2.44141e-4`（第 23 层），最大相对 L2 为 `2.12129e-6`（第 23 层）。候选通过 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`，运行前后设备配置为 CUDA，累计 `fallback_count=0`。

独立脚本、参数清单、NPZ、日志和比较器未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-ia3-l0l1-349d3874/` 与 `20261008-qwen2-ia3-device-349d3874/`。首次 setup 作业 12598 因无条件 `sitecustomize` 导入使 Jittor 编译子进程递归初始化而被停止；它未进入候选模型计算，不计为模型兼容性失败。修正后作业 12600、12602 完成。既有 Slurm 11352/11355/11357 的公开 CLI 证据来自 `ffeb7bd` 基线，不能升级为当前同步基线的 L4 证据。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | PASS | 原生与 shim 的 362 个参数键、shape、dtype、`requires_grad` 一致；两侧全部参数在 CUDA。 |
| L1 | PASS | 同模型、同 adapter、同输入；完整 logits 与 25 层 hidden states 的设备、shape、有限值及误差已核验。 |
| L2 | not-applicable | 此项是推理前向，不做反向或更新。 |
| L3 | not-run | 未验证 adapter/模型状态保存与新进程恢复。 |
| L4 | not-run | 未在当前同步基线上调用 ms-swift 的公开 CLI、Python pipeline 或 launcher。旧基线 CLI 结果不继承。 |
| L5 | blocked | L4 未通过；未运行稳态性能协议。 |
