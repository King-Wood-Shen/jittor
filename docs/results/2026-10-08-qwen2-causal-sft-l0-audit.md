# Qwen2-0.5B 公开 SFT：单卡初始参数 CUDA 审计

- 状态：真实公开 SFT 的单步构造与 290 个 trainable 参数初始值审计完成；本配置 L0 partial，后续层未由此运行验收。
- 日期：2026-10-08。
- 基线：Jittor `facd79899291744a141f879ca957fe4c72251de7`（代码合并基线 `1ba2eafb827ab4a4bf36351956bf0611e512bb83`，上游 `2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B、公开 `python -m swift.cli.main sft`、单 RTX 4090、FP32/eager、全参数 SGD、batch 4、固定数据、一步、最大长度 64。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补齐参数名/状态键映射、输入和输出设备清单、独立 logits/hidden 对拍；Torch shim、Jittor CUDA 执行器或 Swift Trainer 改动时重验。

Slurm 13040 先运行原生 PyTorch oracle，13043 随后运行严格 shim 候选，均在 `cscg-qh04` 的 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`），环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6。模型 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，数据 SHA256 为 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。两侧都是同种子 `1234`、同数据顺序和公开 CLI。13043 首次预热 Jittor 扩展后才启动候选训练；候选 scope 强制 `use_cuda=1` 与 `backend_fallback='error'`。

在首次 `SGD.step` 前，worker 对每个 optimizer 参数记录 group/index、shape、dtype、device、`requires_grad` 和连续初始值 SHA256。原生与候选的 290 项、共 494,032,768 个元素逐项元数据和 SHA256 全相同；全部参数都为 FP32、`cuda:0` 且 trainable。shim 进程启动与结束均记录 `use_cuda=1`、fallback `0`。两侧均完成 step-1 checkpoint；日志 loss 为原生 `3.82736707`、shim `3.82736802`，仅作为该单步运行日志，不作独立前向张量验收。

该 hook 使用 optimizer group/index，没有输出模型参数名或 state-dict key 对照；也没有收集模型输入/输出的设备清单或逐层前向张量。因此 L0 记 partial，L1 未运行。已有三步梯度与前向报告绑定各自运行键和基线，本报告不把它们合并为本配置 L1/L2 通过。长 JIT 首次构建耗时不属于性能测量。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 真实公开 CLI 模型、数据、trainer 与 optimizer 已构造；290 个 optimizer 参数的初始值、shape、dtype、CUDA device、trainable 状态相同。state-dict 参数名/状态键映射未采集。 |
| L1 | not-run | 未做单独同权重同输入 logits、hidden-state 与 loss 张量比较。 |
| L2 | blocked | 仅一步构造审计；没有三步完整梯度、optimizer 状态及逐步更新对拍。 |
| L3 | blocked | 未验证 checkpoint 恢复、RNG 或数据游标。 |
| L4 | blocked | 公开 CLI 完成一步，但前置层未完整验收。 |
| L5 | blocked | 前置层未通过；没有预热后的十次同步稳态性能协议。 |

原始参数清单、哈希、脚本、checkpoint、缓存和日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-l0-audit-v3/`。13039 的原生尝试因深层 `TMPDIR` 导致 AF_UNIX socket 路径过长，在数据预处理时退出；13040 原生成功，随后一次候选启动遇到 Jittor `jit_utils` 重建退出码 3。13043 先预热扩展，再复用 13040 原生记录完成候选；这些 harness 修正不计作兼容失败。该固定配置不代表整个 ms-swift。

文档门禁 Slurm 13052 首轮因本报告缺少 results toctree 条目而出现唯一失败（其余结构测试 `1385 passed, 6 skipped, 1027 subtests passed`）；补入导航后，Slurm 13053 定向可达性测试 `1 passed`，manifest 检查及布局检查通过。完整首轮门禁和修正后日志均保留在上述运行目录。
