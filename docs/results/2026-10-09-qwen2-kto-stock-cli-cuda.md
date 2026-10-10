# Qwen2-0.5B stock `swift rlhf --rlhf_type kto` CUDA 入口审计

- 状态：原生与 strict shim 均通过公开 KTO CLI 完成固定配置三步训练并保存 checkpoint；固定配置 L1 前向与 L2 三步全参数梯度/更新对拍通过。L0 partial，L3 未运行，L4 入口已运行但整体功能合同未通过，L5 blocked；不代表其他偏好算法或 ms-swift 整体兼容。
- 日期：2026-10-10 复核。
- 基线：Jittor `35963c2e047dfa81751f82e8127064adf865d6bd`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，公开单卡 `swift rlhf --rlhf_type kto`，固定二条 preference 数据（chosen/rejected）、batch 2、SGD、三步；早期单步调用栈与完整 logits 证据仍保留。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补 optimizer/scheduler/RNG/dataloader 状态恢复后，重新按 L0–L5 逐层验收；若调整自动 graph replay 的首次 capture 行为，复核该 CUDA 训练调用路径；Torch shim、Transformers、TRL 或 ms-swift KTO 入口变更时重跑。

模型权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；固定 preference 数据 SHA256 为 `05640c5363ecd9bf8162504ffee786a21f4c92a2bcd3fd64403a2269dd53baec`。运行环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0，ms-swift 自报版本 `4.6.0.dev0`。

Slurm 13834 在 `cscg-qh04` RTX 4090 上预检了 ms-swift 模块来源、CUDA 可用性和公开 RLHF CLI 帮助。Slurm 13836 随后在同一设备运行 native，再运行 shim。native 和 shim 的完整 290 项参数初态名称、shape、dtype、trainable 状态和值 hash 一致，均位于 `cuda:0`；策略与 reference 两次 KTO forward 所用的 CUDA 输入键、张量值及标签 `[false, true]` 完全相同。strict shim 的父、子 CLI 进程均记录 `use_cuda=1`、`shim=True` 和 `fallback=0`。

单步 loss 分别为 `0.5` 与 `0.5000002980232239`。比较器覆盖 290 个参数、494,032,768 个梯度元素，梯度最大绝对差 `5.14090e-7`、相对 L2 `7.45108e-6`。checkpoint 包含相同的 290 个权重键；168 个张量有更新，末态最大绝对差 `7.45058e-9`、相对 L2 `3.92164e-11`。KTO forward 返回的 chosen/rejected log-prob、logit-sum 和 per-example KL log-prob 聚合值均已比较；这些值的相对 L2 约为 `2.44e-7` 至 `1.59e-6`。chosen/rejected logit-sum 的最大绝对差分别为 `0.875–1.125`。补充逐模型调用采集后，两个分支共同的前两次模型调用输入逐值相等，全部 25 层 hidden states 与完整 logits（形状 `[2, 29, 151936]`）均有限；最大相对 L2 `3.59102e-6`、最大绝对差 `3.29971e-4`。逐 token 证据说明 logit-sum 绝对差来自大规模求和累积，单个 logits 的最大绝对差为 `8.77e-5`。

运行键 `20261009-qwen2-kto-stock-cli-v3` 保存单步脚本、模型/数据摘要、输入与 forward 记录、checkpoint、比较 JSON 及原始日志，均未版本化。三步梯度/更新补验运行键 `20261010-qwen2-kto-stock-cli-l2-three-step-v2` / Slurm 15311 完成 native 与 strict shim 两条公开 CLI 轨迹。三步中 policy 与 reference 输入、标签和元数据逐项相同；每步覆盖 290 项参数、494,032,768 个元素，梯度最大绝对差分别不超过 `7.68e-7`，相对 L2 分别不超过 `9.70e-6`；更新最大绝对差每步为 `7.45e-9`，相对 L2 分别为 `8.78e-5`、`8.84e-5`、`8.37e-5`。三步 loss 最大差 `2.68221e-7`；step-3 checkpoint 的 290 个权重键均相同，171 个张量更新，最大绝对差 `7.45058e-9`、相对 L2 `7.28111e-11`。两组 SGD 公共选项相同（参数数 169/121，weight decay 0.1/0），每步状态项均为空；native Torch archive 与 shim 便携 pickle 经各自安全读取器解码后，checkpoint optimizer state dict 内容相等。shim 父子进程均记录 CUDA 与 `fallback=0`。审计键 v1/v2 的读取器与格式假设失败已保留；最终 GPU worker 只读审计键 `20261010-qwen2-kto-stock-cli-l2-three-step-artifact-audit-v3` / job 15317 通过全部三步、状态、末态权重及 fallback 断言。作业 15310 的 harness 错误未进入 shim，不属于兼容性结果。单步作业 13831/13835 分别因 PYTHONPATH 与过长 TMPDIR socket 路径退出；修正后的 v3 完成。

逐 token 补充证据保存在未版本化 state：native 模型输出运行键 `20261010-qwen2-kto-logits-v1` / Slurm 15259，strict shim 输出运行键 `20261010-qwen2-kto-logits-v3` / Slurm 15262，比较作业 Slurm 15266。15259 的 native 训练、采集和 checkpoint 均完成，之后因 harness 文件名断言错误退出；15261 的 shim 训练与采集完成，但遗漏 strict CUDA bootstrap，故不作为 strict 证据；新键 15262 补跑 shim 并通过 strict 启动检查，父/子进程均为 `cuda=1`、`shim=True`、`fallback=0`。同设备为 cscg-qh04 RTX 4090，native/shim 使用相同固定数据、模型权重、源码和 CLI 参数。比较器只对齐双方共同的前两次 reference 调用；shim 多出的第三次调用留作未解释差异。

### Reference 首次 graph replay 补查

新调用栈运行键 `20261010-qwen2-kto-extra-ref-call-trace-v1` / Slurm 15300 在相同单卡公开 CLI 配置下，先 native 后 strict shim，确认两次 `KTOTrainer.forward` 分别收到 `self.model` 和 `self.ref_model`，两侧分支身份相同。native policy/reference 底层模型 forward 均为 2 次；strict shim policy 为 2 次、reference 为 3 次。第三次来自 reference 分支 no-grad completion call 的首次 `GraphReplay`：`jittor/_core/module.py` 进入 `_runtime/graph_replay.py::__call__` 与 `_capture_now`，其中 eager warm-up 与 capture 都执行模型。Jittor 默认 `auto_graph_replay=1`，该机制用于满足重复签名/输入阈值的 no-grad 模块调用；它增加一次初始化计算，不代表 KTO 多执行了一个 reference 分支。

因果诊断运行键 `20261010-qwen2-kto-extra-ref-call-trace-noautoreplay-v1` / job 15303 只关闭严格 CUDA shim 进程的 `jt.flags.auto_graph_replay`，两分支底层 forward 数均恢复为 2；模型仍由公开 CLI 训练并保存 checkpoint，父子进程 CUDA=1、fallback=0。GPU-worker artifact audit job 15306 核对原 native 与 strict-shim reference 输入完全相同；KL 与 completion 的全部 27 个输出张量形状/dtype 对齐且有限，最大 relative L2 `3.59102e-6`、最大 absolute error `3.29972e-4`。strict shim completion 的 eager warm-up 与 graph-capture 返回张量逐元素完全相等；capture 返回与 native completion 输出亦在同一容差内。故固定 Qwen2-0.5B、FP32/eager、batch-2、一步的 KTO L1 前向合同通过；自动 replay 首次调用多一次内部计算作为性能边界保留，不将其推广为所有模块或场景结论。原始 trace/NPZ/worker 证据留在各运行键目录。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 真实模型、tokenizer/template、数据、KTO trainer 与 optimizer 经公开入口构造；290 项初态参数及三步两种输入分支逐项核对。完整 state/buffer 与 tokenizer/template 状态映射未审计。 |
| L1 | pass | 两侧 reference 输入逐项一致；KL 与 completion 的完整 logits、25 层 hidden states 均有限且达容差。strict shim 第一次 no-grad completion 触发自动 GraphReplay eager warm-up/capture；两输出逐元素相同，capture 返回与 native completion 的最大相对 L2 `3.59102e-6`、最大绝对差 `3.29972e-4`。仅限本报告固定配置。 |
| L2 | pass | 固定二条 preference 数据、batch 2、FP32/eager、SGD 的公开单卡 KTO CLI 三步轨迹。每步 290 项全参数梯度与更新、两路输入和 loss 均对齐；三步 loss 最大差 `2.68221e-7`，末态权重最大差 `7.45058e-9`。optimizer 两组公共选项、空 state 和 checkpoint 解码内容相等；严格 shim CUDA 父子进程 fallback=0。仅限本报告配置。 |
| L3 | not-run | 未验证同进程/新进程 checkpoint 恢复、scheduler/RNG/dataloader 游标与后续轨迹。 |
| L4 | 入口已运行，功能未通过 | native 与 strict shim 均通过 stock `swift rlhf --rlhf_type kto` 完成三步并保存 checkpoint；L0 状态映射、L3 恢复和 L5 性能合同尚未通过，不能标 L4 通过。 |
| L5 | blocked | 前置层未通过；未执行预热后至少 10 次同步稳态性能与显存协议。 |

Slurm 13856 的原始报告门禁通过；报告更新门禁 15269、15307、15319 的布局检查均通过，Torch-mode `tests/structure` 各为 1384 passed、8 skipped、1019 subtests passed。三个 worker 均没有 nvcc，因此 Jittor 结构门禁部分运行于 CPU；这只验证布局和结构，不构成 CUDA 兼容性证据。旧 v1/v2 的环境错误及其复验记录在 state，不影响门禁结论。Slurm 13838 与 13843 的首次原始门禁仅因根 `MANIFEST.in` 未列入报告及新增行排序不符而失败，修正清单后的 13856 全部通过。该报告只覆盖上列单卡 Qwen2-0.5B KTO 配置。
