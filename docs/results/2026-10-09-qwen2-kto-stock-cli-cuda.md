# Qwen2-0.5B stock `swift rlhf --rlhf_type kto` CUDA 入口审计

- 状态：固定配置当前树 L0–L2、L4 通过；L3 未运行、L5 因 L3 前置范围未完成而 blocked。L0 清单与 L1/L2/L4 训练基线分别见下文。结论不代表其他偏好算法或 ms-swift 整体兼容。
- 日期：2026-10-12 复核。
- 历史基线：Jittor `35963c2e047dfa81751f82e8127064adf865d6bd`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；当前树的复验 SHA 单列于各小节；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，公开单卡 `swift rlhf --rlhf_type kto`，固定二条 preference 数据（chosen/rejected）、batch 2、SGD、三步；早期单步调用栈与完整 logits 证据仍保留。
- 维护者：ms-swift CUDA 适配。
- 复查条件：当前树的 L1/L2/L3/L4/L5 尚未逐层验收；恢复与完整公开训练流程需用新运行键继续。若调整自动 graph replay 的首次 capture 行为，复核该 CUDA 训练调用路径；Torch shim、Transformers、TRL 或 ms-swift KTO 入口变更时重跑受影响层。

模型权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；固定 preference 数据 SHA256 为 `05640c5363ecd9bf8162504ffee786a21f4c92a2bcd3fd64403a2269dd53baec`。运行环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0，ms-swift 自报版本 `4.6.0.dev0`。

### 当前树 L0 完整构造审计

新运行键 `20261012-qwen2-kto-current-tree-l0-inventory-v3` / Slurm 18419 在 `cscg-qh04` RTX 4090 上分别经 stock `swift rlhf --rlhf_type kto` 构造 native 与 strict shim 的真实模型、reference model、tokenizer/template、数据集、Trainer、SGD 和 LambdaLR，然后在任何 forward、backward、更新或 checkpoint 前停止。基线为 Jittor HEAD `2f0de6f8d51694bf1903a95cc7855d972132efd2`（`python/jittor` tree `b066dc9e303d2d3e7eb2c4bd4a824ab68657e835`，`compat` tree `558002060ff0d29791daa7402c2ecc6fcd5fa49e`）和 ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。两侧 policy/reference 各有 291 个 state 条目（290 参数、1 buffer）；名称、shape、dtype、CUDA device、元素数和值 SHA256 相同。policy 的 290 个参数均可训练，reference 的 290 个参数均冻结，两侧对应标志逐项一致；模型类/config、Trainer、数据首行摘要、tokenizer、template 类、collator 与本 profile 参数也一致。optimizer 按语义类比较为 SGD，两个有序组的参数名和次序、显式 `lr`/`initial_lr`/`weight_decay` 一致，构造后 state 均为空；scheduler 按语义类比较为 LambdaLR，完整 state 相等。

只读 worker 投影审计键 `20261012-qwen2-kto-current-tree-l0-projection-audit-v3` / Slurm 18423 对以上保存清单完成断言，结果 `KTO_CURRENT_TREE_L0_PROJECTION_PASS`。审计明确保留三项运行时表示差异：policy/reference 的非持久 RoPE buffer `model.rotary_emb.inv_freq` 原生 `requires_grad=false`、shim `true`，但它不属于 `named_parameters` 或 optimizer groups；native 与 shim 分别将 SGD/LambdaLR 以不同实现模块路径发布，Jittor SGD 的 group 还含内部 `values` 缓冲，故比较的是语义类、公共选项和状态而非完整 Python 类型路径/group 字典；native 报告 Torch `2.6.0+cu124`，shim 报告其 Torch API 兼容版本 `2.0.0`。RoPE buffer 标志差异也见同一兼容树下其他固定 profile，仍是显式兼容性边界，本项不据此声称训练期 buffer/autograd 等价。两侧 strict shim 父/子进程均为 CUDA、`fallback=0`；worker 核对的 `jittor_core` 与 `jit_utils_core` SHA 前后相同，未重编核心。初始严格比较作业 18419 因比较器将实现类路径当成必须相同而退出；只读审计 v1/18421 的额外“所有 policy 参数均需可训练”假设、v2/18422 的诊断字符串语法错误也均未产生新的兼容性结论。以上原始脚本、清单、日志和失败均保存在未版本化 state。

### 当前树 L1/L2/L4 固定配置复验

新运行键 `20261012-qwen2-kto-current-tree-l2-shim-v2` 使用当前 Jittor HEAD `da7c978f1af8dec93bed3f827ec1887e945487a6`（`python/jittor` tree `b066dc9e303d2d3e7eb2c4bd4a824ab68657e835`、`compat` tree `558002060ff0d29791daa7402c2ecc6fcd5fa49e`）和 ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。Slurm 18432 在 `cscg-qh04` RTX 4090 预检通过，确认 Python 3.11.15/CPython 3.11 ABI、g++ 11.5、CUDA 12.2、模型与数据 SHA、相同实现缓存指纹和独占锁；`JITTOR_NO_BUILD=1` 检查前后 `jittor_core`/`jit_utils_core` 哈希相同。训练作业 18439 在同节点另一张 RTX 4090（UUID `GPU-d813000b-1bfa-885e-cd61-0de97b365732`）通过实际公开 `python -m swift.cli.main rlhf --rlhf_type kto` 执行 strict-shim 三步全参数 FP32/eager profile，父/子进程均 `cuda=1`、`shim=True`、`fallback=0`，运行期核心缓存哈希不变，写出 step-3 checkpoint。复用原生公开 CLI oracle `20261010-qwen2-kto-stock-cli-l2-three-step-v2` / job 15311；其与本次运行的 `python/jittor`、模型、数据、ms-swift、Torch/Transformers/TRL 及参数一致，变化仅在 native 不加载的 compat 层，原始 oracle 只读复用未重跑。

原生与 shim 的 policy/reference 输入、标签逐步完全相同；forward 输出键、shape、dtype 与有限性满足合同，跨实现误差只记录诊断。全部 290 个参数的梯度均采集且有限；每步梯度跨实现相对 L2 分别为 `6.64e-6`、`8.33e-6`、`8.27e-6`。SGD 两个公开参数组语义、选项、参数次序和空状态相符，更新公式舍入 FP32 残差不超过 `7.46e-9`。每步 native/shim 非零更新元素分别约 `493,818,744/746`、`493,818,709/711`、`493,818,659/656`；47 个 shim 参数张量的逐张量更新为零（局部 FP32 舍入诊断），但每步整体更新均有限且非空。比较器初版曾错误地要求每个梯度/更新张量都非零，故 job 18439 如实以 harness assertion 退出；训练及产物已经完成。只读 posthoc 比较器 `compare-posthoc-v2.py` 按本 Skill 的“梯度存在且有限、整体更新非空”合同修正该断言，未重跑训练，并对相同原始产物通过。两侧三步 loss 分别为 native `0.50000000, 0.49984092, 0.49968153` 与 shim `0.50000030, 0.49984118, 0.49968162`；各自 step 1 到 step 2–3 均值下降。末态 checkpoint 的 290 个权重键匹配，171 个张量变化，native/shim 最大绝对差 `7.45e-9`、相对 L2 `7.05e-11`。据此当前树固定配置 L1、L2、L4 通过；不据此声称恢复或性能通过。

Slurm 13834 在 `cscg-qh04` RTX 4090 上预检了 ms-swift 模块来源、CUDA 可用性和公开 RLHF CLI 帮助。Slurm 13836 随后在同一设备运行 native，再运行 shim。native 和 shim 的完整 290 项参数初态名称、shape、dtype、trainable 状态和值 hash 一致，均位于 `cuda:0`；策略与 reference 两次 KTO forward 所用的 CUDA 输入键、张量值及标签 `[false, true]` 完全相同。strict shim 的父、子 CLI 进程均记录 `use_cuda=1`、`shim=True` 和 `fallback=0`。

单步 loss 分别为 `0.5` 与 `0.5000002980232239`。比较器覆盖 290 个参数、494,032,768 个梯度元素，梯度最大绝对差 `5.14090e-7`、相对 L2 `7.45108e-6`。checkpoint 包含相同的 290 个权重键；168 个张量有更新，末态最大绝对差 `7.45058e-9`、相对 L2 `3.92164e-11`。KTO forward 返回的 chosen/rejected log-prob、logit-sum 和 per-example KL log-prob 聚合值均已比较；这些值的相对 L2 约为 `2.44e-7` 至 `1.59e-6`。chosen/rejected logit-sum 的最大绝对差分别为 `0.875–1.125`。补充逐模型调用采集后，两个分支共同的前两次模型调用输入逐值相等，全部 25 层 hidden states 与完整 logits（形状 `[2, 29, 151936]`）均有限；最大相对 L2 `3.59102e-6`、最大绝对差 `3.29971e-4`。逐 token 证据说明 logit-sum 绝对差来自大规模求和累积，单个 logits 的最大绝对差为 `8.77e-5`。

运行键 `20261009-qwen2-kto-stock-cli-v3` 保存单步脚本、模型/数据摘要、输入与 forward 记录、checkpoint、比较 JSON 及原始日志，均未版本化。三步梯度/更新补验运行键 `20261010-qwen2-kto-stock-cli-l2-three-step-v2` / Slurm 15311 完成 native 与 strict shim 两条公开 CLI 轨迹。三步中 policy 与 reference 输入、标签和元数据逐项相同；每步覆盖 290 项参数、494,032,768 个元素，梯度最大绝对差分别不超过 `7.68e-7`，相对 L2 分别不超过 `9.70e-6`；更新最大绝对差每步为 `7.45e-9`，相对 L2 分别为 `8.78e-5`、`8.84e-5`、`8.37e-5`。三步 loss 最大差 `2.68221e-7`；step-3 checkpoint 的 290 个权重键均相同，171 个张量更新，最大绝对差 `7.45058e-9`、相对 L2 `7.28111e-11`。两组 SGD 公共选项相同（参数数 169/121，weight decay 0.1/0），每步状态项均为空；native Torch archive 与 shim 便携 pickle 经各自安全读取器解码后，checkpoint optimizer state dict 内容相等。shim 父子进程均记录 CUDA 与 `fallback=0`。审计键 v1/v2 的读取器与格式假设失败已保留；最终 GPU worker 只读审计键 `20261010-qwen2-kto-stock-cli-l2-three-step-artifact-audit-v3` / job 15317 通过全部三步、状态、末态权重及 fallback 断言。作业 15310 的 harness 错误未进入 shim，不属于兼容性结果。单步作业 13831/13835 分别因 PYTHONPATH 与过长 TMPDIR socket 路径退出；修正后的 v3 完成。

逐 token 补充证据保存在未版本化 state：native 模型输出运行键 `20261010-qwen2-kto-logits-v1` / Slurm 15259，strict shim 输出运行键 `20261010-qwen2-kto-logits-v3` / Slurm 15262，比较作业 Slurm 15266。15259 的 native 训练、采集和 checkpoint 均完成，之后因 harness 文件名断言错误退出；15261 的 shim 训练与采集完成，但遗漏 strict CUDA bootstrap，故不作为 strict 证据；新键 15262 补跑 shim 并通过 strict 启动检查，父/子进程均为 `cuda=1`、`shim=True`、`fallback=0`。同设备为 cscg-qh04 RTX 4090，native/shim 使用相同固定数据、模型权重、源码和 CLI 参数。比较器只对齐双方共同的前两次 reference 调用；shim 多出的第三次调用留作未解释差异。

### Reference 首次 graph replay 补查

新调用栈运行键 `20261010-qwen2-kto-extra-ref-call-trace-v1` / Slurm 15300 在相同单卡公开 CLI 配置下，先 native 后 strict shim，确认两次 `KTOTrainer.forward` 分别收到 `self.model` 和 `self.ref_model`，两侧分支身份相同。native policy/reference 底层模型 forward 均为 2 次；strict shim policy 为 2 次、reference 为 3 次。第三次来自 reference 分支 no-grad completion call 的首次 `GraphReplay`：`jittor/_core/module.py` 进入 `_runtime/graph_replay.py::__call__` 与 `_capture_now`，其中 eager warm-up 与 capture 都执行模型。Jittor 默认 `auto_graph_replay=1`，该机制用于满足重复签名/输入阈值的 no-grad 模块调用；它增加一次初始化计算，不代表 KTO 多执行了一个 reference 分支。

因果诊断运行键 `20261010-qwen2-kto-extra-ref-call-trace-noautoreplay-v1` / job 15303 只关闭严格 CUDA shim 进程的 `jt.flags.auto_graph_replay`，两分支底层 forward 数均恢复为 2；模型仍由公开 CLI 训练并保存 checkpoint，父子进程 CUDA=1、fallback=0。GPU-worker artifact audit job 15306 核对原 native 与 strict-shim reference 输入完全相同；KL 与 completion 的全部 27 个输出张量形状/dtype 对齐且有限，最大 relative L2 `3.59102e-6`、最大 absolute error `3.29972e-4`。strict shim completion 的 eager warm-up 与 graph-capture 返回张量逐元素完全相等；capture 返回与 native completion 输出亦在同一容差内。故固定 Qwen2-0.5B、FP32/eager、batch-2、一步的 KTO L1 前向合同通过；自动 replay 首次调用多一次内部计算作为性能边界保留，不将其推广为所有模块或场景结论。原始 trace/NPZ/worker 证据留在各运行键目录。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | PASS（当前树新键） | 当前树 `da7c978f1af8dec93bed3f827ec1887e945487a6` 的 stock CLI 构造审计覆盖 policy/reference 各 291 项 state、tokenizer/template/dataset/Trainer/optimizer/scheduler；两侧所需键、shape、dtype、CUDA device、值 hash、参数 trainability、数据/配置和调度状态相同。明确保留 `inv_freq.requires_grad`、实现类型路径及 optimizer 内部 group 表示差异，详见当前树 L0 小节。 |
| L1 | PASS（当前树固定 profile） | 当前树训练路径中，policy/reference 输入和标签逐步相同；输出键、shape、dtype和有限性符合合同。跨实现 forward 数值误差作为诊断记录于 `20261012-qwen2-kto-current-tree-l2-shim-v2`，不设训练 logits 容差门槛。 |
| L2 | PASS（当前树固定 profile） | 当前树 strict-shim 三步公开 CLI：290 个参数梯度逐项存在且有限、SGD 整体更新非空且有限、两个公开参数组和空状态语义有效、两侧各自固定 loss 窗口下降。job 18439 的首个比较器因逐张量非零断言过严退出，独立 posthoc comparator 对同一产物通过；详见上文。 |
| L3 | not-run（当前树） | 当前树未验证同进程/新进程 checkpoint 恢复、scheduler/RNG/dataloader 游标与后续轨迹。 |
| L4 | PASS（当前树固定 profile） | strict-shim 真实公开 RLHF CLI 完成三步 KTO 并保存 step-3 checkpoint；复用同配置、同模型/数据和相同 `python/jittor` 基线的既有原生 CLI 轨迹。完整恢复合同仍归 L3。 |
| L5 | blocked（当前树） | L3 尚未完成；本轮未做预热后至少 10 次同步稳态性能和显存测量。 |

Slurm 13856 的原始报告门禁通过；报告更新门禁 15269、15307、15319 的布局检查均通过，Torch-mode `tests/structure` 各为 1384 passed、8 skipped、1019 subtests passed。三个 worker 均没有 nvcc，因此 Jittor 结构门禁部分运行于 CPU；这只验证布局和结构，不构成 CUDA 兼容性证据。旧 v1/v2 的环境错误及其复验记录在 state，不影响门禁结论。Slurm 13838 与 13843 的首次原始门禁仅因根 `MANIFEST.in` 未列入报告及新增行排序不符而失败，修正清单后的 13856 全部通过。该报告只覆盖上列单卡 Qwen2-0.5B KTO 配置。
