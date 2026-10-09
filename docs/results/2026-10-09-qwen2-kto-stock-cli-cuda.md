# Qwen2-0.5B stock `swift rlhf --rlhf_type kto` CUDA 入口审计

- 状态：原生与 strict shim 均通过公开 KTO CLI 完成单步训练并保存 checkpoint；输入、初始参数、聚合 forward 输出、首步全参数梯度和末态权重均已采集。该固定单步配置的 L0–L2 为 partial，L3 未运行，L4 入口已运行但整体功能合同未通过，L5 blocked；不代表其他偏好算法或 ms-swift 整体兼容。
- 日期：2026-10-09。
- 基线：Jittor `35963c2e047dfa81751f82e8127064adf865d6bd`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，公开单卡 `swift rlhf --rlhf_type kto`，固定二条 preference 数据（chosen/rejected）、batch 2、SGD、一步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补三步固定轨迹、完整 token logits、optimizer/scheduler/RNG/dataloader 状态恢复后，重新按 L0–L5 逐层验收；Torch shim、Transformers、TRL 或 ms-swift KTO 入口变更时重跑。

模型权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；固定 preference 数据 SHA256 为 `05640c5363ecd9bf8162504ffee786a21f4c92a2bcd3fd64403a2269dd53baec`。运行环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0，ms-swift 自报版本 `4.6.0.dev0`。

Slurm 13834 在 `cscg-qh04` RTX 4090 上预检了 ms-swift 模块来源、CUDA 可用性和公开 RLHF CLI 帮助。Slurm 13836 随后在同一设备运行 native，再运行 shim。native 和 shim 的完整 290 项参数初态名称、shape、dtype、trainable 状态和值 hash 一致，均位于 `cuda:0`；策略与 reference 两次 KTO forward 所用的 CUDA 输入键、张量值及标签 `[false, true]` 完全相同。strict shim 的父、子 CLI 进程均记录 `use_cuda=1`、`shim=True` 和 `fallback=0`。

首步 loss 分别为 `0.5` 与 `0.5000002980232239`。比较器覆盖 290 个参数、494,032,768 个梯度元素，梯度最大绝对差 `5.14090e-7`、相对 L2 `7.45108e-6`。checkpoint 包含相同的 290 个权重键；168 个张量有更新，native/shim 末态最大绝对差 `7.45058e-9`、相对 L2 `3.92164e-11`。KTO forward 返回的 chosen/rejected log-prob、logit-sum 和 per-example KL log-prob 聚合值均已比较；这些值的相对 L2 约为 `2.44e-7` 至 `1.59e-6`。chosen/rejected logit-sum 的最大绝对差分别为 `0.875–1.125`，因为输入 logits 的量级很大，不能只凭相对误差把它们报为通过；原始逐 token logits 与 hidden state 未保存，因此 L1 保持 partial。

运行键 `20261009-qwen2-kto-stock-cli-v3` 保存脚本、模型/数据摘要、逐参数梯度、输入与 forward 记录、checkpoint、比较 JSON 及原始日志，均未版本化。前两次作业不属于兼容性结果：13831 在模型加载前因 CLI 子进程未继承 ms-swift checkout 的 `PYTHONPATH` 退出；13835 在数据 map 时因 TMPDIR 的 AF_UNIX socket 路径过长退出。v3 以短路径 symlink 指向运行键内 TMPDIR 后完成。map 子进程退出时出现 NFS `.nfs` 文件清理警告，但数据 map 与训练完成，未影响结果。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 真实模型、tokenizer/template、数据、KTO trainer 与 optimizer 经公开入口构造；290 项初态参数及两种输入分支逐项核对。完整 state/buffer 与 tokenizer/template 状态映射未审计。 |
| L1 | partial | 同权重、同 CUDA 输入下 KTO 聚合输出与 loss 已比较；未保存原始 token logits/hidden state，logit-sum 绝对误差超过默认前向绝对容差，需补充逐 token 分析。 |
| L2 | partial | 一步覆盖全部 290 项梯度与末态权重，数值差如上；仅一步，未形成至少三步固定轨迹，也未直接审计 optimizer 状态。 |
| L3 | not-run | 未验证同进程/新进程 checkpoint 恢复、scheduler/RNG/dataloader 游标与后续轨迹。 |
| L4 | 入口已运行，功能未通过 | native 与 strict shim 均通过 stock `swift rlhf --rlhf_type kto` 完成一步并保存 checkpoint；前置层不完整，不能标 L4 通过。 |
| L5 | blocked | 前置层未通过；未执行预热后至少 10 次同步稳态性能与显存协议。 |

Slurm 13856 的布局检查通过；结构测试为 1386 passed、6 skipped、1027 subtests passed。Slurm 13838 与 13843 的首次门禁仅因根 `MANIFEST.in` 未列入报告及新增行排序不符而失败，修正清单后的 13856 全部通过。该报告只覆盖上列单卡 Qwen2-0.5B KTO 配置。
