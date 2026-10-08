# Qwen2-0.5B PEFT LoRA 公开 `swift export --merge_lora` CUDA 对拍

- 状态：固定 PEFT LoRA adapter 与 Qwen2-0.5B FP32/eager 配置下，原生与严格 CUDA shim 的公开合并导出成功；导出 safetensors 的 290 个参数逐项完全相同。此结论只覆盖该模型、adapter 和导出路径，不代表所有 ms-swift 导出或量化路径。
- 日期：2026-10-08。
- Jittor 基线：`82c2f9e4671003f4ffc5e7d929c9f4c0318a95c7`；同步目标 `origin/2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`，为当前提交祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B 基座 `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；输入 adapter 权重 SHA256 `e34a28cdc1d227907e13b96f061da71053f13fe0e68d678d274fdc69816e7c56`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：`swift export`/PEFT merge、`save_checkpoint`、Transformers 加载或 torch shim 的 device/fallback 行为变更时。

运行键 `20261008-qwen2-peftlora-export-82c2-r1/r2`。Slurm 13351 与 13352 均运行在 cscg-qh13 RTX 4090（UUID `GPU-5331cb2a-7ec6-ad9a-5890-cb43fd723398`，driver `580.178.04`）。原生阶段 13351 和 shim 阶段 13352 使用相同 `swift export --model <Qwen2-0.5B> --adapters <initial-adapter> --model_type qwen2 --task_type causal_lm --tuner_type lora --tuner_backend peft --torch_dtype float32 --attn_impl eager --merge_lora true --safe_serialization true`。两侧各自输入模型权重和 adapter 权重哈希相同。native 导出日志明确记录 `device_map='cuda:0'` 并成功保存；候选从 CLI 父进程和实际 `swift/cli/export.py` 服务进程启用严格 CUDA scope 与 `forbid_backend_fallbacks()`。

候选保存钩子逐项检查合并后模型参数：290/290 参数均在 CUDA，然后通过 ms-swift 原有 `save_checkpoint` 写 safetensors。GPU worker 上逐张量读取两份导出文件并检查键集合、shape、dtype、有限值及数值差：290 个键完全一致，最大绝对误差 `0.0`、相对 L2 `0.0`，非零差异张量数为 0。候选父/子进程启动与正常退出的 `use_cuda=1`、`fallback_count=0`，参数合并和保存事件的 fallback 也均为 0。

首次候选尝试 13351 发现 bootstrap 只审计了 `swift.cli.main` 父进程，未覆盖其实际 `swift/cli/export.py` 子进程，因此未形成可采信的候选审计/比较结果并取消；该任务的原生导出阶段完整成功，原始日志保留。修正 bootstrap 后，13352 对候选实际 CLI 子进程审计并成功完成。本次不复用失败候选结果。

原始日志、导出模型、参数设备清单、strict bootstrap、比较脚本和比较 JSON 未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-peftlora-export-82c2-r1/` 与 `...-r2/`；r1 保存 13351 原生导出和被取消候选的痕迹，r2 保存 13352 shim 完整导出及逐参数比较。模型合并计算及文件比较均在 GPU worker 执行。

提交前 Slurm 13360 在 GPU worker 通过 `bash tools/check_repo_layout.sh` 与 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`：1386 passed、6 skipped、1027 subtests passed（11 分 27 秒）。清单生成器 `--check` 与 `git diff --check` 也通过；完整门禁日志保存在 r2 运行目录。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | partial | 两侧公开 export 均构造模型/PEFT adapter；native 记录 `device_map=cuda:0`，shim 合并后完整 290 参数 CUDA 清单通过；native 未留逐参数设备清单。 |
| L1 | partial | 公开 merge 结果的 290 个参数逐项完全相同；没有从导出模型再执行 logits/hidden forward，故前向验收未完整。 |
| L2 | not-applicable | 该路径不执行反向传播或 optimizer 更新。 |
| L3 | not-run | 没有在新进程加载导出模型并验证后续轨迹；无训练 optimizer/scheduler/RNG 恢复证据。 |
| L4 | partial | 两侧均通过公开 `swift export --merge_lora` 保存可序列化模型；受 L0/L1 的部分证据限制，保守记 partial。 |
| L5 | not-applicable | 离线合并导出不是训练或在线推理性能场景；本次未测模型服务稳态性能。 |
