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
| L0 | 历史基线 `82c2`：partial；复验基线 `a77725d5`：PASS | 历史 export 记录见上。job15999 新进程从两侧已合并导出模型加载完整 291 项 state（参数和 buffer），名称、shape、dtype、device、finite 元数据相同且均为 FP32 CUDA；tokenizer 与真实模型权重仍为固定产物。 |
| L1 | 历史基线 `82c2`：partial；复验基线 `a77725d5`：PASS | job15999 相同 tokenizer 输入 `[2,8]`，native/shim logits 均为有限 FP32 CUDA `[2,8,151936]`；最大绝对误差 `8.03471e-5`、相对 L2 `1.75080e-6`、argmax 一致率 100%，满足固定阈值 `abs≤1e-4, relative L2≤1e-5`。 |
| L2 | not-applicable | 该路径不执行反向传播或 optimizer 更新。 |
| L3 | not-run | 没有在新进程加载导出模型并验证后续轨迹；无训练 optimizer/scheduler/RNG 恢复证据。 |
| L4 | 历史基线 `82c2`：PASS；复验基线 `c43f87d3`：PASS | job16141 在当前代码基线通过原生与 strict shim 公开 CLI 导出、完整参数 CUDA 清单及 safetensors 逐张量比较；补充细节见下文。 |
| L5 | not-applicable | 离线合并导出不是训练或在线推理性能场景；本次未测模型服务稳态性能。 |

## 当前基线补充：导出模型 L0/L1 复验（2026-10-11）

本节仅补齐当前 Jittor 基线 `a77725d596454c333a38e4e3cf5670aead599523` 下的导出模型构造与前向证据，不改写上方 `82c2f9e` 的历史公开 CLI 结论。ms-swift 仍为 `88d727951203256baa564c643c651b6f8d90fd7e`。原生 merged export 来自 job13351，strict-shim merged export 来自 job13352；新审计只读复用这两个产物，未重新导出。

新运行键 `20261011-qwen2-peftlora-export-merge-l01-v3` / Slurm job15999 在 cscg-qh10 RTX 4090（UUID `GPU-9fd9b4e3-8bd4-9db8-0a3d-0ea6cfbd07bf`）运行 strict shim；独立原生 oracle 及输入、logits 由 job15947 在 cscg-qh06 RTX 4090（UUID `GPU-e0c8764d-4b51-62e5-f0d3-4aa27a8ddb68`）先行完成。job15947 因35分钟时限在 Jittor 冷编译中终止，但原生完整产物有效；job15997 因缺 CUDA 初始化环境变量在 shim 前置阶段失败，均作为 harness/资源失败保留，不作为兼容结论。v3 补齐 CUDA toolkit、Python config 与 build 环境后通过。

两侧在各自导出目录用 Transformers `AutoModelForCausalLM.from_pretrained`/`AutoTokenizer.from_pretrained` 加载 Qwen2-0.5B，FP32、eager、local-only。291 项 state 参数与 buffer 的名称、shape、dtype、device、有限性清单相同且全在各自 `cuda:0`；两条固定文本各自分词后得到相同 `[2,8]` input IDs 和 attention mask。logits 是有限 FP32 CUDA `[2,8,151936]`，最大绝对误差 `8.0347061e-5`，相对 L2 `1.7507991e-6`，argmax 一致率 100%；预设绝对阈值 `1e-4` 与相对 L2 阈值 `1e-5` 均通过。候选 `shim-runtime.json` 记录 `use_cuda=1`、shim=true、fallback_count=0。

前向比较跨不同 RTX 4090 worker 完成，故仅声称该固定模型/导出物和输入 profile 的阈值通过，不声称同卡严格复验或性能比值。原始产物和失败日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-peftlora-export-merge-l01-v1/`、`...-v2/` 与 `...-v3/`。v1 是已完成的 native oracle 加超时 shim 尝试，v2 是 CUDA 初始化环境缺失的无效尝试，v3 为通过的严格候选；不重跑任一运行键。

## 当前基线补充：公开合并导出 CLI L4（2026-10-11）

新运行键 `20261011-qwen2-peft-export-l4-cli-v1` / job16141 使用当前 Jittor 提交 `c43f87d3ef1c990a567580c4c3a2dbf423f2a4ee`（相对上节 `a77725d5` 仅增加结果文档，不改变运行代码），ms-swift checkout 仍为 `88d727951203256baa564c643c651b6f8d90fd7e`。在 cscg-qh13 RTX 4090（UUID `GPU-47aa9ed3-c3cd-c68a-15af-c3fd65e992e5`）顺序运行独立原生 PyTorch CLI，再运行 `JITTOR_TORCH_SHIM=1` strict CUDA CLI；两侧均使用同一 Qwen2-0.5B FP32/eager 参数、模型 SHA256 `9cd8fc8c...df77cad` 和 adapter SHA256 `e34a28cd...16e7c56`，均由真实 `swift export --merge_lora` 入口完成合并并保存。

原生和候选的实际 `swift/cli/export.py` 服务进程保存钩子各记录 290/290 参数在 `cuda:0`。候选 CLI 父进程与子进程均记录 shim marker、`use_cuda=1`、`fallback_count=0`；原生两进程均没有 shim marker。两侧 safetensors 的 290 个张量键/shape/dtype/有限值检查通过，最大绝对差和相对 L2 均为 0，非零差异张量数为 0。job16141 Slurm 状态为 FAILED，原因仅是其尾部审计器错误地要求每侧恰有单进程四条事件；真实父、子进程分别记录 start/end，故各侧共六条事件。没有重跑 CLI；独立 GPU worker job16147 以修正的父/子进程审计规则检查原始事件、CUDA 清单、零 fallback、比较 JSON、保存文件及两侧 CLI 成功日志，全部通过。

原始事件、逐参数 device inventory、模型导出与日志位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-peft-export-l4-cli-v1/`；只读审计日志位于 `.../20261011-qwen2-peft-export-l4-artifact-audit-v1/`。本结论只覆盖固定模型、adapter 和公开合并导出 CLI；不外推到量化、其他 tuner 或整个 ms-swift。当前层级仍为 L0/L1 PASS、L2 not-applicable、L3 not-run、L4 PASS、L5 not-applicable。
