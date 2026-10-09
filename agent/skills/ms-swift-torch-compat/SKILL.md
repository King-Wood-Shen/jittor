---
name: ms-swift-torch-compat
description: 在真实 NVIDIA CUDA 上按 ms-swift 公开入口及 L0-L5 覆盖矩阵验证 Jittor torch shim；原生 PyTorch 是独立数值基准，未运行的面不得记为通过。
---

# ms-swift × Jittor torch shim：CUDA profile

## 目标与边界

回答当前 ms-swift checkout 的公开训练、推理和生态入口在 Jittor torch shim 上能否保持原生 PyTorch CUDA 的设备、数值、梯度、状态及性能合同。单个 `ms_swift_lora_llama` 生态用例只属于入口烟测，不能代表整个 ms-swift。结论只适用于真实 NVIDIA CUDA；CPU 可作独立 oracle，不把 NPU、ROCm 或软件回退计入 CUDA 通过。

开始前读仓库 `AGENTS.md`、`agent/manuals/collaboration.md`、`agent/manuals/project-context.md`、`../downstream-library-adaptation/SKILL.md`、`../transformers-torch-compat/SKILL.md` 与 `../github-collaboration-commit/SKILL.md`。以历史提交 `fb23d894e:refactor-wip/results/2026-09-26-ms-swift-cuda-compat.md` 及最新实验目录为历史证据；历史 PASS 必须绑定原 SHA、运行键和验收层级，不自动继承到新基线。

## 统一验收口径与完成范围

本任务以本 Skill 的 L0–L5 表为唯一等级：L0 构造、L1 前向、L2 梯度与更新、L3 完整恢复、L4 公开入口、L5 稳态性能。通用 `downstream-library-adaptation` 的同名等级只供其自身使用，不能换算为本任务等级。缺陷的 C 类是问题归属/严重度，不是验收等级；每个断点分别记录 C 类、首个失败 L 层与负责代码层。

先冻结本轮必须交付的代表性工作负载清单及其模型、tuner、设备、拓扑和公开入口。按实际调用机制选代表用例，不逐个穷举 ms-swift 注册的所有模型；未列入本轮合同或未执行的面仍在矩阵标为 `not-run`，不得推断为支持。新增功能面须有任务要求或具体共享机制缺口，不能因为探针容易跑就自动扩大必验范围。

每个格子同时记录“入口是否实际运行/产出”和“L0–L5 哪层已验收”。公开 CLI 跑完但 BF16 autocast、激活重算或完整状态合同不符时，写“入口已运行，功能未通过”，不能以 L4 通过，也不能把“L4 blocked”误写成 CLI 未运行。诊断性性能数字可记录，但只有适用正确性层与 L4 通过后才授予 L5。

## 先固定可复现范围

从实际 `$MS_SWIFT_ROOT/swift/`、`tests/`、`examples/`、`requirements/` 与 CI 入口生成 surface manifest。记录 ms-swift/Jittor SHA、解释器 ABI、torch/transformers/peft 等真实版本、可选依赖、模型权重和校验、数据、seed、dtype、优化器、步数、设备、Slurm job/node/GPU、独立 `JITTOR_HOME`。检查 Git 脏文件、正在运行的作业与缓存锁；不自动 stash、不清除既有补丁，不重复提交相同运行键。

manifest 至少逐面列状态、公开入口、真实代表用例、首个断点与证据：

| 面 | 必查内容 |
| --- | --- |
| 入口与配置 | `swift` CLI/launcher、Python API、参数解析、seed、device、dtype、保存和加载 |
| 模型 | 当前注册的 causal LM、encoder/seq2seq、分类、embedding/reranker、reward、MoE、视觉/音频/多模态 |
| tuner | ms-swift 私有 Swift LoRA、PEFT LoRA、QLoRA、prompt/prefix/adapters、冻结与 trainable 参数 |
| 训练 | SFT/pretrain/分类/embedding/reranker、梯度累积与裁剪、checkpointing、AMP、SGD/AdamW/Muon |
| 偏好与 RL | 当前 checkout 的 DPO/CPO/KTO/GKD/PPO/GRPO 等实际入口 |
| 分布式 | 单卡、单机双卡 NCCL、资源可用时多机；FSDP/DeepSpeed/Ray 各列状态 |
| 推理与服务 | `swift infer`、Python pipeline、Transformers engine、batch/stream、server/client/deploy |
| 数据与持久化 | tokenizer/template/dataset/collator、safetensors、完整 checkpoint、adapter export |
| 评测与扩展 | eval、sampling、metrics、自定义模型/数据/模板、callbacks 和插件 |
| 性能 | 真实尺寸稳态延迟、吞吐、显存口径、原生比值及 fallback |

每组尽量包含纯 Transformers 路径、ms-swift 私有组件和公开入口。模型权重、数据、依赖、外部服务或资源缺失分别记 `resource-blocked` 或 `not-run`；协议测试、随机 tiny 模型和排队作业不能填为真实数值 PASS。

## 作业前预检与编译阶段

在短时 Slurm worker 预检解释器/模块来源、安装 pin、模型和数据摘要、CLI 参数、钩子签名、审计变量、短 `TMPDIR`、JIT/cache 路径、结果读取器对实际 dtype 的支持。预检未过不启动长模型作业；harness 或调度失败记为实验无效，不记为兼容失败。对拍前保存并比较每步样本 ID、`input_ids`、mask、labels、初始权重与参数名映射；相同 seed 和相近 loss 都不能替代输入同一性。

把冷编译、正确性训练、结果比较分成独立阶段与耗时。仅在源码/ABI/编译器/设备指纹完全一致时复用已验证的预编译产物；并发作业仍使用独立可写缓存，不并发写同一 `JITTOR_HOME`。不得将首次 JIT 计入稳态 L5。前置作业失败时立即收集日志、标记运行键、取消不可能有效的依赖作业；修复后用新运行键重建，不留下 `DependencyNeverSatisfied` 队列。

## 执行与归属

只经 Slurm NVIDIA worker 做所有导入、JIT、测试和计算；登录节点只用于静态阅读、编辑、Git、提交和调度。原生 PyTorch CUDA 在没有 shim 标记的独立进程先跑；候选随后在 `JITTOR_TORCH_SHIM=1` 下跑同一 checkpoint、初始权重、输入与数据顺序。候选从导入到结果保存启用 `jt.runtime.scope(use_cuda=1, backend_fallback="error")` 和 `forbid_backend_fallbacks()`，证明模型参数、输入、输出和梯度均为 CUDA，`fallback_count=0`。并行 JIT 使用不同 `JITTOR_HOME`，首次编译串行；单测与 benchmark 不共用正在使用的缓存。

按三行判据归属：Jittor core/CUDA 负责能力、autograd、kernel、optimizer；`jittor.compat.torch` 负责公开 Torch API 的名称、签名、容器、device/dtype/state 语义；adapter 只处理经验证确属 ms-swift 私有 glue 的可迁移逻辑。不得修改 ms-swift 源码掩盖失败。每个问题最多五轮；失败保存根因、日志、受影响层和解除条件，随后推进独立面。已达到上限的历史问题不重启，除非目标或证据发生实质变化。

双卡须验证实际安装的 NCCL include/lib、`RANK`/`WORLD_SIZE`/`LOCAL_RANK` 与 Jittor rank 变量，逐进程记录 `CUDA_VISIBLE_DEVICES` 和设备 UUID。提交前查 `squeue`/`sacct` 与已有产物，避免重复作业。

## L0–L5 验收

前一层失败则后续层写 `blocked`，不跨级标记 PASS。

| 层 | 必需证据 |
| --- | --- |
| L0 构造 | 独立原生和 shim 导入，真实模型/tokenizer/template/dataset/tuner/optimizer/trainer 或 engine 构造；状态键、dtype、device 一致 |
| L1 前向 | 同权重同输入 CUDA 前向；结构、shape、dtype、有限值、logits/hidden/loss 与适用的 greedy token 对齐 |
| L2 反向与更新 | 全部 trainable 参数及适用输入梯度、optimizer 状态与更新，至少三步固定数据轨迹；需要时用独立 FP64 公式复核 |
| L3 恢复 | 同进程和新进程恢复模型/adapter、buffer、optimizer、scheduler、RNG、dataloader 游标与 global step，并对齐后续轨迹 |
| L4 公开入口 | 真正通过适用的 `swift` CLI、Python API、launcher 或 server/client 端到端，保留输出与 checkpoint |
| L5 性能 | 适用 L0–L4 已通过；真实尺寸预热并同步计时至少 10 次稳态，给延迟/吞吐、原生比、显存口径和零 fallback |

默认 CUDA 前向容差 `5e-3`，反向 `2e-2`；同时给最大绝对误差、相对 L2、dtype 与首个分歧层。BF16、RMSNorm、attention、量化和 logits 要单列误差，不事后放宽门槛。L5 不能把首次 JIT、缓存构建或不同显存统计口径混进对比。

## 根因优先与远端续接

先修已证实的首个失败层：最小复现、按 core/compat/adapter 分流、修复、定向回归，再扩展模型或长轨迹。同一根因最多五轮有不同假设的修复；仍失败就记录日志、解除条件和边界，转到独立面，不把重复运行算作新尝试。

Slurm 作业运行或冷编译期间由轻量状态检查等待；仅在作业终态、失败、证据完成或需要决策时唤起 Codex。无状态变化不按分钟重复开启推理轮次、重复 Git fetch 或写进度提交。Git 同步按任务开始、集成与推送边界执行；fork 的 fetch/push 使用已验证的同一 SSH 身份，短暂网络失败保留最近可信 SHA 并退避重试。结果矩阵只在验收结论、根因或范围变化时更新，原始逐次日志保留在 state。

若当前只剩等待 Slurm 作业，在回复末尾独占一行写 WAIT_JOBS=123,456（真实作业号）；监督脚本仅用 shell 等状态变化或最多 30 分钟再唤起 Codex。

## 证据与提交

原始日志、NPZ/JSON、模型、缓存、环境和精确命令放在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/<run>/`，仓库结果报告只存结论和可复现索引。报告要保留 manifest、每面 L0–L5、SHA、dirty diff 摘要、运行键、job/node/GPU、误差、性能口径、fallback、首个失败及 `not-run`/`resource-blocked` 原因。只以真实原生基准、严格 CUDA、零 fallback 共同支持通过结论。

修复先最小复现，再做定向回归、布局和受影响结构测试。所有测试在 Slurm worker；结束前静态执行 `git diff --check`。按明确文件路径暂存并写中文、功能粒度 commit；不提交缓存或未完成实验，不推送未经用户要求的远端。
