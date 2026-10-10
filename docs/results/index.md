# 验证结论

这一节保存**可复现的维护者验证结论**：一个具体的现象、它的根因、修复，以及修复
之后在真实设备上量到的数字。写的是"我们查过什么、结果怎么样"，不是项目历史，也
不是教程——想让某个模型或后端跑起来、想对照某个已知问题的结论时看这里。

每条结论开头都注明状态、日期、基线提交、验证范围、维护者和复查条件。过期的结论会
被删除或归档，不在多个文件里重复维护；压缩过的报告只保留结论、最终数字、复现命令与
未结事项，逐步的记录留在 Git 历史里。

```{toctree}
:maxdepth: 1

2026-08-28-ascend-910b-validation
2026-08-30-qwen3-ascend-training
2026-08-31-vllm-ascend-jittor-bootstrap
2026-09-02-verl-ascend-core-algorithms
2026-09-04-import-jittor-cost-attribution
2026-09-12-minimax-h3-torch-compat
2026-09-12-cuda-metaop-launch-index-scalar
2026-09-12-batched-linear-graph-nodes
2026-09-13-host-path-per-node-cost
2026-09-14-autocast-conv-mixed-dtype
2026-09-14-exit-heap-corruption
2026-09-14-jittor-vs-pytorch
2026-09-14-vllm-omni-h3-enablement
2026-09-19-torch-compat-runbook-verification
2026-09-24-torch-compat-real-models
2026-09-25-profiling-tools
2026-10-07-ms-swift-cuda-compat-sync
2026-10-07-qwen2-infer-1k-ffeb
2026-10-07-qwen2-ia3-infer-ffeb
2026-10-08-qwen2-ia3-forward-349d
2026-10-07-qwen2-causal-ddp-ffeb
2026-10-07-qwen2-causal-sft-ffeb
2026-10-08-qwen2-causal-sft-forward-c23a
2026-10-08-qwen2-causal-sft-grad3-4a7e
2026-10-08-qwen2-causal-sft-gradient-checkpointing-cuda
2026-10-08-qwen2-causal-sft-bf16-amp-cuda
2026-10-08-qwen2-causal-sft-l0-audit
2026-10-08-qwen2-seqcls-sft-cuda
2026-10-08-qwen2-seqcls-infer-cuda
2026-10-08-qwen2-seqcls-python-api-cuda
2026-10-08-qwen2-seqcls-ddp-nccl-cuda
2026-10-08-qwen2-peftlora-sft-init
2026-10-08-qwen2-peftlora-adamw
2026-10-08-qwen2-pt-fullparam-02498
2026-10-08-qwen2-causal-ddp-7a18-v4
2026-10-08-qwen2-causal-ddp-grad3-6c104
2026-10-08-qwen2-causal-ddp-resume-7a18
2026-10-08-qwen2-dpo-full-cuda
2026-10-08-qwen2-cpo-peftlora-cli-cuda
2026-10-08-ms-swift-eval-dependency-block
2026-10-08-torch-randomsampler-generator
2026-10-08-qwen2-swiftlora-forward
2026-10-08-qwen2-swiftlora-grad3
2026-10-08-qwen2-swiftlora-sft-cli-d48
2026-10-08-qwen2-embedding-trainer-infonce-817fb
2026-10-08-qwen2-embedding-sft-cli-361bf
2026-10-08-qwen2-causal-transformersengine-api
2026-10-08-qwen2-service-infer-361bf
2026-10-08-qwen2-peftlora-export-merge-82c2
2026-10-09-qwen2-kto-stock-cli-cuda
2026-10-09-qwen2-orpo-stock-cli-cuda
2026-10-09-qwen2-reward-stock-cli-cuda
2026-10-09-qwen2-rewardtrainer-api-cuda
2026-10-09-qwen2-simpo-2gpu-cli-cuda
2026-10-09-qwen2-simpo-stock-cli-cuda
2026-10-09-torch-lrscheduler-state-cuda
2026-10-10-qwen2-causal-ddp-sft-cuda
2026-10-10-qwen2-peftlora-public-infer-cuda
```

## 按主题索引

| 结论 | 状态 | 日期 |
| --- | --- | --- |
| [Ascend 910B3：冷缓存启动、ACL 执行与 NPU 门禁](2026-08-28-ascend-910b-validation.md) | NPU 门禁 397 passed / 9 skipped；skip 对应的能力边界仍开放 | 2026-08-28，2026-09-01 复查 |
| [Qwen3-0.6B 在 Ascend 910B3 上训练](2026-08-30-qwen3-ascend-training.md) | FP32 训练与 BF16 单步对齐接受；BF16 精确路径慢约 6%、训练轨迹未验收 | 2026-08-30，2026-09-02 |
| [vLLM 在 Ascend 910B3 上经 Jittor 运行](2026-08-31-vllm-ascend-jittor-bootstrap.md) | 单请求 Qwen3-0.6B 正确且快 6.8%；外置 NPU 插件未入仓，多请求未测 | 2026-08-31，2026-09-02 |
| [verl 核心算法在 Ascend 910B3 上](2026-09-02-verl-ascend-core-algorithms.md) | 六类 loss 与梯度逐位一致；五条慢 1.25–1.75 倍，NPU 端到端 PPO 未跑 | 2026-09-02 |
| [热缓存 `import jittor` 的耗时归因](2026-09-04-import-jittor-cost-attribution.md) | 归因接受，热缓存已低于 1 s；冷缓存仍在 import 时编译核心 | 2026-09-04 |
| [MiniMax-H3 在 Torch 兼容层下跑通](2026-09-12-minimax-h3-torch-compat.md) | 端到端跑通，达到 parity 速度 | 2026-09-12，2026-09-14 复验 |
| [autocast 请求了混合 dtype 的卷积](2026-09-14-autocast-conv-mixed-dtype.md) | 已修复，真实 CUDA 设备验证 | 2026-09-14 |
| [退出期 "corrupted double-linked list"](2026-09-14-exit-heap-corruption.md) | 已修复，多次真实运行验证 | 2026-09-14 |
| [jittor + torch-compat + vLLM-Omni 接入 H3 暴露的 shim 缺陷](2026-09-14-vllm-omni-h3-enablement.md) | 单卡与 TP2 请求端到端正确（710.7 s → 39.6 s），部署侧视频噪声已修；参考 VAE 解码仍比 PyTorch 慢 1.33x | 2026-09-14 至 2026-09-22 |
| [CUDA 元算子：发射配置、strided 下标与标量常量融合](2026-09-12-cuda-metaop-launch-index-scalar.md) | 部分完成，H20 上实测 | 2026-09-12 |
| [主机受限步长：把每算子的 Python 开销从图构建里拿掉](2026-09-13-host-path-per-node-cost.md) | Python 路径已完成 | 2026-09-13 |
| [batched Linear 的两个展平节点](2026-09-12-batched-linear-graph-nodes.md) | 已落地，H20 上实测；对拍结论未附 | 2026-09-12，2026-09-23 归档 |
| [Jittor vs 真 PyTorch 2.9.1：现在差在哪](2026-09-14-jittor-vs-pytorch.md) | 快照：对 eager 赢 6 平 5 输 1，对 `torch.compile` 赢 2 平 2 输 5 | 2026-09-14 |
| [Torch 兼容层在真实模型上对 PyTorch：差距表、显存与性能修复](2026-09-24-torch-compat-real-models.md) | 11 项全部跑通，几何平均 1.26x，进程显存峰值为 PyTorch 的 0.87–1.45 倍；余下差距在主机侧 | 2026-09-24 |
| [性能/显存分析工具：审计与重写](2026-09-25-profiling-tools.md) | 已实现，RTX 4090 上验证；未合入 | 2026-09-25 |
| [ms-swift CUDA 适配：基线同步与长上下文流式断点](2026-10-07-ms-swift-cuda-compat-sync.md) | 新基线完整性能协议段错误；长上下文流式 L4/L5 未通过 | 2026-10-07 |
| [Qwen2-0.5B 公开 infer CLI 千 token 非流式复验](2026-10-07-qwen2-infer-1k-ffeb.md) | 补齐完整序列 logits 与 hidden-state 对拍；状态恢复和完整 L0–L5 验收未完成 | 2026-10-07 |
| [Qwen2-0.5B IA3 adapter 公开 infer](2026-10-07-qwen2-ia3-infer-ffeb.md) | 同一 adapter 下两条公开输出相同；逐参数设备、完整 logits 与恢复未验 | 2026-10-07 |
| [Qwen2-0.5B IA3：Transformers/PEFT 直接前向](2026-10-08-qwen2-ia3-forward-349d.md) | 当前同步基线参数设备、完整 logits/hidden-state 对拍及公开 infer CLI 输出对齐；恢复未验，形式 L4/L5 受阻 | 2026-10-08 |
| [Qwen2-0.5B 公开全参数 SFT 双卡三步与恢复复验](2026-10-07-qwen2-causal-ddp-ffeb.md) | 一步双卡逐参数梯度对拍完成；三步扩展在原生 qh09 设备枚举处失败；完整 L0–L5 待验 | 2026-10-07 |
| [Qwen2-0.5B 公开全参数 SFT 三步复验](2026-10-07-qwen2-causal-sft-ffeb.md) | 补 scheduler 修复后的 step-3→4 新进程续训：290 键权重与 scheduler 对齐且零 fallback；optimizer、RNG 实值、数据游标和同进程恢复未验，L3 仍 blocked | 2026-10-09 |
| [Qwen2-0.5B 公开全参数 SFT 三步逐参数梯度对拍](2026-10-08-qwen2-causal-sft-grad3-4a7e.md) | 290 项全参数梯度三步对齐；状态、更新轨迹与恢复仍未完成 L0–L4 验收 | 2026-10-08 |
| [Qwen2-0.5B 公开 SFT gradient checkpointing CUDA 复验](2026-10-08-qwen2-causal-sft-gradient-checkpointing-cuda.md) | 三步数值与末态权重接近且零 fallback；shim 直通不重算激活、不节省显存，checkpointing 内存语义不兼容 | 2026-10-08 |
| [Qwen2-0.5B 公开 SFT BF16 autocast CUDA 复验](2026-10-08-qwen2-causal-sft-bf16-amp-cuda.md) | FP32 模型请求 BF16 autocast 时兼容层实际转 FP16；三步训练零 fallback，末态权重接近但 loss/梯度轨迹不同，optimizer 状态未比对 | 2026-10-08 |
| [Qwen2-0.5B 公开 SFT 单卡初始参数 CUDA 审计](2026-10-08-qwen2-causal-sft-l0-audit.md) | 290 个 trainable 参数的 CUDA 初态跨运行时逐项相同；state-dict 参数名和后续层仍未验，L0 partial | 2026-10-08 |
| [Qwen2-0.5B 序列分类公开 SFT CUDA 对拍](2026-10-08-qwen2-seqcls-sft-cuda.md) | 共用 checkpoint/首批后 logits 与 loss 对齐；参数名映射、梯度状态与恢复未验，L0/L1 partial | 2026-10-08 |
| [Qwen2-0.5B 序列分类公开 infer CLI CUDA 对拍](2026-10-08-qwen2-seqcls-infer-cuda.md) | 固定 checkpoint 与输入下预测类别一致、首批 logits 达容差、严格 CUDA/零 fallback；L0/L1 partial，恢复和性能未验 | 2026-10-08 |
| [Qwen2-0.5B PEFT LoRA 公开 `swift infer` CUDA 对拍](2026-10-10-qwen2-peftlora-public-infer-cuda.md) | 固定 adapter 与两条请求的 L0、完整生成 logits L1、公开 CLI L4、10 次稳态性能 L5 通过；L2/L3 不适用 | 2026-10-10 |
| [Qwen2-0.5B 序列分类 Python 推理 API CUDA 对拍](2026-10-08-qwen2-seqcls-python-api-cuda.md) | 固定 batch 的 state_dict、输入、logits 与输出通过；Python API L4 端到端及 10 次稳态 L5 完成，其他模型/任务未覆盖 | 2026-10-08 |
| [Qwen2 序列分类公开 SFT 双卡 NCCL CUDA 对拍](2026-10-08-qwen2-seqcls-ddp-nccl-cuda.md) | 固定场景双 rank 三步输入、291 参数梯度、更新权重对齐且零 fallback；scheduler 状态不等，恢复与性能未验 | 2026-10-08 |
| [Qwen2-0.5B 全参数 SFT 首批前向对拍](2026-10-08-qwen2-causal-sft-forward-c23a.md) | 固定 batch 的 logits/hidden/loss 对拍通过数值门槛；完整层级仍受限，诊断钩子在 backward 触发内部断言 | 2026-10-08 |
| [Qwen2-0.5B PEFT LoRA 公开 SFT：同状态三步对拍](2026-10-08-qwen2-peftlora-sft-init.md) | 历史严格逐值 L0-L2 结论保留；新收敛协议复验在 strict shim 第 5 步触发 NanoVector slice overflow、L2 未通过；L3 partial，L5 blocked | 2026-10-10 |
| [Qwen2-0.5B PEFT LoRA：AdamW 三步公开 SFT 对拍](2026-10-08-qwen2-peftlora-adamw.md) | 历史严格逐值结论保留；新协议固定配置 L0-L4 通过；L5 原生完成 10 个稳态测量，strict shim 第 3 步同步触发 slice overflow，blocked | 2026-10-10 |
| [Qwen2-0.5B 公开 `swift pt` 全参数 CUDA 三步对拍](2026-10-08-qwen2-pt-fullparam-02498.md) | 三步公开入口完成；checkpoint 290 键逐位一致，设备清单/梯度/optimizer 状态不足，L0-L2 partial | 2026-10-08 |
| [Qwen2-0.5B 公开全参数 SFT：双卡 NCCL 三步新基线复验](2026-10-08-qwen2-causal-ddp-7a18-v4.md) | 双卡公开 CLI 与 290 键 checkpoint 对齐；step-3→4 恢复为 partial，逐参数梯度和 L5 未验，L0-L4 partial | 2026-10-08 |
| [Qwen2-0.5B 公开全参数 SFT 双卡三步逐参数梯度补证](2026-10-08-qwen2-causal-ddp-grad3-6c104.md) | 已补候选初态清单；输入采集边界和 `use_logits_to_keep` 不一致，L0-L2 partial，L3 未运行，L4 partial，L5 blocked | 2026-10-09 |
| [Qwen2-0.5B 双卡公开 SFT checkpoint 恢复对拍](2026-10-08-qwen2-causal-ddp-resume-7a18.md) | 各 runtime 连续与 step-3→4 恢复的模型、优化器、scheduler、RNG 和末步指标对齐；sampler cursor 与前序层级证据不足，L3 partial | 2026-10-08 |
| [ms-swift 公开 `swift eval`：依赖前置阻断](2026-10-08-ms-swift-eval-dependency-block.md) | Slurm worker 缺 `evalscope` 且代理拒绝连接；模型评估未运行，L0-L5 blocked/not-run | 2026-10-08 |
| [Qwen2-0.5B 公开全参数 DPO 三步 CUDA 对拍](2026-10-08-qwen2-dpo-full-cuda.md) | 原生三步成功；strict CUDA shim 首步反向触发 `NanoVector` 断言；固定 batch 汇总前向标量接近但完整张量未验，L1 partial、L2 failed | 2026-10-08 |
| [Qwen2-0.5B 公开 ORPO CLI 固定样本收敛与多样本输入身份](2026-10-09-qwen2-orpo-stock-cli-cuda.md) | 单固定样本12步 profile 的 L0-L2/L4 通过，L3 not-run、L5 blocked；多样本 sampler 顺序仍未对齐，不能外推至整个 ORPO 或 ms-swift | 2026-10-10 |
| [Qwen2-0.5B 公开 CPO LoRA `swift rlhf` CUDA 对拍](2026-10-08-qwen2-cpo-peftlora-cli-cuda.md) | 固定单偏好对、同初态单卡 profile：L0-L2 与公开 CLI L4 通过，strict CUDA/零 fallback；L3 未运行，L5 blocked。其他 CPO 场景仍未覆盖 | 2026-10-10 |
| [Torch `RandomSampler` 显式 Generator CUDA 运行时对拍](2026-10-08-torch-randomsampler-generator.md) | 同 seed 跨 runtime 的样本顺序不同；各 runtime 内 generator 状态恢复可重放；仅测试 sampler API，不包含模型或 Swift CLI | 2026-10-08 |
| [Qwen2-0.5B ms-swift 私有 Swift LoRA 固定状态前向](2026-10-08-qwen2-swiftlora-forward.md) | 当前基线下直接 `Swift.prepare_model` 前向通过数值门槛，严格 CUDA/零 fallback；L0 partial，训练/恢复/公开入口/性能未运行 | 2026-10-08 |
| [Qwen2-0.5B ms-swift 私有 Swift LoRA 三步梯度与更新对拍](2026-10-08-qwen2-swiftlora-grad3.md) | 96 个 trainable 参数三步梯度、AdamW 状态和 adapter 更新通过固定直接 API 对拍；公开 Trainer/CLI、恢复与性能未验证 | 2026-10-08 |
| [Qwen2-0.5B ms-swift 私有 Swift LoRA 公开 `swift sft` CUDA 对拍](2026-10-08-qwen2-swiftlora-sft-cli-d48.md) | 固定 CLI 配置的 logits、三步 adapter 梯度/AdamW 更新通过；新进程恢复因 Transformers 未识别私有 tuner 而在训练前失败，L0/L4 partial、L3 blocked、L5 blocked | 2026-10-08 |
| [Qwen2-0.5B ms-swift EmbeddingTrainer InfoNCE CUDA 基线复验](2026-10-08-qwen2-embedding-trainer-infonce-817fb.md) | 新基线下固定 direct Python Trainer API 三步 embedding forward/loss/梯度/更新对拍通过，严格 CUDA/零 fallback；L3、L5 未运行 | 2026-10-08 |
| [Qwen2-0.5B ms-swift Embedding `swift sft` CLI CUDA 复验](2026-10-08-qwen2-embedding-sft-cli-361bf.md) | 当前源码补验完整参数构造、embedding 前向、三步梯度/更新、step-1 中途 checkpoint 恢复及 10 次同步性能测量，L0-L5 固定配置通过、严格 CUDA/零 fallback | 2026-10-10 |
| [Qwen2-0.5B causal LM `TransformersEngine` Python API CUDA 对拍](2026-10-08-qwen2-causal-transformersengine-api.md) | 固定两请求非流式生成在 31 步 logits、token IDs 和文本上对齐；L0/L1/L4/L5 通过、L2/L3 不适用、strict CUDA/零 fallback | 2026-10-08 |
| [Qwen2-0.5B ms-swift `swift deploy` 服务端/客户端 CUDA 复验](2026-10-08-qwen2-service-infer-361bf.md) | 原生与严格 shim 服务对 12 次 HTTP 请求文本/token 用量完全一致；10 次稳态延迟比 0.964，服务进程零 fallback，L0/L1/L4/L5 partial | 2026-10-08 |
| [Qwen2-0.5B PEFT LoRA 公开 `swift export --merge_lora` CUDA 对拍](2026-10-08-qwen2-peftlora-export-merge-82c2.md) | 历史 `82c2` 公开 CLI 结论保留；当前 `a77725d5` 导出物 L0/L1 复验通过、零 fallback；当前基线 L4 not-run，L3 not-run | 2026-10-11 |
| [Qwen2-0.5B stock `swift rlhf --rlhf_type kto` CUDA 入口审计](2026-10-09-qwen2-kto-stock-cli-cuda.md) | 解释 shim 额外 reference forward 来自 no-grad 首次 GraphReplay eager/capture；三步全参数梯度、更新、loss 与末态权重对拍通过，固定配置 L1/L2 pass、L0 partial，L3 未运行，L4/L5 未通过 | 2026-10-10 |
| [Qwen2-0.5B stock `swift rlhf --rlhf_type orpo` CUDA 输入身份审计](2026-10-09-qwen2-orpo-stock-cli-cuda.md) | 两样本三步轨迹发现 step-3 顺序反转；单样本固定批次三步输入、前向、梯度和末态权重对齐；optimizer 参数组/状态一致，scheduler 独有字段不同；L0/L1/L2/L4 partial、L3 not-run、L5 blocked | 2026-10-09 |
| [Qwen2-0.5B stock `swift rlhf --rlhf_type rm` CUDA 入口审计](2026-10-09-qwen2-reward-stock-cli-cuda.md) | 原生和严格 shim 公开 CLI 均完成三步并保存 checkpoint；291 个权重键 shape 对齐，但逐步输入身份不一致、loss 差最大 0.01331；L0 partial、L1/L2 blocked、L4 入口已运行但功能未通过 | 2026-10-09 |
| [Qwen2-0.5B RewardTrainer Python API CUDA 对拍](2026-10-09-qwen2-rewardtrainer-api-cuda.md) | 固定 preference 数据三步 logits/loss/head 梯度与更新对齐，状态元数据一致且零 fallback；L0 pass，L1/L2/L4 partial，L3 未运行、L5 blocked | 2026-10-09 |
| [Qwen2-0.5B stock `swift rlhf --rlhf_type simpo` 双卡 CUDA 对拍](2026-10-09-qwen2-simpo-2gpu-cli-cuda.md) | 固定六行重复 preference 数据下双卡三步输入/梯度及 step-3→4 新进程恢复轨迹对齐；strict shim 零 fallback，跨 runtime 状态哈希不同，同进程恢复与可区分样本游标未验；L0-L4 partial，L5 blocked | 2026-10-10 |
| [Qwen2-0.5B stock `swift rlhf --rlhf_type simpo` CUDA 对拍](2026-10-09-qwen2-simpo-stock-cli-cuda.md) | 单样本三步输入、首步 logits、290 项全参数梯度和末态权重对齐且 strict CUDA/零 fallback；L0/L1/L2/L4 partial、L3 not-run、L5 blocked | 2026-10-09 |
| [JTorch 学习率调度器状态字段 CUDA 对拍](2026-10-09-torch-lrscheduler-state-cuda.md) | 修复 verbose 与瞬态 `_is_initial` 状态差异；LambdaLR/MultiplicativeLR/StepLR 的 CUDA 状态、学习率轨迹和参数值对齐；定向测试 11 passed，core tier 仍有环境与无关失败 | 2026-10-09 |
| [Qwen2-0.5B 公开双卡全参数 SFT：strict shim 第六步 backward 失败](2026-10-10-qwen2-causal-ddp-sft-cuda.md) | native 12 步完成；strict shim 在 step 6 `grad_optional` 触发 NanoVector slice overflow，前五步逐参数梯度已审计；L2 未完成、无恢复或性能结论 | 2026-10-10 |
