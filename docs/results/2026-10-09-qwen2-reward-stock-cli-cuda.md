# Qwen2-0.5B stock Reward Model CLI CUDA 入口审计

- 状态：原生与 strict shim 均通过公开 `swift rlhf --rlhf_type rm` CLI 完成三步训练并生成 checkpoint；当前基线补验确认固定 profile 的随机训练样本顺序不同，L1/L2 功能未通过，L4 入口已运行但功能未通过。不代表 RewardTrainer 或 ms-swift 整体兼容。
- 日期：2026-10-09；当前基线补验：2026-10-11。
- 基线：历史严格对拍 Jittor `dfde300e618dffcf38965b08777cae67c641476c`；当前基线 Jittor `487935592dd6b42cc9cde4212c7713906beee819`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B `Qwen2ForSequenceClassification`、公开单卡 `swift rlhf --rlhf_type rm`、全参数 FP32/eager、SGD、六组固定 preference 数据、batch 2、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：修复或隔离 RewardTrainer 有效 sampler 的跨实现随机排列差异后，在相同输入上补 L1；按当前训练收敛协议补全两侧逐 trainable 梯度、更新与固定窗口 loss 证据，再评估恢复和性能层。

初始模型由原生 PyTorch 从固定 Qwen2 checkpoint 构造，确定性初始化 `score.weight` 后保存；同一 safetensors 被原生与候选 CLI 读取，SHA256 为 `415ca5ddc706cea55ef2a66850061291272c744bd5e1acdca6eeddc8d4035a1b`。底模文件 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。六条 preference JSONL SHA256 为 `dc072b5ee211afbb4c6e8f1a1f5689d7f4e5a98f4f79c86b70d4d39a4608eee4`。ms-swift 为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6。

Slurm 13761 在 `cscg-qh04` RTX 4090（UUID `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44`，driver `580.178.04`）先运行原生、再运行 strict shim。strict shim 使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；父子 CLI 进程均记录 `use_cuda=1`、fallback 0。两侧 CLI 均保存 `checkpoint-3`。全模型 291 个 state 键 shape 一致且 finite；末态最大绝对差 `1.0155141e-4`、相对 L2 `1.1141148e-6`。但第 2/3 步 loss 轨迹不一致，最大差 `0.0133114`。该运行没有保存实际批次输入，不能把 loss 分叉解释成模型数值误差。

Slurm 13784 在 `cscg-qh17` RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）复跑相同 checkpoint、数据与 CLI，并显式传入 `--dataset_shuffle false --train_dataloader_shuffle false`。外部审计插件仅记录 `RewardTrainer.compute_loss` 收到的 CUDA `input_ids` 与 `attention_mask`；两边各记录三步、记录中的张量都在 CUDA、输出 shape 结构相同，但逐步输入记录不相等。最终权重差和 loss 最大差与首轮基本相同（loss `0.0133115`）。候选进程仍为 strict CUDA 且 fallback 0。因为第二轮已确认数据身份未对齐，后续层的数值比较不能作为同输入对拍。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 同一确定性初始 safetensors 和 291 个末态键；shim 日志显示全参数 Qwen2ForSequenceClassification、`device_map=cuda:0`，审计到的批输入为 CUDA。缺少两侧逐参数初态 dtype/device 清单与同一批次身份。 |
| L1 | blocked | 两侧各自训练前向完成，但逐步 input IDs/mask 不同；未做同权重同输入 logits 对拍。 |
| L2 | blocked | 已比较末态权重与训练日志，但没有全部 trainable 参数逐项梯度、optimizer 状态或逐步更新对拍，且输入不一致。 |
| L3 | not-run | 没有同进程或新进程恢复轨迹。 |
| L4 | 入口已运行，功能未通过 | 原生和 shim stock 公开 RLHF CLI 均完成三步并保存 checkpoint；输入身份差异使当前功能合同未通过。 |
| L5 | blocked | 前级不通过；训练墙钟含冷 JIT，不作为性能结果。 |

## 2026-10-11 当前基线采样顺序追踪

在不重跑旧运行键的前提下，新运行键 `20261011-qwen2-reward-stock-cli-index-trace-v1` / job16181 和 `20261011-qwen2-reward-stock-cli-loader-trace-v1` / job16186，于 `cscg-qh04` 同一 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）再次依次运行 native 与 strict shim 公共 CLI。两次均完成三步；job16181 实收输入哈希与历史 r2 完全相同，job16186 同时记录 DataLoader 结构、实际 sampler index 和 loss hook 输入。候选父子进程均 `use_cuda=1`、`fallback=0`。v1 的首个 sampler hook 未挂到实际对象，没有产生索引日志；v2 从实际返回的 loader 对象包裹 batch sampler，采集有效索引。所有原始脚本、日志、JSONL 与 checkpoint 均在 state，未版本化。

job16186 中两侧 `seed=42`、`data_seed=42`、`train_dataloader_shuffle=false`，但有效 loader 使用 Accelerate `SeedableRandomSampler`；实际 sampler 长度为 5。原生索引依次为 `[2,4]`、`[3,0]`、`[1]`，strict shim 为 `[4,2]`、`[3,1]`、`[0]`，且每步 `compute_loss` 实收 CUDA `input_ids`/mask 的哈希仍不相等。原 JSONL 有 6 行而 sampler 只迭代5个索引；其 data source 长度与哪条记录在 lazy dataset 构造时被排除尚未直接采集，作为未定位边界保留。两侧 sampler 输出的索引集合均为 `0..4`，顺序不同已足以解释批次身份不一致。

静态源码追踪显示 `swift.rlhf_trainers.reward_trainer.RewardTrainer` 的基类为 `RLHFTrainerMixin, SwiftMixin, HFRewardTrainer`；它没有继承定义自有 `train_dataloader_shuffle` sampler 的 `DataLoaderMixin`，所以 CLI 保存的 `false` 未阻止有效 loader 的随机采样。Accelerate `SeedableRandomSampler` 对两侧使用同一 data seed，并经 `torch.randperm(generator=...)` 取样；shim 的 `torch.Generator` stream 当前由 `numpy.random.default_rng` 提供，`randperm` 使用 `Generator.permutation`。这解释了同 seed 下实测索引排列差异。该路径同时暴露 ms-swift RewardTrainer 的 shuffle 参数未作用于此继承链，以及 shim `Generator`/`randperm` 未复刻 PyTorch 的 CPU 随机序列；本轮只记录责任边界，未修改 ms-swift 或 shim 源码。

等级按 2026-10-10 新训练协议复核：L0 partial（原生/候选构造和 CUDA 参数/输入可运行，仍缺完整两侧初态状态清单）；L1 blocked（输入不一致，未做同权重同输入前向）；L2 blocked（尚无逐 trainable 参数梯度与完整 optimizer 状态审计，也没有按固定窗口验证两侧 loss 下降；数值逐值对拍本身不再是门槛）；L3 not-run；L4 入口已运行、功能未通过；L5 blocked。当前基线结论不覆盖其它 RewardTrainer、RM 数据格式、采样设置或 ms-swift 训练面。

仓库门禁：Slurm 13785 的布局与 Manifest 生成/校验通过。完整 `tests/structure` 为 1383 passed、8 skipped、1 failed；唯一失败是 CPU matmul 测试需要 oneDNN v3，而 GPU worker 下载源码时到 `codeload.github.com` 连接被拒。Slurm 13720、13738 已在前序文档验证中遇到同一环境限制；本次未把完整结构门禁标记为通过。

运行键 `20261009-qwen2-reward-stock-cli-r1`、`20261009-qwen2-reward-stock-cli-r2`、`20261011-qwen2-reward-stock-cli-index-trace-v1` 与 `20261011-qwen2-reward-stock-cli-loader-trace-v1` 保存脚本、checkpoint、原始输入、索引和日志，均未版本化。r2 使用上一运行固定初始模型；追踪作业顺序复用与源码/ABI/工具链/设备架构相符的缓存。报告只确认上述单卡固定配置。
