# Qwen2-0.5B stock Reward Model CLI CUDA 入口审计

- 状态：原生与 strict shim 均通过公开 `swift rlhf --rlhf_type rm` CLI 完成三步训练并生成 checkpoint；但两侧每步实际 `input_ids`/mask 不一致，loss 最大差 `0.0133115`。入口已运行，功能未通过；不代表 RewardTrainer 或 ms-swift 整体兼容。
- 日期：2026-10-09。
- 基线：Jittor `dfde300e618dffcf38965b08777cae67c641476c`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B `Qwen2ForSequenceClassification`、公开单卡 `swift rlhf --rlhf_type rm`、全参数 FP32/eager、SGD、六组固定 preference 数据、batch 2、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：先隔离 ms-swift 数据构造/采样与 Torch shim dataloader 的批次身份差异；确认两侧输入一致后再评估梯度、恢复和性能层。

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

仓库门禁：Slurm 13785 的布局与 Manifest 生成/校验通过。完整 `tests/structure` 为 1383 passed、8 skipped、1 failed；唯一失败是 CPU matmul 测试需要 oneDNN v3，而 GPU worker 下载源码时到 `codeload.github.com` 连接被拒。Slurm 13720、13738 已在前序文档验证中遇到同一环境限制；本次未把完整结构门禁标记为通过。

运行键 `20261009-qwen2-reward-stock-cli-r1` 与 `20261009-qwen2-reward-stock-cli-r2` 保存脚本、checkpoint、原始输入记录、比较 JSON 和日志，均未版本化。r2 使用了上一运行的固定初始模型；其编译缓存与源码、Python ABI、编译器和设备指纹一致，且未并发写入。报告只确认上述单卡固定配置。批次身份的负责层尚未隔离，不据此擅自归因于 Jittor core、torch shim 或 ms-swift。
