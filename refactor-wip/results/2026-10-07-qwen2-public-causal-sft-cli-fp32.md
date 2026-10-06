# Qwen2 公开因果 SFT CLI：显式 FP32 通过，默认混合精度边界

基线 Jittor `8d9a2abd066dd6311c4c4df7b76d303c7ea6e48a`，ms-swift `88d7279`；全部导入、JIT、计算和比较在 Slurm NVIDIA worker。入口是 `python -m swift.cli.main sft`，真实缓存 Qwen2-0.5B、四条离线对话样本、batch4、eager attention、full tuner 冻结基座仅训练 `model.norm.weight`，SGD 0.01 三步，保存完整 checkpoint。原始命令、日志、checkpoint、比较 JSON 位于 `_state/ms-swift-cuda/20261007-qwen2-public-causal-sft-cli`。

最初仅指定 `--torch_dtype float32` 时，CLI 实际解析 `fp16=True`。原生 10382、候选 10383 均 `COMPLETED 0`，候选严格 CUDA、shim 标记真、fallback0；但 10387 比较三步 loss 最大差 `0.00726342`，超过固定阈值 `0.005`，而 grad norm 最大相对差 `0.00456`、参数最大差 `2.19345e-5`、token_acc 完全一致。10391/10392 关闭 `use_logits_to_keep` 的对照仍有候选恒定 `3.82421875` 的 loss，故该特性不是必要原因；两侧原始训练配置是混合精度，不能将此失败归为纯 FP32。

显式加 `--fp16 false --bf16 false` 后，原生 10393 和候选 10394 均 `COMPLETED 0`，CLI 日志证实两项均 false。10395 独立 worker 比较通过：三步 loss 最大绝对差 `9.53674e-7`，grad norm 最大相对差 `6.88503e-6`，token_acc 精确相同，896 维训练权重最大绝对差 `3.91155e-8`；两侧相对原始权重最大更新均 `0.00189018`，非零；290 个模型状态键、global_step3，optimizer/scheduler/RNG 文件存在。候选 CLI 父子进程启动和结束均 strict CUDA、shim marker 真、fallback0。首次编译耗时及训练总历时不作为 L5 稳态数据。

判定：此固定真实 Qwen2、公开因果 SFT、显式 FP32、单归一化层训练三步的 L4 数值合同 PASS。默认 `fp16=True` 混合精度路径未过同一 loss 阈值，应单列未解决；不外推全参数、LoRA（历史五轮跳过）、BF16、其他数据/模型、checkpoint 恢复、双卡或 L5。此前公开 Embedding CLI 报告中 `torch_dtype=float32` 却 `fp16=True` 的项目已更正为权重加载与训练精度分开表述，原始对照结论仅适用于其真实 CLI 设置。
