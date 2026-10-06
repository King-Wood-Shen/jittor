# Qwen2 公开 Embedding SFT CLI：显式 FP32 单卡合同

基线 Jittor `4252547c8b43bd90f882f50d120aa3a5b302dc33`、ms-swift `88d7279`。原生 10400（cscg-qh13 RTX 4090）、候选 10402（cscg-qh04 RTX 4090）均在 Slurm NVIDIA worker 执行真实缓存 Qwen2-0.5B 的 `python -m swift.cli.main sft` 公共入口。四条固定离线不同语义的 InfoNCE 组、整批 batch4、eager、full tuner 冻结基座只训练 `model.norm.weight`，SGD 0.01 三步。与旧默认实验不同，此次显式传 `--torch_dtype float32 --fp16 false --bf16 false`，两侧 args.json 证实训练精度标志均 false。原始脚本、日志、checkpoint、比较 JSON 位于 `_state/ms-swift-cuda/20261007-qwen2-public-embedding-sft-cli-fp32`。

10400/10402 均 `COMPLETED 0`、保存完整 checkpoint-3。独立 worker 10419 复核：三步 loss 最大绝对差 `4.05312e-6`（固定阈值 0.005），grad norm 最大相对差 `5.15518e-6`（阈值 0.02），896 维 `model.norm.weight` 最终逐值相同，两侧相对原始 BF16 checkpoint 的训练变化 L2 均 `0.000585909`；冻结的 q_proj 权重精确相同，290 个模型状态键，global_step3，optimizer/scheduler/RNG 文件齐全。候选 CLI 父子进程 start/end 均 strict CUDA、shim marker 真、fallback0。10401 首次 JIT 重建要求进程重启，尚未训练；10418 比较器文件名错误，修复测试脚本后 10419 通过，两次无产品源码修改。

判定：该固定真实模型、公开 Embedding SFT、单归一化层 SGD、整批样本、显式纯 FP32 三步的 L4 数值合同 PASS。首次 JIT 和不同 GPU UUID 不用于 L5 性能归因；半批数据重排、全参数、LoRA、其他优化器、分布式、公开 checkpoint 恢复及 L5 保持独立边界。原始脏工作树的 30 个修改及 2 个未跟踪文件未触碰。
