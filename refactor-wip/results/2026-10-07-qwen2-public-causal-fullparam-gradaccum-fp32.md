# Qwen2 公开因果 SFT CLI 全参数纯 FP32 梯度累积 2

- 基线：Jittor `8d82487761c4e138e3a7e941e317a2557bea387f`；ms-swift `88d7279`。原始脏工作树保留 30 个修改与 2 个未跟踪文件。实验目录：`/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-gradaccum-fp32`。
- 路径：原生 PyTorch 10618、Jittor torch shim 10619，均经 `python -m swift.cli.main sft` 公共分发器，在 Slurm NVIDIA worker 执行并 COMPLETED 0:0。独立比较 10623 也在 worker COMPLETED 0:0；`worker.sh` 保存精确命令、环境和输出目录。
- 输入与合同：缓存真实 Qwen2-0.5B checkpoint，四条固定离线对话；全参数 494.0328M Trainable，`torch_dtype=float32`、`fp16=false`、`bf16=false`、eager attention、SGD lr=1e-5；单卡每微批 2 条，`gradient_accumulation_steps=2`，三个 optimizer step，共 12 条样本曝光。两侧均保存 `checkpoint-3`。
- 候选主进程/CLI 子进程在启动和退出时 shim marker=true、use_cuda=1、fallback=0；`shim-fp32-bootstrap.log` 保留四个标记。原生进程独立运行且未加载 shim。GPU/节点原始信息见作业日志。两后端先后运行，未并发占用同一编译缓存。
- 独立比较三个 optimizer step：loss 绝对差依次 4.768e-7、9.537e-7、2.384e-7；grad norm 相对差依次 1.574e-6、1.027e-6、2.123e-6；token_acc 三步完全相同。最终 290 个模型张量最大绝对差 7.451e-9，整体 L2 差 5.433e-8。相对初始权重，原生/候选更新 L2 分别为 0.001200761543/0.001200761586；两侧各有 246/290 张量出现 FP32 可见变化。其余张量虽可训练，小学习率更新低于 FP32 ULP，不能据此说它们被冻结。原始对比在 `comparison3.json`、`steps-comparison.json`。
- 结论：限定真实 Qwen2、纯 FP32、全参数 SGD、单卡梯度累积 2 的三步公开 CLI 数值路径 L0/L1/L2/L4 PASS；L3 新进程累积状态恢复 not-run，L5 预热同步稳态性能 not-run。AdamW 梯度累积、BF16、LoRA、双卡累积、不同模型/数据长度 not-run。先前全参数 AdamW 间歇 core 反向故障未因此解除，不能将本项外推为全面适配。
