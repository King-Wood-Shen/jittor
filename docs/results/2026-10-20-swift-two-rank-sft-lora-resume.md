# ms-swift 双卡 CUDA SFT/LoRA 与新进程恢复

- **状态：** 通过，附限制
- **日期：** 2026-10-20
- **Jittor 基线：** `87daee79b757c77ebb9283a66f4f80904f2cc8b5`（同步 `origin/2.0-refactor` 到 `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64` 后）
- **ms-swift：** `modelscope/ms-swift` `88d727951203256baa564c643c651b6f8d90fd7e`
- **维护者：** Jittor CUDA / Torch compatibility
- **复查条件：** Swift/Jittor launcher 协议、Torch shim optimizer 或 NCCL 初始化审计改变时重跑

## 验证范围

在单机两张 CUDA 卡上经公开 `swift.cli.main sft` 执行 Qwen2 0.5B FP32 LoRA。数据集是 16 条固定 lookup 样本，每 rank 每步 2 条，全局 batch 4。连续运行 16 步；独立运行保存第 8 步检查点，再由新的两 rank 进程恢复到第 16 步。只验两卡代表多卡，不包含 CPU SFT、其他卡数、其他模型/训练器或速度验收。

主线 Swift 的多进程 CLI 会调用 `torch.distributed.run`，该模块不在当前 Jittor Torch shim 中。保留公开 Swift SFT 命令，由 `jittor.distributed.launch --backend=nccl --nproc-per-node=2` 启动两 rank，并设 `SWIFT_SINGLE_DEVICE_MODE=1`，让 Swift 每个只看见一张卡的进程按单设备模式配置。这个启动边界与 workaround 记在 [KI-DIST-004](../../agent/manuals/known-issues.md#ki-dist-004-swift-cli-multiprocess-launcher-is-not-provided-by-the-jittor-torch-shim)。

训练参数为 `--tuner_type lora --lora_rank 4 --lora_alpha 8 --torch_dtype float32 --attn_impl eager --max_length 64 --optim adamw_torch --learning_rate 0.001 --per_device_train_batch_size 2 --gradient_accumulation_steps 1`，连续/分段/恢复目标步数分别为 16、8、16。离线缓存模型 SHA256：config `b2388df676bbbe4f6aae6b617bdfa6e7c471c7a5b23862baae3dfbf85a5fe9ef`，weights `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，tokenizer `f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8`。CUDA toolkit 为 12.2，设备编译目标 `sm_89`。

## 结果

- 三段作业的 launcher 日志都记录 rank 0/1、world size 2、NCCL backend；每 rank 的 Jittor NCCL 初始化日志均为成功。训练进程实际通过 Torch shim 发起 17 次（连续）和 9 次（分段、恢复）all-reduce。每个实际优化步骤的 rank 间梯度 hash 一致。
- continuous/split/resume 每 rank 分别观察到 16/8/8 个 CUDA batch，每个 rank 的 batch 有两条样本；每一步两 rank 样本 hash 不重叠。恢复运行第 8 至 15 步的每 rank batch hash 与连续运行相同。
- 六个实际训练 worker 均以 `backend_fallback="error"` 退出，fallback 数为 0，CUDA 已启用。训练期间 NVML 峰值：GPU UUID `GPU-920e2bd4-2afd-93c8-64f4-c55dd3568e64` 为 **4,870 MiB / 24,564 MiB**；UUID `GPU-0a3bba05-32f4-4794-0fd0-8267f4bb390e` 为 **4,403 MiB / 24,564 MiB**。
- 40 份实际 AdamW 梯度快照（每 rank 20 份）全部有限、非零，且对应 rank 的 hash 相同。Swift 的 `_fix_grad_norm_nan` 在 grad norm 为 NaN 时清空参数梯度并跳过权重更新；因此这不等价于声称 32 个 batch 都产生了优化器更新。连续轨迹的 loss 记录均有限，前四步平均 **7.5065**，末四步平均 **2.2667**。
- 三个可恢复检查点及 optimizer/scheduler/RNG 状态均已发布：continuous 第 16 步、split 第 8 步、resume 第 16 步。恢复使用了 split 检查点，使用了新的两 rank PID，并正常写出第 16 步 Trainer 状态。

## 限制与解释

`sitecustomize` 中针对 Python `torch.distributed.init_process_group` 的观测 hook 没有触发：本路径由 Jittor launcher 先建立 native NCCL group。原临时 driver 只接受该 Python hook 和 `Var.mpi_all_reduce` 计数，训练结束后错误地拒绝了合法路径（它观察到了 0 个上述事件，但各 rank 有 17/9 次 Torch shim all-reduce）。离线复核改按 launcher rank、Jittor NCCL 初始化、Torch shim all-reduce、rank 间梯度 hash 和 fallback 计数验收；复核脚本与机器日志保存在 `$JITTOR_LAB_ROOT/_state/msw-cuda-parallel/20261020-swift-sft-lora-two-rank-87daee79-v12/evidence/`。因此原 driver 的退出码不是训练失败，也不作为通过证据。

continuous 与 split 是两个独立初始化的 Swift run；末端 LoRA adapter 的 2,199,552 个元素与独立 continuous adapter 不相同。该任务验收新进程加载完整 checkpoint 并恢复至第 16 步，不宣称独立初始化轨迹逐位相同，也不把这一比较当成恢复失败。要验严格轨迹等价，需先固定 adapter 创建前的 RNG 状态，再设计单一初始权重的 continuous/split 配对实验。

正式训练没有运行 CPU 适配或其他卡数穷举。同步后已复用同一基线上的 CUDA 回归证据：`tests/ops/test_binary_broadcast_construction.py::TestBinaryBroadcastConstruction::test_cuda`，1 passed，22 subtests passed。提交前 `bash tools/check_repo_layout.sh` 通过（301 个 active Markdown 文件），`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 通过（1512 passed、8 classified skips、1058 subtests passed）；该结构门禁环境没有 NVCC，未执行 CUDA 训练。
