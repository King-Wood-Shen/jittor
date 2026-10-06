# 真实 Qwen2 RewardTrainer checkpoint 状态恢复

- 状态：限定路径 L3 PASS；全 ms-swift 矩阵未完成。
- 基线：Jittor `700070d58`，ms-swift `88d7279`，Python 3.11.15，真实缓存 Qwen2-0.5B FP32/eager seq_cls 模型；仅 `score.weight` 可训练，固定六组配对样本、非零 margin、奖励中心化、batch 2、默认 fused AdamW、常数学习率。
- 运行：所有导入、JIT、训练、保存、读取与比较均在 Slurm NVIDIA RTX 4090 worker。原生 PyTorch CUDA 先跑，shim 后跑；候选全程 `backend_fallback="error"`、`forbid_backend_fallbacks()`，各训练阶段 fallback 0。原工作树 30 个脏文件补丁及 2 个未跟踪文件未触碰。
- 设备索引：9988/9990 在 cscg-qh15 RTX 4090 GPU-afd56a2e；10013 在 cscg-qh15 GPU-a7a0606f；10015 在 cscg-qh06 GPU-802213e4；10010、10016 分别在 cscg-qh13、cscg-qh04；GPU UUID 与 driver 580.178.04 见各自 Slurm 日志。

## 验收协议与结果

连续三步训练分别与「两步保存至 checkpoint-2，另起进程恢复第三步」及「同进程两步后恢复第三步」比较。原生和候选各自记录三步的真实输入、mask、margin、奖励 logits、损失、全部可训练梯度、参数、AdamW 一阶/二阶矩与 step；保存文件包含 `model.safetensors`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`。恢复后的第三批输入为 `[5,6,13,14]`，global_step=3，两侧 scheduler 与 Python/NumPy/Torch CPU/CUDA RNG 末态一致。

- 原生新进程：9988 COMPLETED0；9991 在 worker 独立比较 30 字段逐值一致。
- 候选新进程：9990 COMPLETED0；10010 按既定 CUDA 前向 `5e-3`、反向 `2e-2` 门槛完成 60 项内部轨迹、60 项原生对照，并以保存前状态和恢复步真实梯度按 FP64 AdamW 公式复核 6 项，全部通过。候选 fallback 0。
- 原生同进程：10013 COMPLETED0；候选同进程：10015 COMPLETED0、fallback 0；10016 在 worker 完成 60 项内部轨迹、30 项原生对照和 6 项独立 FP64 AdamW 更新复核，全部通过。
- 10020 在 worker 读取六个真实 `model.safetensors` 的第 2 步快照：共 291 个状态张量、290 个冻结张量在连续/拆分及原生/候选之间逐值一致；`score.weight` 最大绝对差异 `9.313225746e-10`。键集合、shape、dtype 一致。
- 10007 的探索性 `rtol=atol=1e-7` 全字段逐位近似比较失败，首差在保存前两次独立候选运行的第 0 步梯度（最大绝对 `1.211e-4`）；10008 诊断证实保存前最大缩放差异约 `2.685e-6`，恢复后最大缩放差异约 `6.823e-7`，恢复未放大波动。没有把 10007 记为 PASS，也没有按它放宽已声明的跨框架数值门槛。L3 结论是既定数值合同内的轨迹对齐，不声称 bitwise 重现。

## 证据与边界

原始 probe/worker/审计脚本、Slurm 日志、各阶段 NPZ/JSON、`resume-diagnostic.json`、`l3-contract.json`、`l3-same-process-contract.json` 和 `checkpoint-model-audit.json` 均在未版本化目录 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261006-qwen2-reward-checkpoint-l3/`。未用首次 JIT 或训练墙钟声称 L5 性能。

本结果只覆盖公开加载的 Qwen2-0.5B、公有 seq_cls 奖励模型、ms-swift 私有 RewardTrainer、FP32 单评分头 AdamW 固定三步的同/新进程完整状态恢复。未验证全参数、LoRA/adapters、BF16、其他模型/模板、分布式恢复、公开 CLI 与 L5；这些项目维持 not-run 或已有失败状态，不能外推为全矩阵 L3 PASS。

## 提交前验证

10042 在 Slurm cscg-qh06 RTX 4090 GPU-c7472e09 上执行仓库布局检查与文档/API 结构门禁：布局通过，38 个测试与 40 个子测试通过。10028 同节点全量结构门禁已运行至约 60%，因单个长时用例停留超过七分钟而主动取消；该全量门禁不记为通过，也不据此扩大 L3 结论。
