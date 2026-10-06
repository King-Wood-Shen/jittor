# ms-swift CUDA 默认 fused AdamW 二阶矩精度

## 结论

先前真实 Qwen2-0.5B RewardTrainer 默认 AdamW 的二阶矩不满足独立 FP64 公式：首步最大绝对误差 2.71e-5，高于固定门槛 1e-5；已验证根因为 CUDA fused kernel 在计算 `1-beta2` 前把 `beta2=0.999` 舍入为 float32。此轮在第五轮源码修复中保留原始双精度 beta，按块计算补系数和 bias correction，再转为 kernel 所用的 float。默认 fused 路径仍启用。

本轮真实 Qwen2-0.5B FP32/eager 奖励模型经 ms-swift 公共加载、私有 RewardTrainer、非零 margin 和奖励中心化，在 score.weight 上完成三步默认 AdamW。原生 PyTorch CUDA 先跑，Jittor torch shim 严格 CUDA 后跑。双方各 30 字段对齐；独立 FP64 奖励损失 6 项和 AdamW 一阶矩、二阶矩、参数共 18 项全部通过。候选二阶矩最大公式绝对误差 2.0783e-7，显著低于原定 1e-5；global_step=3，fallback=0。

## 运行键和原始证据

- 基线：集成分支 `integration/ms-swift-cuda-upstream-71eff-20261006`，提交前 HEAD `c4f4f2ac7`；ms-swift checkout 位于 `/home/xinshen/projects/ms-swift-cuda/ms-swift`。原 30 个脏文件补丁及 2 个未跟踪文件未触碰。
- Slurm 9970 首次测试入口把原生与 shim pytest 混合，被模式检查拒绝；9976 和 9978 是测试入口导入路径错误。三者未作为数值失败或通过。修正外部测试入口后，9981 COMPLETED 0：原生 fused AdamW CUDA 测试 2 项、兼容层默认 fused 路径 1 项均通过，两个进程各 fallback=0。
- Slurm 9971 COMPLETED 0：cscg-qh15，RTX 4090，GPU-afd56a2e-3deb-3bd7-f122-1075e6de8956，driver 580.178.04。真实 native 与 shim 分进程，独立 JITTOR_HOME，严格 CUDA `backend_fallback="error"` 和 `forbid_backend_fallbacks()`，无 Ascend。
- 原始脚本、Slurm 日志、native/shim 日志及 NPZ/JSON、comparison.json、reward-loss-contract.json、adamw-contract.json：`/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261006-fused-adamw-beta/`。
- 测试中的冷 JIT 与模型初始化耗时不能作为 L5 稳态性能证据。

## 覆盖边界与后续

仅证实该真实 Qwen 奖励模型、FP32 单评分头、固定三步、默认 fused AdamW 的 L0-L2 子路径。全参数、LoRA、BF16、完整 checkpoint 恢复、公开 CLI、分布式及稳态性能仍未覆盖；历史已跳过的 GKD BF16 梯度累积、TinyLlama streaming logprob 和 RewardTrainer BF16 不因本结果重新启动。全 ms-swift L0-L5 矩阵未完成。
