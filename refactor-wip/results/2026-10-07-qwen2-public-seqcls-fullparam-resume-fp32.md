# Qwen2 公开 seq_cls 全参数纯 FP32 新进程恢复

基线 Jittor `4865e82151be7f9fb27b4430e9e860d1f0fd4794`、ms-swift `88d7279`；真实缓存 Qwen2-0.5B 固定零初始化 3×896 分类头，公开 `python -m swift.cli.main sft`、`task_type=seq_cls`、494.0355M 全参数、显式纯 FP32/eager、SGD lr1e-5、四条固定离线分类数据整批三步。两后端各做连续三步和从 checkpoint-2 另起 CLI 进程恢复第 3 步，save_steps=1。原始脚本、Slurm/训练日志、checkpoint、独立比较 JSON 在 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-seqcls-fullparam-resume-fp32`。

原生 10641、严格 CUDA 候选 10642、独立比较 10643 均在 Slurm NVIDIA worker COMPLETED 0:0。候选连续和恢复阶段各 CLI 父子进程 start/end 共 8 条标记，均 use_cuda=1、shim marker=true、fallback=0。原生连续/恢复第 3 步 loss、grad norm、acc、291 权重精确相同；候选第 3 步 loss、grad norm、acc 相同，权重最大差 1.735e-18、整体 L2 差 5.968e-18。两后端各自 optimizer/scheduler 字典相等。跨后端连续/恢复权重最大差均 1.592e-11，整体 L2 差约 3.951e-11，loss 相同、grad norm 相对差 1.163e-7。

判定：该固定头真实模型、单卡全参数纯 FP32 无动量 SGD 的公开分类 CLI 第三步新进程恢复数值合同通过。SGD optimizer state 为空，只核实 RNG 文件存在；未核验 RNG 内容、随机初始化头或通用 DataLoader 游标重放，不能称完整通用 L3。AdamW/LoRA/BF16、双卡恢复、L5 预热同步稳态性能 not-run。未修改产品源码，原始 30 个脏修改与 2 个未跟踪路径保留，完整适配矩阵未完成。
