# Qwen2 公开 seq_cls 双卡全参数纯 FP32 新进程恢复

基线 Jittor `b63414b565e9703bb08df999e26e560d3abb8d69`、ms-swift `88d7279`。真实缓存 Qwen2-0.5B、固定零初始化 3×896 分类头、四条离线分类数据、公开 `python -m swift.cli.main sft`、双 RTX 4090/NCCL、每卡 batch2、494.0355M 全参数、显式纯 FP32/eager、无动量 SGD lr1e-5、三步。保存每步 checkpoint；原生和候选分别完成连续训练与从 checkpoint-2 另起 CLI 进程恢复第三步。候选仍由原生 torchrun 控制面启动 Jittor rank，并非纯 Jittor launcher。

原生 10659、候选 10660、独立比较 10661 均在 Slurm NVIDIA worker COMPLETED 0:0。候选两个阶段的 main、rank0、rank1 共 12 条 start/end 标记均 shim marker=true、use_cuda=1、fallback=0；两个 rank 的 RNG 文件以及 optimizer/scheduler 文件存在。原生连续/恢复第三步 loss、grad norm、acc、291 权重精确相同；候选对应三项日志指标完全相同，291 权重最大恢复差 3.469e-18、整体 L2 差 8.888e-18。两侧各自 optimizer/scheduler 字典相等。跨后端连续/恢复的第 3 步 loss 相同、grad norm 相对差 2.326e-7，最终权重最大差均 9.095e-12、整体 L2 差约 3.298e-11。原始脚本、Slurm/训练日志、checkpoint、`resume-comparison.json` 在 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-seqcls-fullparam-ddp-resume-fp32`。

判定：该固定头、固定数据、双卡纯 FP32 全参数无动量 SGD 公共分类 CLI 的第三步新进程恢复有效数值轨迹通过。optimizer state 为空，原实验只核实双 rank RNG 文件存在；后续独立审计见下。通用 DataLoader 游标/随机顺序重放仍未核验，不宣称完整通用 L3。纯 Jittor launcher、AdamW/LoRA/BF16、随机头、L5 预热同步稳态性能 not-run。未改产品源码，原始 30 个脏修改与 2 个未跟踪路径保留，完整矩阵未完成。

补充 RNG 内容审计（Slurm NVIDIA worker 10674/10676）：10674 用原生 torch.load 读取候选纯 pickle RNG 文件发生格式错误，原生两个 rank 已显示四类状态完全相同；10676 按候选实际序列化格式用 pickle 读取后 COMPLETED0，逐层比较字典/序列/NumPy 数组/张量，原生与候选各自 rank0、rank1 的 checkpoint-3 连续训练与新进程恢复状态均 overall_same=true；四键 python/numpy/cpu/cuda 均 true。候选文件不是原生 torch.save 格式，不能以原生 torch.load 打开；该审计证明各后端各 rank 的这四类保存状态值相同，不证明跨后端 RNG 算法等价或实际 DataLoader 样本游标重放。脚本与原始输出在同一 _state 路径的 audit_rng.py、slurm-rng-audit-10674.log、slurm-rng-audit-10676.log。
