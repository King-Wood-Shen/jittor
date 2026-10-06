# Qwen2 公开 seq_cls CLI 全参数纯 FP32 三步训练

基线 Jittor `c6bb6f4a30b3f2d86fc85ac4055899d0619bb447`、ms-swift `88d7279`。真实 Qwen2-0.5B 基座与固定零初始化 3×896 分类头沿用先前分类头实验的分片 checkpoint，确保两后端同起点；未修改原始缓存权重。四条固定离线分类数据、公开 `python -m swift.cli.main sft`、`task_type=seq_cls`、全参数 `tuner_type=full`、显式 FP32/eager、SGD lr1e-5、单卡整批四条、三步。ms-swift 两侧报告 494.0355M 参数全部 Trainable。

原生 10635、候选 10636 均在 Slurm NVIDIA worker COMPLETED 0:0，保存 checkpoint-3；候选 CLI 父子进程各自 start/end 标记 use_cuda=1、shim marker=true、fallback=0。独立 worker 10639 比较 291 个最终权重：三步 loss 逐值相同，grad norm 最大相对差 2.323e-7，acc 三步相同；最终权重最大绝对差 1.592e-11、整体 L2 差 3.951e-11。相对固定初始权重，原生/候选更新 L2 为 2.999998618e-5/2.999998744e-5，分类头更新 L2 均约 3e-5；两侧各 174/291 张量发生 FP32 可见变化。其余张量仍属于可训练参数，小更新低于 FP32 ULP。原始脚本、训练/Slurm 日志、checkpoint 和 `comparison.json` 在 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-seqcls-fullparam-fp32`。

10638 首次独立比较器用 NumPy 直接读取 BF16 基座，报 `data type 'bfloat16' not understood`；仅将基座读取改为 worker 上的 Torch FP32 转换，10639 通过。两次训练没有重跑，未改产品源码。原始脏工作树的 30 个修改与 2 个未跟踪路径保持原样。

结论：固定分类头、真实 Qwen2、全参数、单卡纯 FP32 SGD 三步的公开 seq_cls CLI 限定 L0/L1/L2/L4 数值路径 PASS。随机头默认初始化、AdamW/LoRA/BF16、双卡全参数、完整恢复与 L5 稳态性能 not-run；不能外推整个分类任务矩阵。
