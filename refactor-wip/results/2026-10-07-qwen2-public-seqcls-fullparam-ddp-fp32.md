# Qwen2 公开 seq_cls CLI 双卡全参数纯 FP32 三步训练

基线 Jittor `ec2d8acf56a971cad4471689bd81f488dae78ab9`、ms-swift `88d7279`。固定零初始化 3×896 分类头的真实缓存 Qwen2-0.5B、四条固定离线分类数据、公开 `python -m swift.cli.main sft`、`NPROC_PER_NODE=2`、每卡 batch2、`task_type=seq_cls`、全参数 494.0355M Trainable、显式纯 FP32/eager、SGD lr1e-5、三步并保存 checkpoint-3。两侧均由 ms-swift CLI 启动双 RTX 4090/NCCL；候选 torchrun 为原生控制面，rank 是 Jittor torch shim，不代表纯 Jittor launcher。

原生 10654、候选 10655、独立比较 10656 均在 Slurm NVIDIA worker COMPLETED 0:0。候选 main 及 rank0/rank1 的 6 条 start/end 标记均 shim marker=true、use_cuda=1、fallback=0；原生 torchrun 控制面另记录在 `control-plane.log`。独立比较逐步三次 loss 完全相同，grad norm 最大相对差 2.326e-7，acc 相同；291 最终权重最大绝对差 9.095e-12、整体 L2 差 3.298e-11。原生/候选相对固定初始权重更新 L2 分别为 2.999998469e-5/2.999998632e-5；两侧各 174 张量出现 FP32 可见变化，其余仍属于可训练参数。两 rank 各自 RNG 文件及 optimizer/scheduler checkpoint 文件存在。原始脚本、日志、模型与 `comparison.json` 在 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-seqcls-fullparam-ddp-fp32`。

判定：固定头真实模型、双卡全参数纯 FP32 无动量 SGD 的公开 seq_cls CLI 三步数值路径限定 L0/L1/L2/L4 PASS。未验证纯 Jittor launcher、双卡新进程恢复、随机分类头、AdamW/LoRA/BF16 或 L5 预热同步稳态性能；不能外推完整分类任务矩阵。未改产品源码，原始 30 个修改加 2 个未跟踪补丁保留。
