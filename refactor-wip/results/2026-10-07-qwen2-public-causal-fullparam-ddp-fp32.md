# 真实 Qwen2 公开因果 SFT CLI：双卡全参数纯 FP32 SGD 数值对齐

基线Jittor `fac25ee2960f5906469219dc64d029742119cea5`、ms-swift `88d7279`。仅在Slurm NVIDIA worker导入、JIT、训练和比较。公开入口`python -m swift.cli.main sft`、`NPROC_PER_NODE=2`，真实缓存Qwen2-0.5B、四条固定离线对话、每卡batch2、eager、显式`--torch_dtype float32 --fp16 false --bf16 false`、`--tuner_type full`且无冻结参数、SGD lr1e-5；两侧模型日志均确认494.0328M参数全部Trainable。候选使用原生torchrun控制平面启动Jittor rank子进程，不能宣称纯Jittor启动器兼容。原始脚本/日志/checkpoint/比较JSON位于`_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-ddp-fp32`。

单步原生10535、严格CUDA候选10537及独立比较10540均`COMPLETED 0`：loss差4.768e-7，grad norm相对差1.779e-6，token_acc一致；290个最终权重逐键最大差7.451e-9、全体差L2=3.206e-8，双方相对真实基座更新L2约0.0004002543；246个张量出现FP32可见变化。两rank RNG文件均存在。

延长三步，原生10541、候选10542、全部权重比较10544、逐步日志比较10546均`COMPLETED 0`。逐步loss最大差1.192e-6，grad norm最大相对差2.191e-6，token_acc全相同；最终290个权重最大差7.451e-9、全体差L2=5.416e-8，双方相对原始基座更新L2约0.0012007615/0.0012007616，246个张量出现FP32可见变化。其余参数均设置为Trainable，小学习率短训练中部分更新未跨越FP32可表示间隔。候选主进程与两rank起止均`use_cuda=1`、shim marker真、fallback0。

判定：固定真实Qwen、公开因果SFT CLI、原生torchrun控制平面+Jittor双rank、显式纯FP32、494M全参数SGD三步的限定L4数值合同PASS。不能外推AdamW/有状态优化器、混合精度、其他模型与样本、纯Jittor启动器、双卡全参数恢复或预热稳态L5；历史公开双卡Embedding数值失败仍不变。完整适配矩阵未完成。
