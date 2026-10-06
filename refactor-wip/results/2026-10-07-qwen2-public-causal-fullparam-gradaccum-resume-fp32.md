# Qwen2 公开因果 SFT 全参数纯 FP32 梯度累积 2 新进程恢复

基线 Jittor `ce2975114547bd53003a9c5ed426b9ecad46512f`，ms-swift `88d7279`。所有导入、JIT、训练、数值比较在 Slurm NVIDIA worker；原始脚本、Slurm 日志、训练日志、checkpoint、独立比较脚本与 JSON 位于 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-gradaccum-resume-fp32`。原始 30 个修改和 2 个未跟踪补丁保留。

真实缓存 Qwen2-0.5B，经 `python -m swift.cli.main sft` 公开入口，四条固定离线对话、单 GPU、494.0328M 全参数 Trainable、显式纯 FP32/eager、每微批 2 条、累积 2、SGD lr1e-5、constant scheduler、三步。保存每一步 checkpoint，分别做连续三步，以及新进程从 `checkpoint-2` 恢复第 3 步；无代码修改。原生 10629、候选 10630、独立 worker 比较 10631 均 COMPLETED 0:0；候选两个 CLI 阶段父子进程共 8 条 start/end 标记均 use_cuda=1、shim marker=true、fallback=0。

独立比较 `resume-comparison.json`：原生连续/恢复第 3 步 loss、grad norm 与 290 个权重完全相同；候选 loss 差 2.384e-7，grad norm 相同，290 权重最大差 3.725e-9、整体 L2 差 7.187e-9。各后端连续/恢复的 optimizer 与 scheduler 字典分别相等，token_acc 相同。跨后端连续/恢复第 3 步 loss 差 1.192e-6/1.431e-6，grad norm 相对差均 1.027e-6，最终权重最大差均 7.451e-9、整体 L2 差约 5.51e-8。各 checkpoint 的 RNG 文件存在，global_step=3；比较只使用第 3 步日志，恢复进程累计 train_loss 因分母口径不同不用于判定。

判定：固定真实模型、纯 FP32 全参数无动量 SGD、梯度累积 2 的公开单卡 CLI 模型与有效训练轨迹新进程恢复数值合同通过。该 optimizer 无动量且状态为空；RNG 文件内容、可变样本顺序下的数据游标与精确重放尚未证明，因此不宣称通用 L3。AdamW/BF16/LoRA/双卡累积恢复 not-run；L5 预热同步稳态性能 not-run。先前全参数 AdamW 间歇 core 反向故障不因本实验解除，完整适配矩阵未完成。
