# 真实 Qwen2 公开因果 SFT CLI：纯 FP32 全参数 SGD 三步数值对齐

基线Jittor `17c5e2f7d9b1ad1286673af20dae8c3a2ddce920`、ms-swift `88d7279`。所有导入、JIT、训练和比较均在Slurm NVIDIA worker。公开入口`python -m swift.cli.main sft`，真实缓存Qwen2-0.5B、四条固定离线对话整批、eager、显式`--torch_dtype float32 --fp16 false --bf16 false`、`--tuner_type full`且无冻结参数、SGD lr1e-5；ms-swift日志确认494.0328M参数全部Trainable。脚本、日志、checkpoint与比较结果位于`_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-fp32`。

单步原生10521、严格CUDA候选10522均`COMPLETED 0`，候选CLI父子进程起止`use_cuda=1`、shim marker真、fallback0。10525比较器拼接语法错误、10526原始BF16 checkpoint不能按NumPy解码，均未得到有效数值结论；修正为worker内torch BF16解码后10528独立比较通过：loss差1.1921e-6、全局grad norm相对差4.1063e-7、token_acc一致；290个保存张量逐键比较最大差7.4506e-9、整体差L2=3.3048e-8，双方对真实原始权重的更新L2约0.0004002543，均有246个张量出现FP32可见变化。

进一步原生10530、严格CUDA候选10531均完成相同配置连续三步、保存checkpoint-3；10532逐一比较290个最终张量，最大差7.4506e-9、整体差L2=5.4777e-8；两侧相对真实基座更新L2分别0.00120076154/0.00120076158，246个张量出现FP32可见变化。10534逐步日志复核：三步loss差分别9.537e-7/4.768e-6/5.245e-6，grad norm最大相对差1.233e-6，token_acc全相同，global_step3。候选父子进程全程strict CUDA、shim marker真、fallback0。其余参数虽然均设置为可训练，但在该小学习率三步中部分更新小于FP32可表示间隔；不能把246个可见变化误述为仅246个Trainable。

判定：该真实Qwen2、固定整批四条数据、公开单卡因果SFT CLI、显式纯FP32、494M全参数SGD三步的限定L4数值合同PASS。首次JIT、不同GPU工作节点及单次显存读数不构成L5。AdamW、混合精度、其它数据/模型、全参数双卡、完整checkpoint恢复和预热稳态性能仍not-run；历史已跳过LoRA问题不重启。完整适配矩阵未完成。
