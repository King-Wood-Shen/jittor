# 真实 Qwen2 公开 Embedding SFT CLI：AdamW 恢复可执行，跨后端参数未对齐

基线 Jittor commit `c45e58dde3474b0965d076ff60188856b8cc9401`，ms-swift `88d7279`。所有导入、JIT、训练和比较均在 Slurm NVIDIA worker。真实缓存 Qwen2-0.5B，公开 `python -m swift.cli.main sft`，4 条固定离线 InfoNCE 组样本、batch4、float32 权重加载/eager、CLI fp16=True（非纯 FP32 训练），仅 `model.norm.weight` 可训练，AdamW lr=1e-4、weight_decay=0、constant scheduler，三步。原始脚本、日志、checkpoint 和比较在 `_state/ms-swift-cuda/20261006-qwen2-public-embedding-cli-resume-adamw`。

原生 10364 与严格 CUDA 候选 10368 均 `COMPLETED 0`：连续三步保存 checkpoint-2/3；另起公开 CLI 进程从 checkpoint-2 恢复第 3 步，保存 checkpoint-3。候选双阶段父子进程 bootstrap 起止均为 `use_cuda=1`、shim marker 真、fallback0。候选首次 10367 仅在核心 `jit_utils` 重建后要求进程重启，尚未进入训练；10368 重试通过。

10376–10380 五轮独立比较/诊断后按上限暂停。本机原生 `torch.load` 不接受候选的 portable pickle 文件，比较器改用 `pickle.load` 解码候选 optimizer/scheduler；原始失败 10376 保留。比较器随后对每个后端的连续/恢复轨迹验证：第 3 步 loss、grad norm、896 维训练参数、AdamW 一阶/二阶状态及 scheduler 在 1e-6 阈值内一致；optimizer step 均为 3，moment 非零，checkpoint 含 RNG 文件，模型 290 个键，global_step=3。此为各后端自身的状态恢复可执行证据。

跨后端未过预设参数阈值。第 3 步原生 loss `2.03301287`、候选 `2.03345919`，绝对差 `0.00044632`（小于前向阈值 0.005）；grad norm 相对差 `0.0027363`（小于 0.03）；最终 `model.norm.weight` 最大绝对差 `0.00059795`，大于固定 0.0001；AdamW `exp_avg` 最大差 `3.6794e-5`、`exp_avg_sq` 最大差 `3.1930e-7`。不能以 loss 接近或各自恢复一致代替跨后端更新对齐。10377/10378 比较日志保留；10379/10380 的逐步诊断因原始 BF16 safe_open 解码与 checkpoint-1 被保留策略清理而未产出有效逐步数值，不能据此推测具体 AdamW 算子根因。

判定：该固定公开 CLI 的构造、训练、保存、新进程恢复可执行性成立，但跨后端训练更新 L2 未通过，故公开 AdamW 状态恢复 L3 的完整数值适配仍 `blocked`；L5 `not-run`。此项到五轮暂停，不调宽阈值、不改产品源码。原始 30 个修改和 2 个未跟踪文件未触碰，完整功能面矩阵未完成。
