# 真实 Qwen2 公开因果 SFT CLI：双卡全参数纯 FP32 新进程恢复

基线 Jittor `9dd2506bfa6eb081ad77212e272f5b8000d2af28`、ms-swift `88d7279`。所有导入、训练、JIT、数值比较均在 Slurm NVIDIA worker。公开 `python -m swift.cli.main sft`、`NPROC_PER_NODE=2`，真实缓存 Qwen2-0.5B、四条固定离线对话、每卡 batch 2、eager、显式 float32 且关闭 fp16/bf16、全 494.0328M 参数可训练、SGD lr1e-5、constant scheduler 三步；save_steps=1。候选由原生 torchrun 控制平面启动 Jittor 双 rank，并非纯 Jittor 启动器。原始脚本、Slurm/训练日志、checkpoint、比较脚本与 JSON 均在 `_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-ddp-resume-fp32`。

原生连续 10550、新进程从 checkpoint-2 恢复 10551，候选严格 CUDA 连续 10552、恢复 10553，均 `COMPLETED 0` 且到达 checkpoint-3。两 rank 的 RNG 状态文件存在。候选主进程与两 rank 在连续及恢复的 start/end 共 12 条记录均 `use_cuda=1`、shim marker 真、fallback0。

10557 首次比较因使用 grad norm 绝对差 `1e-5` 的不合比例门槛失败；10568 独立诊断确认候选连续/恢复 grad norm 约 445，绝对差 `3.05176e-5`、相对差 `6.850e-8`，未发现权重或状态失配。改为相对差 `1e-6` 的同后端门槛后，10569 比较 `COMPLETED 0`。原生连续/恢复第三步 loss、grad norm 和 290 个模型权重完全相同；候选第三步 loss 差 `2.384e-7`，grad norm 相对差 `6.850e-8`，290 权重最大差 `1.863e-9`、全体差 L2 `6.813e-9`。两后端各自 optimizer 与 scheduler 字典完全相等，token_acc 相同。跨后端连续/恢复第三步 loss 差 `7.153e-7`/`4.768e-7`，grad norm 相对差 `9.590e-7`/`1.027e-6`，最终权重最大差均 `7.451e-9`。

判定：该固定真实 Qwen、公开双卡全参数纯 FP32 SGD 配置的新进程恢复，模型参数及有效训练轨迹的限定数值合同通过。SGD 无动量且 optimizer state 为空；只核实双 rank RNG 文件存在，未证明 RNG 内容或通用 dataloader 游标与重放语义，故不宣称完整通用 L3。纯 Jittor 启动器、AdamW/有状态优化器、BF16、其他模型/数据、L5 稳态性能均 not-run；历史公开双卡 Embedding 失败仍不变，完整适配矩阵未完成。

补充 RNG 内容审计（Slurm NVIDIA worker 10682 COMPLETED0）：对原生和严格CUDA候选各自的连续训练与新进程恢复 checkpoint-3，在rank0/rank1逐层比较python、numpy、cpu、cuda四类保存状态；八组键比较均为true，四个rank/backend组合的overall_same均为true。原生使用torch.load读取，候选按其纯pickle实际格式读取；原始audit_rng.py与slurm-rng-audit-10682.log位于同名_state目录。此证据补强固定四条数据、无动量SGD的双卡恢复状态；跨后端RNG算法等价和通用DataLoader游标仍未验证。
