# Qwen2-0.5B ms-swift 私有 Swift LoRA 公开 `swift sft` CUDA 对拍

- 状态：固定配置的公开 CLI 原生与严格 shim 三步训练通过；只覆盖本报告中的 Qwen2-0.5B/FP32/eager/Swift LoRA 配置，不代表其他模型、tuner 或 ms-swift 整体兼容。
- 日期：2026-10-08。
- Jittor 基线：`d48af778b5f20fab39e8901241299b2be43a22d9`；上游 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是其祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0。Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；固定四条数据 SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Swift tuner、Trainer、Torch shim 的 CUDA/autograd/AdamW 路径变化时；完成公开入口 checkpoint 恢复与稳态性能协议时。

恢复补测运行键 `20261008-qwen2-swiftlora-sft-resume-d48-r1`，Slurm 13426 在同一 RTX 4090、同一模型/数据与已完成的 `checkpoint-3` 上尝试公开 CLI 新进程恢复。原生 PyTorch 在 Trainer 初始化后、首步前失败；未启动 shim，未执行任何额外训练。checkpoint 文件包括 `adapter_model.safetensors`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth` 和 `trainer_state.json`，没有 Transformers base-model 分片索引。Transformers 4.57.6 的 `Trainer._load_from_checkpoint` 未将 ms-swift 私有 SwiftLoRA 模型识别为 PEFT 模型，落入普通分片模型加载分支并因缺少 `model.safetensors.index.json` 抛出 `ValueError`。这是该固定配置的新进程恢复路径失败证据；不能据此断言其他 Swift tuner 或 checkpoint 格式。原始日志、事件和采集器保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-swiftlora-sft-resume-d48-r1/`。此面不修改 ms-swift 源码绕过公开入口行为；待上游恢复路径可加载 SwiftLoRA adapter 后再复测。

运行键 `20261008-qwen2-swiftlora-sft-cli-d48-r2`，Slurm 13380 在 cscg-qh17 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`，driver `580.178.04`）先执行原生 PyTorch CUDA，再执行严格 shim。公开命令为 `python -m swift.cli.main sft`，启用 `--use_swift_lora true --tuner_type lora`，rank 8、alpha 32、`q_proj/v_proj`、dropout 0；Qwen template、四条本地数据、FP32/eager、batch 4、长度 64、无 shuffle、seed 1234、AdamW（lr `1e-3`、weight decay `0.01`）、三步并保存 `checkpoint-3`。使用直接 Swift LoRA 梯度对拍的原生 96 个 adapter 张量作为两侧相同初态。原生侧没有 shim 标记；候选的父子 CLI 进程均有 shim 标记、`use_cuda=1`、`backend_fallback='error'`，结束 fallback 计数均为 0。模型所有 386 个参数在 Trainer 构造与首个 forward 时驻留 CUDA。

两侧首批 `input_ids`、`labels`、`attention_mask` 完全相同，shape `[4,31]`；首批 logits shape `[4,31,151936]`，最大绝对差 `1.32859e-4`、相对 L2 `3.15184e-6`；首批 loss 原生 `3.82736683`、shim `3.82736659`。三步 loss 原生 `[3.82736683, 1.13082719, 0.27699450]`，shim `[3.82736659, 1.13082838, 0.27699441]`。每步均审计 96/96 adapter 梯度、更新后参数、AdamW `exp_avg` 与 `exp_avg_sq`；所有张量 finite 且在 CUDA。三步最坏类别聚合相对 L2 分别为：梯度 `4.79e-6 / 6.15e-6 / 1.65e-5`，参数 `6.95e-6 / 2.69e-5 / 2.73e-5`，`exp_avg` `4.80e-6 / 4.84e-6 / 9.87e-6`，`exp_avg_sq` `6.37e-6 / 7.31e-6 / 1.17e-5`。两侧保存的 checkpoint 都有 96 个 adapter 键、`global_step=3`；adapter 最大绝对差 `2.54433e-4`。

shim 训练时长 `356.256 s`，包含首次 JIT 编译；原生 `4.1085 s`。该数值不是稳态对比，未做规定的预热与十次测量，不用于 L5。原始日志、逐步张量、checkpoint、环境和采集器均未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-swiftlora-sft-cli-d48-r2/`。首次 r1 尝试 13375 在模型计算前因过长 `TMPDIR` 触发 `OSError: AF_UNIX path too long`；保留在 `...-cli-d48-r1/`。r2 使用 Slurm worker 的短临时目录后完成。

仓库门禁：Slurm 13399 上 `bash tools/check_repo_layout.sh` 与 `tools/build/generate_manifest.py --check` 通过；完整 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 1383 passed、8 skipped、1019 subtests passed、1 failed。唯一失败用例尝试构建 oneDNN 时因 worker 缺少 CMake 抛错；保持同一隔离 JIT 缓存并设置 `use_mkl=0` 后，Slurm 13422 对该用例定向复验 1 passed。该复验说明失败由测试环境启用未安装的可选 oneDNN 引起；完整结构套件未在新变量下重跑。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | partial | 原生与 shim 均构造公开 SwiftLoRA `Seq2SeqTrainer`、Qwen template/dataset/AdamW；386 个模型参数的名称、shape、dtype、requires-grad 一致且均在 CUDA，96 个 adapter 初态逐键一致。未审计 buffer 与完整 `state_dict` 的所有键。 |
| L1 | PASS | 首批输入逐值一致；CUDA logits 和 loss 对齐，误差见上文；候选零 fallback。 |
| L2 | PASS | 三步 96/96 参数梯度、AdamW 两状态与每步更新逐项比较，均在 CUDA、finite，聚合相对 L2 最大 `1.65e-5`。 |
| L3 | blocked | 新进程公开 CLI 恢复在首步前失败：Transformers 将私有 SwiftLoRA 模型送入普通分片加载分支，找不到 `model.safetensors.index.json`；shim、同进程恢复与后续 RNG/dataloader/optimizer/scheduler 轨迹未运行。 |
| L4 | partial | 原生和严格 shim 均真实运行公开 `swift sft`，完成三步并保存 adapter checkpoint；因 L0 仍 partial、L3 未运行，整层验收未通过。 |
| L5 | blocked | L0/L3 未完整通过，且没有预热后十次稳态性能数据。 |
