# Qwen2 序列分类公开 SFT 双卡 NCCL CUDA 对拍

- 状态：本报告覆盖的固定 Qwen2-0.5B 序列分类公开 SFT 双卡配置，L0、L1、L2、L4 为 partial，L3、L5 未通过前置层级或尚未验证。双 rank NCCL、输入、梯度和三步权重对拍完成；不代表所有分类/embedding/reranker/reward 模型或整个 ms-swift。
- 日期：2026-10-08。
- 基线：Jittor `d2ea2dde9635ff7a5f4e9a5b7ecde86667a51546`；最近缓存的 `origin/2.0-refactor` SHA 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。本轮 live fetch 遇 OpenSSL `SSL_ERROR_ZERO_RETURN`，SSH `ls-remote` 遇 `Permission denied (publickey)`，故未能重新核验上游实时 HEAD。
- 范围：公开 `python -m swift.cli.main sft --task_type seq_cls`，Qwen2-0.5B `Qwen2ForSequenceClassification`，双 RTX 4090、FP32/eager、全参数 SGD、固定 8 行数据、每卡 batch 2、三步，NCCL 2.21.5。
- 维护者：ms-swift CUDA 适配。
- 复查条件：修复 scheduler checkpoint 状态差异后补新进程恢复轨迹；补 tokenizer/template、优化器参数名映射与完整模型输出状态；完成稳态双卡性能协议后再评 L5。

Slurm 13122 先运行原生 PyTorch oracle；三步公开训练、checkpoint、rank 0/1 每步输入/输出及 291 个全参数梯度均已保留。随后 Slurm 13123 运行 strict shim，先串行预热 Jittor/NCCL，再以公开 CLI 完成三步训练。两作业均在 `cscg-qh04`，GPU UUID 为 `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea` 与 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。双 rank 的 `RANK/WORLD_SIZE/LOCAL_RANK` 与 Jittor rank 变量一致，NCCL world size 为 2；候选每个 rank 导入/结束均 `use_cuda=1`、fallback `0`。初始 291 个参数名称、shape、dtype、值哈希相同，全部 trainable 且在 CUDA。基础 checkpoint SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`，数据 SHA256 为 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`。

比较器逐 rank、逐步验证全部捕获输入张量逐元素一致，并比较 CUDA logits/loss。logits 最大绝对差为 `3.10e-6` 至 `1.39e-5`；loss 最大绝对差不超过 `7.63e-6`。rank 1 的 loss 绝对值接近零，因此其相对 L2 不作为误差判断依据。每个 rank 的每一步都比较 291 个参数梯度，最大绝对差依次为 `6.55e-7`、`1.32e-7`、`3.45e-7`，相对 L2 为 `3.64e-6`、`2.21e-6`、`4.92e-6`。最终 checkpoint 的 291 个键/shape/dtype 一致，最大权重绝对差 `7.45e-9`、相对 L2 `5.91e-11`，168 个张量有更新。训练日志两侧 loss 分别为 native `4.12568808 / 3.76338148 / 4.11957359`、shim `4.12568474 / 3.76338148 / 4.11957836`。

GPU worker 对 `optimizer.pt` 做 zip 或 pickle 格式识别后逐字段/张量比较，两边相同；SGD 的 state 为空，参数组相同。`scheduler.pt` 不同：native 包含 `verbose=False`，shim 含 `_is_initial=False`。因此未尝试宣称恢复轨迹等价。原始命令、日志、rank JSON、梯度、NPZ、checkpoint、逐字段状态审计和比较结果未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-seqcls-ddp-nccl-v1/`。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 CLI 构造模型、tokenizer、dataset、tuner、optimizer、trainer；291 参数名称、元数据、值哈希和 CUDA 设备一致；尚缺 tokenizer/template 与完整参数到 optimizer 状态键映射。 |
| L1 | partial | 两 rank 三步输入逐值一致，logits/loss 有限且满足 CUDA 前向容差；没有保存完整模型输出结构与 hidden-state 清单。 |
| L2 | partial | 三步均比较全部 291 个 CUDA 梯度、更新后的全量权重和 optimizer 状态；但前置 L0/L1 仍 partial。 |
| L3 | blocked | 未进行同进程或新进程恢复；scheduler 持久化字段不同，未验证 RNG、数据游标、后续输入与续训轨迹。 |
| L4 | partial | native 与 strict shim 均经公开 `swift sft` CLI 端到端完成三步并产出 checkpoint；由于前置层未完整通过，不升级为 PASS。 |
| L5 | blocked | 前置层未通过，且没有预热后至少 10 次同步稳态双卡性能协议。 |

Slurm 13121 是启动脚本指向缺失 NCCL 目录的 harness 错误，模型未运行。13122 在原生阶段成功后，首次 shim 预热触发 Jittor 一次性 reload guard，故保留并复用原生结果；13123 修正预热顺序后完成候选。首步 JIT 期间共享缓存发生锁等待，rank 逐个编译不同 CUDA 算子，产物持续推进，最终训练正常完成；不把该编译等待计作训练失败。Slurm 13130 的状态审计错误地用 `torch.load` 读取 shim 原始 pickle，修正 zip/pickle 识别后，13131 在 worker 上完成只读状态比较。上述原始记录均保留。

文档收尾：Slurm 13133 完整结构测试暴露环境缺少 cmake、oneDNN 默认启用而导致 CPU matmul 用例失败；Slurm 13135 在 `use_mkl=0` 下对此用例定向复验通过。Slurm 13137 按仓库 Torch 模式命令并设 `use_mkl=0` 通过清单检查、布局检查及 `tests/structure`（`1384 passed, 8 skipped, 1019 subtests passed`）。8 个 skip 由 accelerator/声明依赖及未安装的 pytest-xdist 等检查依赖造成；不额外设置 `JITTOR_TEST_REQUIRE_EXECUTION`。Slurm 13136 使用该额外严格变量时虽所有断言通过，但因报告上述环境 skip 而退出 1，不作为最终门禁结果。`git diff --check` 通过。
