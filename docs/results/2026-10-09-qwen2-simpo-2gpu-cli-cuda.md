# Qwen2-0.5B stock `swift rlhf --rlhf_type simpo` 双卡 CUDA 对拍

- 状态：在六条完全相同的 preference 行、每 rank batch 1、三步训练及 step-3→4 恢复的限定配置下，原生 PyTorch 与 strict shim 的训练输入、首步 chosen/rejected logits、三步全参数梯度、rank 间梯度同步和恢复 step-4 前向/末态权重达到数值门槛；strict shim 两个 rank 真实 CUDA/NCCL 运行且 fallback=0。恢复后的 optimizer/scheduler/RNG 在 strict shim 连续与新进程轨迹间字节一致，但 native 与 shim 的状态文件哈希不同。按本 Skill 的 L0–L5 验收，L0/L1/L2/L3/L4 为 partial，L5 blocked；不代表一般多样本 sampler、其他偏好算法或 ms-swift 整体兼容。
- 日期：2026-10-10（恢复补验）。
- 基线：Jittor `211661b69e00a247e91c2852f74e6469a41f87bc`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，stock 双卡 `swift rlhf --rlhf_type simpo`，beta 2.0、simpo_gamma 1.0、cpo_alpha 0、SGD、固定六行 preference 数据、每 rank batch 1、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补完整 tokenizer/template/buffer 清单、同进程恢复，以及使用可区分样本验证 dataloader/sampler 游标；Torch shim、Transformers、TRL 或 ms-swift SimPO/CPO trainer 变化时复查。

## 运行与数值证据

模型权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。训练 JSONL 包含六条完全相同的合法 preference 行，SHA256 为 `592190769bde3c4cfd4db0c02a22b08a031c354447bf9b9608bbbc3e4cc58137`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0、NCCL 2.21.5。Slurm 14057 预检、14059 训练与后处理作业 14078 均在 `cscg-qh04` 的 RTX 4090 worker 上运行。训练分配的 GPU UUID 为 `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44` 和 `GPU-d813000b-1bfa-885e-cd61-0de97b365732`。

运行键 `20261009-qwen2-simpo-2gpu-cli-v13` 先执行独立原生 CLI，再执行 strict shim CLI；两侧均通过公开入口完成三步并保存 `checkpoint-3`。审计插件只包装 CPOTrainer forward/loss 与 SGD step 来记录数据，并调用原函数继续执行。两 rank 每一步 `input_ids`、`attention_mask`、`labels` 与各自原生记录完全一致。shim rank 日志记录 `world_size=2`、CUDA 已启用、私有 FileStore rendezvous、`MASTER_PORT=None`、shim 开启且 `fallback=0`。原生每 rank 三步耗时约 21 秒；shim 约 610 秒且含首次 JIT 编译，因此这些时间不构成 L5 比较。精确 CLI 参数、环境和采集插件保存在未版本化的运行目录 `train.sbatch`、`manifest.json` 与 `audit_plugin.py`。

首步 chosen/rejected logits 的 shape 均为 `[1,29,151936]`，最大绝对误差 `8.16137e-5`，relative L2 分别为 `2.57146e-6` 和 `2.53968e-6`。记录的其他聚合前向张量最大绝对误差分别为 `6.19888e-6` 与 `2.86102e-6`；loss 标量首步完全一致。三步 native loss 为 `1.843277812, 1.837927699, 1.832556486`，shim loss 为 `1.843283415, 1.837923646, 1.832565784`，最大绝对差 `9.29832e-6`。

每 rank、每一步均比较 290 个 trainable 参数的全部梯度，共 494,032,768 个 FP32 元素。三步最大绝对差依次为 `3.05474e-7`、`5.90459e-7`、`6.85453e-7`；rank 0 的 aggregate relative L2 最大 `9.78833e-6`，逐参数最大 relative L2 小于 `8.77e-5`。两个 runtime 在每 rank 每步的 290 项梯度均逐元素跨 rank 完全同步。checkpoint 的 290 个权重键一致，其中 170 个张量发生更新；末态权重最大绝对差 `7.45058e-9`，relative L2 `6.90933e-11`。

状态诊断运行键为 `20261009-qwen2-simpo-2gpu-cli-v16-optimizer-state-semantic`。原生与 shim 的 SGD optimizer 状态树完全相同（该 optimizer 的 per-parameter state 为空），trainer `global_step=3`、`max_steps=3`、`epoch=1.0` 相同。scheduler 状态的公共键与值一致，但 native 保存 `verbose=False`，shim 保存 `_is_initial=False`，形成两个键名差异；该状态字典不能记为完全相同，也未验证加载后的后续调度轨迹。状态读取和比较在严格 CUDA scope 下完成，fallback 增量为 0。

### step-3→4 双卡恢复补验

恢复补验沿用相同模型、六行数据、双 RTX 4090、FP32/eager/SGD 与公开 CLI。作业 15292 的原生连续、原生恢复和 strict shim 连续轨迹已完成；shim 恢复首次因 FileStore rendezvous 竞态在模型构造前超时。两卡 strict-shim TCPStore 的独立 `init_process_group + all_reduce` 探针 15294 三轮均通过，rank 设备与 all-reduce 值正确、fallback=0。随后 job 15295 仅运行 strict shim 从原 shim `checkpoint-3` 的新进程恢复到 step 4，使用每次独立 TCPStore 端口；两 rank CUDA=1、shim 标记正确、fallback=0，公开 CLI 保存 `checkpoint-4`。

GPU-worker artifact audit job 15296 将 native job 15292 的恢复结果与 shim job 15295 对拍：每 rank step-4 输入逐 tensor 完全一致；CPO 前向张量最大 relative L2 `2.584e-6`，loss 差在 `5e-3` 容差内。290 个末态权重键和 global_step=4 一致，最大权重绝对差 `7.45058e-9`、relative L2 `7.91617e-11`。两个 shim rank 从启动到退出均记录 CUDA、strict shim 与 fallback=0。shim 自身恢复与连续轨迹比较中，step-4 输入一致、前向/损失在容差内，290 个权重最大绝对差 `7.45058e-9`；optimizer、scheduler、rank0/1 RNG checkpoint 文件逐字节一致。native 与 shim 的 optimizer、scheduler、RNG 文件哈希不相同，因此没有把跨 runtime checkpoint 文件相等当作通过，也没有证据证明 native/shim RNG 内部状态等价。

该补验只覆盖独立新进程恢复。相同数据行重复六次，step-4 输入身份一致不能证明可区分样本下的 dataloader 游标位置；同进程保存并恢复也未执行。因此 L3 从 `not-run` 更新为 `partial`，L0/L1/L2/L4 保持原有 partial，L5 仍 blocked。第一次 FileStore 失败和 TCPStore 诊断是恢复机制的 harness/core 分流证据，不计为兼容 PASS。恢复运行键与日志、比较器和逐 rank 原始张量保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261010-qwen2-simpo-2gpu-resume-v2/`、`20261010-qwen2-simpo-2gpu-resume-v3/`、`20261010-qwen2-simpo-resume-cross-runtime-audit-v1/`。

L0–L5 状态如下：

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 两 rank 经公开 CLI 构造真实模型、tokenizer/template、trainer、SGD optimizer 与 scheduler；290 项参数初态、逐步输入身份一致。没有完整枚举模型 buffer、tokenizer/template 状态映射；scheduler 状态还存在字段差异。 |
| L1 | partial | 固定同权重、同输入首步 chosen/rejected 完整 logits 与聚合输出已对拍；没有保存 hidden state 或后续训练步逐层前向。 |
| L2 | partial | 三步全部 290 项梯度及末态模型权重达容差，NCCL rank 同步吻合；optimizer 状态树完全相同。scheduler 字段名不同，且未采集每步直接更新量或恢复后的下一步更新。 |
| L3 | partial | 原生和 strict shim 均由独立新进程从 step-3 checkpoint 恢复至 step 4；两 rank 输入相同、前向/loss 与末态权重达容差。strict shim 连续与恢复轨迹的 optimizer/scheduler/RNG 文件字节一致。跨 runtime 状态文件哈希不同；同进程恢复未运行，重复数据无法证明 dataloader/sampler 游标，故不满足完整 L3。 |
| L4 | 入口已运行，功能未通过 | native 与 strict shim 均通过公开双卡 `swift rlhf --rlhf_type simpo` 完成三步并保存 checkpoint；当前只覆盖六条重复样本，状态合同存在 scheduler 键差异，不能据此标为完整 L4 通过。 |
| L5 | blocked | 前置层尚未完整通过；没有执行预热后至少 10 次同步稳态计时及统一显存口径。 |

原始输入、逐 rank/逐步梯度、checkpoint、状态比较器、环境记录和日志均未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-simpo-2gpu-cli-v13/` 与 `20261009-qwen2-simpo-2gpu-cli-v16-optimizer-state-semantic/`。该结果仅支持上述六条相同数据行、Qwen2-0.5B、FP32/eager/SGD 的双卡配置。

文档门禁 Slurm 14083 在 RTX 4090 worker 上通过：`bash tools/check_repo_layout.sh` 成功，`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 1386 passed、6 skipped、1027 subtests passed。恢复补验文档的门禁 Slurm 15297 也通过：布局检查成功；结构测试 1384 passed、8 skipped、1019 subtests passed。该 gate worker 未配置 nvcc，结构测试中的 Jittor 部分按门禁环境运行于 CPU；此结果只验证文档和结构门禁，不构成 CUDA 兼容证据。8 个 skip 中 2 个因该解释器未安装 Jittor、2 个因未安装 pytest-xdist，其余由门禁分类记录。
