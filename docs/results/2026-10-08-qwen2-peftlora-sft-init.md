# Qwen2-0.5B PEFT LoRA 公开 SFT：同状态三步对拍

- 状态：历史严格逐值协议下固定配置 L0-L2 与单卡公开 `swift sft` L4 通过；按新训练收敛协议复验时，L2 在 strict shim 第 5 步被 `nano_vector.h:41` slice overflow 阻断，未通过收敛门槛。L3 新进程续训已部分验证；同进程恢复、buffer 和有持久逐参数状态的 optimizer 配置仍未验；L5 blocked。LoRA 随机初始化本身仍不对齐，需用显式转入同一 adapter 状态建立 oracle；不代表 PEFT LoRA 或 ms-swift 整体兼容。
- 日期：2026-10-08。
- Jittor 基线：`5def3f89d8de1ddc8d95d18778b10d08c72dbedc`（包含上游 `origin/2.0-refactor` 的 `26bf23f9a0c0e3e12838089ec67a9fb380cba9c2`）。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch CUDA RNG/初始化语义修正，或构造相同初始 adapter 的确定性对拍方案可用时；之后重做同权重前向与三步轨迹。

运行键 `20261008-peft-lora-l2-inputcap-try4-d51b` 在 Slurm 12738 的 cscg-qh17 RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）经公开 `swift sft` CLI 先原生后严格 shim 完成三步；model/base checkpoint、PEFT adapter safetensors、Qwen2-0.5B FP32/eager、四条固定样本、batch 4、max length 64、SGD lr `1e-5`、seed 1234 与上一段配置相同。两侧 adapter 文件 SHA256 均为 `e34a28cdc1d227907e13b96f061da71053f13fe0e68d678d274fdc69816e7c56`，数据 SHA256 为 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。完整 386 项参数的名称、shape、dtype、requires_grad 与 CUDA device 元数据一致，初始张量 SHA256 全部一致；其中 96 项可训练。公开 Trainer 首次 forward hook 保存的 `input_ids`、`labels`、`attention_mask` 三者逐项相同，logits shape `[4,8,151936]`，最大绝对差 `1.32173e-4`、相对 L2 `2.65440e-6`；loss 最大绝对差 `5.24521e-6`、相对 L2 `1.37045e-6`，均有限。三步每步 96 项梯度均在 CUDA：最大梯度绝对差依次为 `8.49366e-7`、`6.56582e-7`、`1.02818e-6`；最坏单参数相对 L2 依次 `9.10550e-6`、`9.06069e-6`、`1.47676e-5`。更新权重最大绝对差依次 `8.52651e-12`、`1.28466e-11`、`1.86265e-9`；最坏相对 L2 `9.10680e-6`、`7.12428e-6`、`7.07144e-6`。固定数据集仅四条且无 shuffle，CLI 记录了三轮各一步；SGD momentum 为默认零，不持有逐参数历史状态。shim 两个进程起止、模型构造、前向与三次 optimizer step 的 `use_cuda=1` 且 fallback 全程 0。比较由 GPU worker Slurm 12745 对保存的原始产物完成；12744 是一次因 `cuda:0` 元数据断言过严失败的比较器运行，未重跑模型。

运行键为 `20261008-qwen2-peftlora-sft-l0l2-8ac1`。Slurm 12634 在 cscg-qh17 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`）上先原生后严格 shim，经当前 ms-swift 公开 `swift sft` CLI，对 Qwen2-0.5B FP32/eager 使用 PEFT LoRA（q_proj/v_proj、rank 8、alpha 32、dropout 0）、SGD、四条固定样本、batch 4、长度 64 训练三步。两侧均构造 PEFT `PeftModelForCausalLM`，494,573,440 个总参数、540,672 个可训练参数（96 个张量），并产生 adapter checkpoint。Shim 参数与每步梯度记录均在 CUDA；每步 96 个梯度非空，候选运行开始/结束及各步 fallback 均为 0。原生 loss 为 `[3.8273680210, 3.8268311024, 3.8262913227]`，shim 为 `[3.8273668289, 3.8268554211, 3.8263378143]`。

这些数值不能作为同起点训练对拍：公开 Trainer 捕获钩子未命中 Swift 覆盖的 `compute_loss`，因此公开 SFT 运行没有保存 L1 logits/输入；参数初始值审计和后续窄探针发现 LoRA A 初值不同。Slurm 12645 对三步 96 项梯度及最终 adapter checkpoint 的诊断比较分别得到最大绝对差约 `0.1036`，梯度相对 L2 最大约 `3.63`，最终 adapter 权重最大绝对差 `0.06679`、相对 L2 `3.03`。因初始 adapter 状态不同，这些差值不构成梯度/更新语义回归的结论。

为区分 base model 随机消耗的影响，运行键 `20261008-qwen2-peftlora-initprobe-5e71` 与 `20261008-qwen2-peftlora-resetseed-39bd` 分别在 Slurm 12652/12655 和 12656/12661 对相同 Qwen2 base 在原生与 shim 进程独立构造 PEFT LoRA；第二组又在 `get_peft_model` 前重置 seed。两组相同的 96 个键中，LoRA A 初值仍不相同，最大绝对差 `0.066790726`、相对 L2 `1.43842`。这表明仅重置全局 seed 不足以建立相同 adapter 初态。

运行键 `20261008-peft-kaiming-init-0d4f` 的直接 CUDA primitive 探针在 Slurm 12666 使用相同 seed、同一 cscg-qh17 RTX 4090，对 `(8,896)` FP32 CUDA 张量分别调用原生与严格 shim `torch.nn.init.kaiming_uniform_(a=sqrt(5))`。两侧均有限且位于 `cuda:0`；shim `use_cuda=1`、`fallback_count=0`。Slurm 12668 在 worker 上比较，得到最大绝对差 `0.06606726`、相对 L2 `1.41119`，输出不相等。至此已把 PEFT 初始值差异缩小到匹配 seed 的 CUDA `kaiming_uniform_` primitive 行为；该探针尚未把差异进一步归因为 RNG 生成还是初始化映射，故不声称已定位到具体内部实现。

为建立有效的 PEFT 前向 oracle，运行键 `20261008-peft-lora-l1-transplant-try2-9c1e` 在 Slurm 12690 的 cscg-qh04 RTX 4090 上，以相同 Qwen2-0.5B checkpoint、FP32/eager、PEFT LoRA q/v 配置构造两侧模型；原生生成 96 个 adapter 参数后，将其逐项转入 shim 模型并断言每个参数完全相等。输入 IDs/mask 由原生 tokenizer 生成并保存，两侧都在 CUDA 上执行；全部 386 个模型参数、输入、logits 和 25 组 hidden states 均为 CUDA 且有限。Slurm 12692 在 worker 上比较：logits shape `[2,7,151936]`，最大绝对差 `6.7234e-5`、相对 L2 `2.0526e-6`；hidden shape 均为 `[2,7,896]`，最大绝对差 `6.2561e-4`（h24），最大相对 L2 `2.7906e-6`（h23）。Shim `use_cuda=1`、`fallback_count=0`。该直接 Transformers/PEFT 固定输入前向满足数值门槛，但它没有通过 Swift Trainer 或公开 CLI，且未记录完整 386 项参数状态清单，故不能升级此面的层级 L0/L1 总状态。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | PASS（固定配置） | Slurm 12738 公开 CLI 构造完整模型、PEFT、tokenizer/data/trainer；386 项状态元数据相同且初始 SHA256 全等，全部 CUDA。随机初始化差异由显式相同 adapter checkpoint 消除。 |
| L1 | PASS（固定配置） | Slurm 12738 原生与 shim 公开 CLI 首批实际输入完全一致，logits/loss 结构、有限值和误差通过；Slurm 12690/12692 另对相同 adapter 的直接 PEFT 前向与 25 组 hidden 通过。 |
| L2 | 历史协议 PASS；新协议 blocked | Slurm 12738 的三步固定轨迹严格逐值结论保留。Slurm 15908 新协议要求 8 步并比较首/末各两步平均 loss；native 完成 8 步且 loss 下降，但 strict shim 第 5 步失败，未完成末段窗口，故新协议未通过。 |
| L3 | partial | Slurm 13455 新进程从 step-3 checkpoint 恢复后直接捕获首批 input IDs/labels/mask，三步 RNG 状态分别与各自 checkpoint 完全匹配，scheduler 核心状态、step-4 梯度及 adapter 对齐。配置使用 SGD(momentum=0)，逐参数 optimizer state 为空；shim scheduler 另含 `_is_initial` 字段。未测同进程恢复、buffer 或有持久 optimizer state 的配置。 |
| L4 | PASS（单卡 `swift sft`） | Slurm 12738 两侧公开 CLI 均完成三步并保存 checkpoint；仅代表该 PEFT LoRA 固定配置。 |
| L5 | blocked | L3 完整恢复合同仍不完整；本配置未运行性能协议。 |

### 新训练收敛协议复验（Slurm 15908）

用户确认更新训练验收协议后，用新运行键 `20261010-qwen2-peftlora-sft-convergence-v1` 按预先固定的窗口复验同一公开 `swift sft` PEFT LoRA 配置：Qwen2-0.5B FP32/eager、同一显式 adapter（SHA256 `e34a28cdc1d227907e13b96f061da71053f13fe0e68d678d274fdc69816e7c56`）、相同四行数据（SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`）、batch 4、SGD `1e-5`、constant scheduler、无 shuffle、8 步。验收窗口定为前两步平均 loss 与第 7–8 步平均 loss，末段须更低；不以跨运行时逐值误差判训练通过。

原生 CUDA 在 cscg-qh04 RTX 4090 完成 8 步，loss 从 `3.82736802` 降到 `3.82359934`；前两步均值 `3.82709956`，后两步均值 `3.82386685`，满足原生窗口趋势。strict shim 构造出相同配置并完成前 4 步，loss 为 `3.82736683, 3.82683325, 3.82628608, 3.82575369`；前两步均值 `3.82710004`，目前可见的第 3–4 步均值 `3.82601989`，但这不是合同要求的末段窗口。第 5 步反向后在 Swift `clip_grad_norm_` 调用 `grad_norm.isnan().item()` 时触发 `nano_vector.h:41: slice overflow: 94692996826779 0 1`；job15908 FAILED，shim 未达到第 7–8 步，故本次新协议 L2 未通过。shim 捕获的前 4 次 optimizer step 各有 96 项 CUDA 梯度和更新文件，进程起止 `use_cuda=1`、fallback 0；后续完整性/有限值与窗口审计因作业提前失败未执行。原生 8 步已完成，相关 checkpoint 与逐步文件留在未版本化 state。

本次没有复用旧运行键，也没有改写旧的三步严格逐值结论。该错误形态与既有 NanoVector slice overflow 诊断相同；历史根因追踪已达五轮上限，不在此复验中追加盲目追踪或绕开梯度范数检查。L0/L1/L4 保留此前该固定配置已通过的证据，L3 仍 partial，L5 仍 blocked；此次训练窗口未完整，因此不宣布新协议 L2 通过。

### 新进程恢复补充（Slurm 12799/12802）

运行键 `20261008-peft-lora-l3-restore-ckpt3-r2` 在 cscg-qh17 RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）先原生后严格 shim，通过公开 `swift sft --resume_from_checkpoint` 分别从既有三步 `checkpoint-3` 新进程恢复，目标 `max_steps=4`。两侧 checkpoint 均含 adapter、`trainer_state.json`、optimizer、scheduler 与 RNG 文件。原生与 shim 的 Trainer `global_step` 均从 3 续至 4、epoch 从 3 到 4；四条样本、关闭数据集与 dataloader shuffle，因此下一轮固定一个 batch。第 4 步 loss 分别为 `3.82575250` 与 `3.82575178`，96 个 trainable CUDA 梯度最大绝对差 `6.88713e-7`、最坏单参数相对 L2 `9.50411e-6`；第 4 步 adapter 跨运行最大绝对差 `1.86265e-9`、相对 L2 `6.01197e-6`。候选启动及 optimizer step 的 `use_cuda=1`、fallback 0。比较作业 12802 在同型号 GPU worker 完成。

恢复层仍只记 **partial**：本次捕获器没有命中 Swift 覆盖的 `compute_loss`，故没有保存恢复后的首批 IDs/labels 或恢复前参数快照；固定数据顺序和 global_step/epoch 只能间接支撑游标，不能代替直接输入与 RNG 状态核对。该记录证明这组无 dropout、无 momentum、constant scheduler 的 LoRA CLI 配置可从新进程续一步并保持数值接近，不证明任意 dataloader、optimizer 或 RNG 恢复。首次尝试 12795 因 TMPDIR 路径过长导致 `AF_UNIX path too long`，发生在模型训练前；日志保留。原始比较文件位于未版本化目录 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-peft-lora-l3-restore-ckpt3-r2/`，失败记录位于 `.../r1/`。

### 新进程恢复状态补充（Slurm 13455）

运行键 `20261008-peft-lora-l3-restore-input-rng-r2` 在当前 Jittor HEAD `7fef532497c8dc533148fafc0dbcfd4f38b57b04` 上，于 cscg-qh17 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`）先原生后 strict shim，通过同一个公开 `swift sft --resume_from_checkpoint` 配置分别从既有原生与 shim step-3 checkpoint 续到 step 4。恢复后实际送入模型的 `input_ids`、`labels`、`attention_mask` shape 分别为 `[4,31]`、`[4,8]`、`[4,31]`，跨 runtime 逐项完全一致。RNG 加载后 Python、NumPy、CPU Torch、CUDA Torch 状态分别与各自 checkpoint 保存值完全一致。Trainer 恢复的 scheduler `last_epoch=3`、`_step_count=4`、`base_lrs` 与 `_last_lr` 跨 runtime 一致；shim 多出 `_is_initial=false`。本配置为 SGD momentum 0，optimizer 没有逐参数持久状态，因此不能将此结果外推为 AdamW/momentum 状态恢复。两个 runtime 的 `global_step` 均从 3 运行到 4；step-4 loss 原生 `3.82575250`、shim `3.82575130`（差 `1.19209e-6`）。96/96 梯度有限且跨 runtime CUDA 对拍，最大绝对差 `6.14673e-7`、最坏单参数相对 L2 `9.41146e-6`；96 项最终 adapter 最大绝对差 `1.86265e-9`、相对 L2 `6.01110e-6`。strict shim 进程、恢复与 optimizer step 均记录 `use_cuda=1`、`fallback_count=0`。

首次采集尝试运行键 `20261008-peft-lora-l3-restore-input-rng-r1`（Slurm 13445）在原生训练开始前由采集器错误判定 CUDA RNG 不匹配：非分布式 checkpoint 将单卡 CUDA RNG 保存为 tensor，采集器误按 tensor 列表比较。该尝试没有执行训练，相关日志留在 `.../20261008-peft-lora-l3-restore-input-rng-r1/`。r2 修正状态格式处理后完成两侧恢复、step 4 与比较。原始事件、checkpoint、逐项状态和比较器保存在未版本化目录 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-peft-lora-l3-restore-input-rng-r2/`。

L3 仍为 **partial**：该补充通过新进程直接核对了续训输入和各自 RNG，但没有验证同进程恢复、模型 buffer，以及拥有逐参数状态的 optimizer 恢复。此前试验与本次运行均限于 Qwen2-0.5B、四条固定数据、无 shuffle、FP32、PEFT LoRA、SGD(momentum=0) 和 constant scheduler。

脚本、日志、梯度、checkpoint、元数据及比较文件保留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-peftlora-sft-l0l2-8ac1/`、`20261008-peftlora-initprobe-5e71/`、`20261008-peftlora-resetseed-39bd/`、`20261008-peft-kaiming-init-0d4f/` 与 `20261008-peft-lora-l2-inputcap-try4-d51b/`。该结果只限 Qwen2-0.5B 此 PEFT LoRA 配置，不代表 PEFT LoRA 或 ms-swift tuner 功能面整体兼容。

Slurm 12747 的文档门禁通过 `generate_manifest.py`、`--check` 与布局检查（202 active Markdown files）；Torch 模式结构测试结果为 `1382 passed, 8 skipped, 2 failed, 1019 subtests passed`。失败一项是 CPU matmul 测试需 oneDNN v3，但验收环境没有 `cmake`；另一项发现既有全参数 SFT 前向及三步梯度报告没有列入结果索引 toctree。已把这两篇报告补入 `docs/results/index.md`，Slurm 12782 定向复验可达性规则 `1 passed`，本次报告索引复验 Slurm 12809 也为 `1 passed`；同一作业的发布清单检查与布局检查通过（202 active Markdown files）。全量结构测试未重跑；oneDNN/cmake 环境缺口仍待有 cmake 的验收环境处理。
