# Qwen2-0.5B 公开 CPO LoRA `swift rlhf` CUDA 对拍

- 状态：公开 CPO LoRA 入口在两侧均可运行；原三步轨迹没有同输入条件。新的一步控制诊断确认初始 adapter 相同、实际 batch 不同，且 `train_dataloader_shuffle=false` 未被当前 CPO dataloader 路径采用。该批次顺序缺陷已定位；同输入 CPO 数值兼容仍未验收，不把原三步差异判作算子数值错误。
- 仓库：Jittor `7751f74e8405adda9cd30c950bc037dda19bd5d4`（2.0.0）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15，PyTorch 2.6.0+cu124 / TRL 0.24.0；RTX 4090，CUDA 12.2，驱动 580.178.04。
- 维护者：ms-swift CUDA 适配。
- 复查条件：固定并核实初始 LoRA adapter 和每步样本 ID 后，补同批 logits/loss、逐参数梯度和 optimizer 状态对拍，再隔离本次轨迹与 adapter 分歧。

Slurm 13530 在 `cscg-qh17`（GPU UUID `GPU-d892cd13-505f-f0a8-be2f-ee118c4155b8`）依次运行原生 PyTorch oracle 和 strict shim。模型 Qwen2-0.5B 权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；固定四条偏好数据 SHA256 为 `57b44eaa07bdf2f2a82d5f407df5b04e7ae81ac4f3be04f65ef3ed07322cde28`。配置为公开 `python -m swift.cli.main rlhf --rlhf_type cpo`、LoRA rank 4 / alpha 8 / `all-linear`、FP32、sigmoid CPO、beta 0.1、cpo_alpha 1、SGD `1e-5`、batch 2、固定顺序三步。两侧均保存 checkpoint-3；模型共约 496.23M 参数，2.20M 可训练 adapter 参数。

strict shim bootstrap 父、子进程开始和退出均记录 `use_cuda=1`、shim 标记真、`fallback=0`，Jittor 日志确认 CUDA enabled。候选完成 CPO backward 和三步更新，没有出现此前 DPO 场景的 `NanoVector` 断言。

| step | 原生 loss | shim loss | 原生 grad norm | shim grad norm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.99020 | 7.24632 | 46.93347 | 39.78808 |
| 2 | 5.51598 | 6.13464 | 40.18628 | 40.01855 |
| 3 | 6.04515 | 7.98955 | 40.08770 | 43.27671 |

Slurm 13539 在 worker 上比较 safetensors adapter：336 个键、2,199,552 个元素一致，末态最大绝对差 `6.67664e-2`，全 adapter 相对 L2 `1.41364`。step loss 最大差 `1.94440`；chosen/rejected log-prob、reward margin 和 accuracy 也存在明显差异。该对照没有记录初始 adapter 张量或每步 batch ID，所以不能判断差异究竟来自初始化、样本顺序还是 CPO 数值路径；三步末态不构成梯度或 optimizer 状态等价证据。

首次 shim JIT 混入总耗时：候选训练约 295 秒，首步约 240 秒；原生训练不到 1 秒。此数据不用于性能比值或 L5。预热复用了先前运行目录的 Jittor CUDA cache（Jittor 源码未变，前次缓存包含 `cuda_archs=89`）；缓存与原始日志留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-cpo-peftlora-cli-v5/`。早期 run `...cli-v1` 到 `...cli-v4` 的预检/CLI 参数/空缓存启动配置错误现场均保留，未计为模型兼容结果。

### 根因复核与固定初态诊断（2026-10-09）

更正本报告上一版的静态判断：`get_train_dataloader` 中传递 `train_dataloader_shuffle` 的实现实际属于独立的 `DataLoaderMixin`（`swift/trainers/mixin.py` 第 1345 行起），不是 `SwiftMixin`。ms-swift `88d727951203256baa564c643c651b6f8d90fd7e` 的 `CPOTrainer` 继承 `RLHFTrainerMixin, SwiftMixin, HFCPOTrainer`，没有 `DataLoaderMixin`。`RLHFTrainerMixin.get_train_dataloader` 只调用 `super()`；该 MRO 随后落入 Transformers `Trainer.get_train_dataloader`，其 `_get_train_sampler` 对 map-style dataset 返回 `RandomSampler`，不读取 `train_dataloader_shuffle`。因此本次配置中的 `--train_dataloader_shuffle false` 在该 CPO 路径没有关闭随机采样。此前“Swift CPO dataloader 传递了 false”的结论错误，原因是把两个相邻 class 的方法归属混淆。

运行键 `20261009-qwen2-cpo-fixedinit-audit-v1`：Jittor `5c51281947db7565a6e382d6f26c951719cd4d60`，ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。Slurm 13639 在 worker 完成 CLI/hook 预检；Slurm 13640 在 `cscg-qh17` RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）完成原生与 strict shim 一步公开 CLI 并保存 checkpoint，比较器因错误要求动态 padding 后的输入 shape 完全相同而退出；Slurm 13642 只对已保存数据做 worker 端比较，没有重跑模型。Qwen2 checkpoint SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；初始 adapter SHA256 为 `4033d97f2838b3bdebfb8fbaea2b4d582fa24d12472eae460545afe97172fc4e`；固定偏好数据 SHA256 为 `57b44eaa07bdf2f2a82d5f407df5b04e7ae81ac4f3be04f65ef3ed07322cde28`。原生和 strict shim 从同一 Qwen2 checkpoint-3 adapter 开始；336 个可训练 adapter 张量、2,199,552 个元素逐项完全一致，参数名映射相同且都在 CUDA。首批输入实际不同：原生 `input_ids/labels/mask` shape `[4,30]`，候选 `[4,33]`，有效 label token 也逐行不一致。两侧 seed 与两个 shuffle 参数均相同并记录为 1234/false。候选父子进程 strict CUDA `fallback=0`。旧比较器的 shape 假设错误，原始文件与修正诊断输出保留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-cpo-fixedinit-audit-v1/`。

首步 chosen/rejected forward 标量和 336 个梯度确实不同（最大绝对差分别至多 `5.31135` 和 `3.35117`；梯度最坏单参数相对 L2 `1.53059`），但这些只是在不同样本与不同序列 shape 上得到的诊断值，不构成同输入数值验收。该观察与 Slurm 13189 的 Torch `RandomSampler` 跨 runtime 顺序差异相符；这一 CPO 运行未捕获 sampler index 本身，故不把 sampler RNG 机制进一步外推为所有数据路径的结论。候选父子进程仍记录 strict CUDA、`fallback=0`。

| 层 | 状态 | 证据与缺口 |
| --- | --- | --- |
| L0 | partial | 新诊断确认 336 个 trainable adapter 参数初态、名称和 CUDA 元数据一致；不同首批输入与完整非 trainable 状态清单仍未满足构造合同。 |
| L1 | not-run（诊断入口已运行） | 首批输入内容和 shape 不同；forward 张量已捕获但不能用于同输入数值验收。 |
| L2 | partial（诊断梯度不可作对拍） | 原三步公开 CLI 与新一步 CLI 都完成；两次均缺相同 batch 条件。新一步采集 336 个梯度，但输入不同，不能证明梯度/更新语义差异。 |
| L3 | not-run | 未验证新进程恢复与后续轨迹。 |
| L4 | blocked（入口已运行） | 原三步与固定 adapter 一步的公开 CLI 两侧均完成并保存 checkpoint；前层输入身份合同未通过，按层级规则不升级。 |
| L5 | blocked | 前层未通过，且首次 JIT 混入候选时长；未做预热后至少 10 次稳态测量。 |

这是 Qwen2-0.5B、固定小数据、LoRA CPO 的单配置结果，不代表 CPO 所有实现、其他偏好算法或整个 ms-swift。

### 固定初态收敛补验（2026-10-10）

本节按当前 Skill 的 L0–L5 口径给出一个更窄的 CPO profile 结果；不改写上文旧运行键按旧输入合同得出的历史结论，也不外推到其他偏好数据、优化器或 ms-swift 整体。

原生 oracle 为 Slurm 15352 的 run key `20261010-qwen2-cpo-fixedinit-convergence-v3`，strict shim 轨迹由新 run key `20261010-qwen2-cpo-fixedinit-convergence-v4` / Slurm 15357 完成。两侧依次在 `cscg-qh04` 同一 RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`）运行公开 `swift rlhf --rlhf_type cpo`。Jittor HEAD `555f5b8bfcdc3b809ce104b62014fb296e48e5d6`，ms-swift HEAD `88d727951203256baa564c643c651b6f8d90fd7e`；Qwen2-0.5B 权重 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，初始 adapter SHA256 `4033d97f2838b3bdebfb8fbaea2b4d582fa24d12472eae460545afe97172fc4e`，单条 preference JSONL SHA256 `5fd06bff17bc409ce5d69325b68abcef5bdd4d9549ba317746c1d0be051808e4`。配置为 FP32/eager、LoRA rank 4 / alpha 8 / all-linear、SGD `lr=2e-4`、`weight_decay=0`、batch 1、固定同一偏好对、12 步；损失窗口预先固定为 step 1–3 与 10–12。

原生 12 步和 strict shim 12 步分别保存 checkpoint-12。336 个 trainable adapter 初态逐张量一致；所有 12 步输入一致。首个 CPO forward 张量 shape、dtype 相同且有限，最大绝对差为 `1.383e-4`，相对 L2 最大 `3.134e-6`（误差仅作为诊断）。两侧每步 336 个 trainable 参数梯度都存在、在 CUDA、有限且非零；每步更新 300 个（step 1）或 336 个（step 2–12）参数，更新有限且非空。SGD 有 2 个参数组（336 与 0 个参数），学习率、动量和 weight decay 配置有效；momentum 为 0，optimizer state entries 为 0，符合该 SGD 配置。保存的 adapter checkpoint 各有 336 个有限张量。

| runtime | step 1–3 平均 loss | step 10–12 平均 loss |
| --- | ---: | ---: |
| PyTorch CUDA | 1.51681169 | 1.50025185 |
| strict Jittor torch shim CUDA | 1.51681161 | 1.50025292 |

候选父子进程均记录 CUDA=1、shim 标记真、fallback=0。v4 插件每个逻辑 step 写出两条逐字段完全相同的梯度/更新记录；Slurm 15396 在 GPU worker 上只折叠完全相等的重复 JSON 对象并完成其余对拍，没有合并或忽略任何字段。重复记录由哪个 shim/CLI 调用层触发尚未定位，因此保留该 harness 观察为限制，不把它解释成模型层缺陷。

| 层 | 固定 profile 状态 | 证据边界 |
| --- | --- | --- |
| L0 | pass | 同一真实模型、tokenizer/template、单条数据、LoRA adapter、trainer 与 SGD 在 CUDA 构造；336 个 trainable 初态、dtype/device 和 optimizer 参数组核实。 |
| L1 | pass | 初态和 12 步输入一致；首批 forward 输出结构、shape、dtype、有限性通过，误差只作诊断。 |
| L2 | pass | 两侧 12 步各自 loss 窗口下降；每步 336 个梯度完整、有限、非零，参数更新非空且 optimizer state 与配置匹配。 |
| L3 | not-run | 未测试同进程/新进程恢复、scheduler/RNG 和 dataloader cursor。 |
| L4 | pass | 两侧均从公开 `swift rlhf --rlhf_type cpo` CLI 端到端训练并保存 checkpoint-12。 |
| L5 | blocked | L3 未运行；没有至少 10 次稳态性能测量。 |

审计产物位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261010-qwen2-cpo-fixedinit-artifact-audit-v8/`；原始模型运行产物位于对应 v3/v4 state 目录。v1/v2 仅为 CLI/harness 前置失败；v3 shim 的首步在 harness 序列化内部 Var 时退出；v4 训练成功但首个离线比较器未接受重复记录。最后一次 provenance probe v5 因 `sitecustomize.py` f-string 语法错误未启用 strict shim，明确排除为兼容证据。CPO 本轮问题达到五轮上限，不再重复训练。
