# Qwen2-0.5B 公开 CPO LoRA `swift rlhf` CUDA 对拍

- 状态：固定单偏好对 CPO profile 的当前代码基线（Jittor `python/jittor`/`compat` tree 对应 `73232b9`）L0–L2 与 L4 已通过；L3 恢复仍明确绑定 `67228d557`，不自动转移到新提交；L5 未运行。完整模型 state/buffer 清单已补齐，唯一 buffer `rotary_emb.inv_freq` 的 `requires_grad` 标志差异单独记录。原三步异批次轨迹和其批次顺序根因保留为历史结论，不把跨实现数值差异外推为算子错误。
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

### 固定初态单样本 checkpoint-12 恢复补验（2026-10-12）

本节只补本固定单偏好对 profile 的 L3，并绑定 Jittor `67228d5578495ad35f35d157a2e2c61a7d46b5c7` 与 ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。旧章节的 L0–L2/L4 结果仍绑定各自记录的旧 SHA 和运行键；本次不把它们自动提升为新基线结论。运行键为 `20261012-qwen2-cpo-fixedinit-l3-restore-v2`，Slurm job `17940` 在 `cscg-qh04` 完成，终态 `COMPLETED (0:0, 00:03:58)`。GPU 为 RTX 4090，UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。启动前 job `17937` 的独立 worker 预检通过：当前用户 NFS export、64 KiB fsync 探针、容量/inode、同指纹 Jittor cache、no-build bootstrap 和 strict CUDA 小算子均通过。

固定配置仍为公开 `swift rlhf --rlhf_type cpo`、Qwen2-0.5B、单条 preference JSONL、PEFT LoRA `r=4/a=8/all-linear`、FP32/eager、SGD `lr=2e-4/weight_decay=0`、batch 1、seed/data_seed 1234、step 12 到 16。模型、初始 adapter、数据 SHA256 分别为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`、`4033d97f2838b3bdebfb8fbaea2b4d582fa24d12472eae460545afe97172fc4e`、`5fd06bff17bc409ce5d69325b68abcef5bdd4d9549ba317746c1d0be051808e4`。输入 native checkpoint-12 来自旧运行键 `20261010-qwen2-cpo-fixedinit-convergence-v3`，strict shim checkpoint-12 来自 `...convergence-v4`；两份输入 checkpoint 的文件 SHA 清单在 job 前后完全一致。

同进程路径由同一个公开 CLI Trainer 从 checkpoint-12 恢复至 step14，再在同一 Trainer 对象中加载 checkpoint-14 并继续至 step16。新进程路径由进程 A 从 checkpoint-12 到 step14，再由独立进程 B 从 checkpoint-14 到 step16。每个恢复点都逐项验证 336 个 CUDA adapter 张量精确恢复；optimizer 参数组与状态、scheduler 状态、Python/NumPy/Torch CPU/CUDA RNG 均与对应 checkpoint 精确匹配。每个逻辑 step 13–16 均记录相同输入身份 `1920e2fd87aac1da093272baaa95ad9dee542e47d8def60f7aecdd34dd88aa7d`、预期 epoch/global-step 关系、336 个有限非零 CUDA 梯度和 336 个实际更新参数。原生与 strict shim 的样本身份一致；strict shim 六个 CLI worker 进程均记录 CUDA=1、fallback=0。

| runtime / 路径 | step 13–14 平均 loss | step 15–16 平均 loss |
| --- | ---: | ---: |
| 原生 CUDA / 同进程 | 1.49561375 | 1.49188954 |
| 原生 CUDA / 新进程 | 1.49561375 | 1.49188954 |
| strict shim CUDA / 同进程 | 1.49561477 | 1.49189341 |
| strict shim CUDA / 新进程 | 1.49561512 | 1.49189025 |

各 runtime、各恢复路径的末段窗口均低于初段窗口。这里的数据只有一个 preference pair，因此只验证该单样本 profile 中恢复后的批次身份；不据此声称多样本 dataloader/sampler 游标顺序正确。L3 不要求跨实现 loss 逐值相等，本次表格用于展示各自恢复后的固定窗口趋势。

| 层 | 当前基线证据 | 边界 |
| --- | --- | --- |
| L0–L2 | historical pass | 仅适用于上文明确记录的旧 Jittor SHA 和运行键；本节没有重跑这些层。 |
| L3 | pass | 新基线 `67228d5578495ad35f35d157a2e2c61a7d46b5c7` 下完成同进程及新进程恢复；范围仅限固定单条 preference。 |
| L4 | historical pass | 绑定上文旧基线；本次恢复借助公开 CLI，但不是新初态端到端 L4 复验。 |
| L5 | not-run | 本次未测稳态性能；不从恢复作业推断性能。 |

所有运行日志、JSONL、比较结果及 checkpoint 保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-cpo-fixedinit-l3-restore-v2/`；本节只记录已通过的 L3 补验，不表示其他 CPO 配置或 ms-swift 整体完成。

### 当前 SHA 下 fresh-start 公开 CLI 复验（2026-10-12）

新基线为 Jittor `73232b9399042d83d3d352264d5ab0fa4df74f75`，ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。运行键 `20261012-qwen2-cpo-fixedinit-l0l2l4-v5` / job `17972` 的 native 侧在 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）完成公开 CPO CLI 16 步并保存 checkpoint-16；同一 job 的 shim 侧在首次 backward 前失败，stack 中出现两层相同的 `capture_training_step`。诊断键 `20261012-qwen2-cpo-fixedinit-l2-minimal-shim-v1` / job `17980`（无捕获插件）和 `20261012-qwen2-cpo-fixedinit-l2-instrumentation-once-v1` / job `17985`（只由 `--external_plugins` 加载一次捕获插件）均在严格 CUDA 下完成一步并保存 checkpoint-1；后者采集到 336 个梯度及 300 项实际更新。该对照支持重复安装审计 monkeypatch 是 job 17972 首步失败原因，不指向 Jittor CPO backward 缺陷。

新键 `20261012-qwen2-cpo-fixedinit-l0l2l4-shimonly-v1` / job `17994` 只运行 strict shim 16 步，复用 job 17972 native 侧的只读采集文件；模型、初始 adapter、数据 SHA256 与前述固定 profile 一致，分别为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`、`4033d97f2838b3bdebfb8fbaea2b4d582fa24d12472eae460545afe97172fc4e`、`5fd06bff17bc409ce5d69325b68abcef5bdd4d9549ba317746c1d0be051808e4`。该侧在 RTX 4090（UUID `GPU-d813000b-1bfa-885e-cd61-0de97b365732`，driver `580.178.04`）经公开 `swift rlhf --rlhf_type cpo` 完成 16 步并保存 checkpoint-16。native/shim 初始 336 个 trainable adapter 张量逐项相同，16 步输入张量逐步相同；每步 336 个梯度均存在、在 CUDA、有限且非零，step 1 更新 300 项，step 2–16 各更新 336 项，SGD(m=0) 的状态项为空。两侧 loss 窗口均下降：native step 1–3 / 14–16 均值 `1.51681169 / 1.49282102`，strict shim 为 `1.51681133 / 1.49282189`。首步 forward 输出结构、shape、dtype 与有限性通过；最大 logits 绝对差 `8.82149e-5`、相对 L2 `2.39016e-6` 仅作诊断。离线比较器输出 `CPO_FIXEDINIT_CONVERGENCE_AUDIT_PASS`；worker 在训练前后确认 core `.so` 清单 SHA 相同且 no-build bootstrap 未触发 core 冷编译。runtime 日志记录两个 CLI 进程均 `cuda=1`、strict shim 标记真、`fallback=0`。

Slurm job 17994 最终被 harness 标成 `FAILED(exit1)`，唯一失败发生在训练和比较器均成功之后：末尾脚本检查了不存在的 `shim-runtime.log`，而本运行实际 runtime 文件为 `runtime.jsonl`。训练、checkpoint、16 步采集、比较器和 core 哈希比较均已完成；runtime 文件内容直接核实为两个进程 start/end、`fallback=0`。该退出码是 post-check 路径拼写错误，不是训练兼容失败。原始状态与未修改的日志保留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-cpo-fixedinit-l0l2l4-shimonly-v1/`。

| 层 | 当前 SHA 下状态 | 证据边界 |
| --- | --- | --- |
| L0 | pass（绑定当前代码 tree `73232b9`） | job 18025/18029 完整核对 626 个参数、627 个 state 键和 1 个 buffer 的键/shape/dtype/device；336 个 trainable 参数元数据相同。唯一独立记录的差异是 RoPE buffer 的 `requires_grad`，见下节。 |
| L1 | pass（绑定当前代码 tree `73232b9`） | native/shim 首步前向输出结构、shape、dtype、有限性有效；16 步输入身份相同，误差仅作诊断。 |
| L2 | pass（绑定当前代码 tree `73232b9`） | 两侧各 16 步，全部 trainable 梯度有限非零、真实更新且各自 loss 窗口下降；跨 runtime 训练数值只作诊断。 |
| L3 | pass（绑定 `67228d557`） | 仅引用前节已通过的固定单样本同/新进程恢复；本节 fresh-start 与 L0 补审没有重跑恢复。 |
| L4 | pass（绑定当前代码 tree `73232b9`） | native 与 strict shim 的公开 CPO CLI 均完成 16 步并保存 checkpoint-16，严格 CUDA、fallback=0。 |
| L5 | not-run | 没有做不含逐步张量采集的稳态性能测量；不同 GPU UUID，也不作性能比较。 |

此补验仅覆盖 Qwen2-0.5B、固定单偏好对、PEFT LoRA、FP32/eager、SGD(momentum=0) 的单卡公开 CPO profile；不代表多样本 sampler、其他 CPO 配置或 ms-swift 整体通过。

### 当前代码基线固定初态 L0 完整 inventory（2026-10-12）

本节补齐 fresh-start 章节的 L0 缺口，不重跑 job 17972/17994 的训练、原生 oracle 或 checkpoint。运行键 `20261012-qwen2-cpo-fixedinit-l0-inventory-v1` / job `18025` 在 `cscg-qh04` RTX 4090（UUID `GPU-d813000b-1bfa-885e-cd61-0de97b365732`）分别构造 native 与 strict-shim 的 `SwiftRLHF` CPO pipeline、Qwen2-0.5B、tokenizer/template、固定单行数据、PEFT LoRA、CPOTrainer、SGD 与 scheduler；只枚举状态元数据，不调用 `train()`、不执行前向/反向，也不写 checkpoint。Jittor 运行 HEAD 为 `4f1f06b6c36def432d492259cb317ea013c6e11a`；`python/jittor` tree `b066dc9e303d2d3e7eb2c4bd4a824ab68657e835` 与 `compat` tree `558002060ff0d29791daa7402c2ecc6fcd5fa49e` 和训练基线 `73232b9399042d83d3d352264d5ab0fa4df74f75` 一致；ms-swift 为 `88d727951203256baa564c643c651b6f8d90fd7e`。模型、初始 adapter 和数据 SHA 与固定 profile 一致。两侧均为 RTX 4090、FP32、626 个参数、336 个 trainable 参数/2,199,552 个元素、627 个 state 键、1 个 buffer、单行数据、相同 optimizer 参数分组及 lr/weight-decay、scheduler 状态；完整 parameter 元数据及 state 键的 shape/dtype/device 逐项一致。strict shim `use_cuda=1`、`fallback=0`，native 与 shim 独立标记正确。

初版比较器将 buffer `requires_grad` 也当成硬匹配条件，导致 job 18025 在两个构造清单均成功后以 exit 1 结束；只读复核键 `20261012-qwen2-cpo-fixedinit-l0-inventory-compare-v1` / job `18029` 按本 Skill 的 L0 键/shape/dtype/device 合同重比，确认全量 inventory 通过，输入 JSON SHA 前后不变。唯一 buffer `base_model.model.model.rotary_emb.inv_freq` 的 `requires_grad` 为 native `false`、shim `true`，但 buffer 键、shape `[32]`、`torch.float32` 与 `cuda:0` 均一致；336 个 trainable 参数的 `requires_grad` 完全一致。此标志差异保留为结构元数据边界，不据此推断其对其他任务、梯度或恢复语义的影响。该差异与已有 Qwen2 seq-cls 报告的同名 buffer 观察相符。

worker 预检通过用户 NFS 导出上的 64 KiB fsync、精确输入摘要、匹配 Jittor/compat tree、现有 CUDA cache 路径及 no-build bootstrap。job 18029 前后 `jittor_core`/`jit_utils_core` SHA 不变，无 core 冷编译；其 CPU JSON inventory comparator 不构成额外 CUDA 数值证据。完整 JSON、脚本、日志均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-cpo-fixedinit-l0-inventory-v1/` 和 `20261012-qwen2-cpo-fixedinit-l0-inventory-compare-v1/`。fresh-start 当前代码 tree 的 L0–L2/L4 现在可提升；L3 仍绑定其原先已通过的 `67228d557`，L5 仍未运行。
