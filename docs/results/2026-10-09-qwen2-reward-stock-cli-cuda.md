# Qwen2-0.5B stock Reward Model CLI CUDA 入口审计

- 状态：固定 Qwen2 RM stock CLI 配置的 L0–L5 通过。2026-10-12 sampler、首批 logits、六步梯度/更新与收敛、同进程/新进程完整恢复及稳态性能均有独立 Slurm 证据。job17482 的旧格式断言错误仍按原结论保留，不用它覆盖后续通过的限定证据。不代表 RewardTrainer 或 ms-swift 整体兼容。
- 日期：2026-10-09；当前基线补验：2026-10-11、2026-10-12。
- 基线：历史严格对拍 Jittor `dfde300e618dffcf38965b08777cae67c641476c`；2026-10-11 追踪运行 Jittor `487935592dd6b42cc9cde4212c7713906beee819`；2026-10-12 sampler 修复代码提交 `5f70e7b3321abe2e131a9bb0557b27cf041b037a`；L2 验证 Jittor `3c6ac47f5a69aa0685dc5ea4615f9edd442f6247`；L0 验证 Jittor `60c62e9ce976bca3228fa644d8481918caaa0780`；L3 验证 Jittor `e6def72adad11aeb3a1ff924d5f99aa7eed950c5`（upstream `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B `Qwen2ForSequenceClassification`、公开单卡 `swift rlhf --rlhf_type rm`、全参数 FP32/eager、SGD、六行固定 preference 数据、batch 2；输入追踪和首批 L1 使用三步轨迹，新 L2 使用六步并固定 step1–2/5–6 loss 窗口。
- 维护者：ms-swift CUDA 适配。
- 复查条件：若 sampler、Trainer/checkpoint 实现改变，重新核对恢复批次和状态；若模型、数据、设备或计时口径改变，重新测固定 profile 的稳态性能。

初始模型由原生 PyTorch 从固定 Qwen2 checkpoint 构造，确定性初始化 `score.weight` 后保存；同一 safetensors 被原生与候选 CLI 读取，SHA256 为 `415ca5ddc706cea55ef2a66850061291272c744bd5e1acdca6eeddc8d4035a1b`。底模文件 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。六条 preference JSONL SHA256 为 `dc072b5ee211afbb4c6e8f1a1f5689d7f4e5a98f4f79c86b70d4d39a4608eee4`。ms-swift 为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6。

Slurm 13761 在 `cscg-qh04` RTX 4090（UUID `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44`，driver `580.178.04`）先运行原生、再运行 strict shim。strict shim 使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；父子 CLI 进程均记录 `use_cuda=1`、fallback 0。两侧 CLI 均保存 `checkpoint-3`。全模型 291 个 state 键 shape 一致且 finite；末态最大绝对差 `1.0155141e-4`、相对 L2 `1.1141148e-6`。但第 2/3 步 loss 轨迹不一致，最大差 `0.0133114`。该运行没有保存实际批次输入，不能把 loss 分叉解释成模型数值误差。

Slurm 13784 在 `cscg-qh17` RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）复跑相同 checkpoint、数据与 CLI，并显式传入 `--dataset_shuffle false --train_dataloader_shuffle false`。外部审计插件仅记录 `RewardTrainer.compute_loss` 收到的 CUDA `input_ids` 与 `attention_mask`；两边各记录三步、记录中的张量都在 CUDA、输出 shape 结构相同，但逐步输入记录不相等。最终权重差和 loss 最大差与首轮基本相同（loss `0.0133115`）。候选进程仍为 strict CUDA 且 fallback 0。因为第二轮已确认数据身份未对齐，后续层的数值比较不能作为同输入对拍。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | pass（当前固定构造配置） | job17852 在 RTX 4090 上先 native 后 strict shim，直接走相同 `SwiftRLHF` 构造序列，止于训练前；两侧构造 Qwen2ForSequenceClassification、processor、template、长度5数据集、3 batch DataLoader、RewardTrainer、SGD 两参数组和 scheduler。291 个参数均为 `float32`、`cuda:0`，291 个 state 键也均在 `cuda:0`，494,033,664 个可训练标量；参数与 state 键清单及逐参数值 SHA256 完全匹配。job17862 在 GPU worker 对两份清单和 optimizer 组成员/显式 lr、weight_decay 做归一化比较通过；shim runtime `use_cuda=1`、fallback=0；没有执行 forward/optimizer step 或保存 checkpoint。该 L0 只覆盖报告中锁定的固定 RM 构造配置。 |
| L1 | pass（首批） | job17601 在同一 RTX 4090 上先 native、后 strict shim，各跑公开 CLI 一步；首批 `input_ids`/`attention_mask` 的 shape、dtype、CUDA device 与 SHA 全等，也与旧 native 输入 oracle 和 sampler 修复后 shim 输入记录首批 SHA 一致。score logits shape `[4,1]`、CUDA `float32`、finite；`max_abs=3.8147e-6`、`max_rel=8.8035e-5`，满足预设 `atol=1e-5, rtol=1e-4`。该结论限定于此首批前向。 |
| L2 | pass（固定六步 CLI 配置） | job17714 两侧每步覆盖291个可训练参数的 CUDA 梯度与更新；step1–2和5–6预锁窗口各自下降；step-6模型、optimizer checkpoint及序列化公共状态审计通过。 |
| L3 | pass（固定 step-2 中途恢复配置） | job17897 在 `checkpoint-2`（global step 2、epoch `2/3`）保存完整 model/optimizer/scheduler/RNG/trainer state；native 与 strict shim 均在同进程复用同一 Trainer 恢复到 step 6，并在独立新进程从 step 2 恢复到 step 6。恢复后 step 3–6 的输入/mask SHA 与各自旧 L2 oracle 及两侧互相逐步一致；291 个可训练参数的梯度均存在且有限，更新非空；scheduler `last_epoch=2`、optimizer 两组参数数 `170/121` 与 lr/weight decay 恢复，SGD momentum=0 因此 state slots 按预期为空；step 3 native/shim RNG fingerprint 各自同/新进程一致。job17898 在训练回调中逐项比较了291个模型张量与对应 step-2 checkpoint，native 与 shim 各自精确相等；job17899 随后只读复核这些结果并校验四组 strict-shim 父子进程 scope 计数，全部通过；唯一 buffer `model.rotary_emb.inv_freq` 位于 `cuda:0`、未持久化但同进程与新进程指纹相同。两侧 step 1–2 均值 `0.62229335` / `0.62229320`，step 5–6 均值 `0.54605728` / `0.54605645`（fresh shim `0.54605660`），各自末段下降。 |
| L4 | pass（固定 stock CLI profile） | 通过公开入口 `python -m swift.cli.main rlhf --rlhf_type rm` 完成构造、首批 forward、六步训练和从 step-2 checkpoint 同/新进程恢复；保留输出/checkpoint。strict-shim 父子进程 scope 均为 CUDA 且零 fallback。结论仅覆盖本报告锁定配置。 |
| L5 | pass（固定 stock CLI profile） | job17916 native 与 job17922 strict shim 均完成6步真实形状预热及step 7–16的10次同步训练步+optimizer计时；两侧预热及测量输入身份逐项相同，均在同一 RTX 4090 UUID 上先后使用独立 Slurm allocation。严格 CUDA、零 fallback、训练参数有更新，shim 预热后的 JIT 缓存稳定。完整性能数据与离群值见下方记录；不把训练墙钟或首次 JIT 计入稳态。 |

## 2026-10-11 当前基线采样顺序追踪

在不重跑旧运行键的前提下，新运行键 `20261011-qwen2-reward-stock-cli-index-trace-v1` / job16181 和 `20261011-qwen2-reward-stock-cli-loader-trace-v1` / job16186，于 `cscg-qh04` 同一 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）再次依次运行 native 与 strict shim 公共 CLI。两次均完成三步；job16181 实收输入哈希与历史 r2 完全相同，job16186 同时记录 DataLoader 结构、实际 sampler index 和 loss hook 输入。候选父子进程均 `use_cuda=1`、`fallback=0`。v1 的首个 sampler hook 未挂到实际对象，没有产生索引日志；v2 从实际返回的 loader 对象包裹 batch sampler，采集有效索引。所有原始脚本、日志、JSONL 与 checkpoint 均在 state，未版本化。

job16186 中两侧 `seed=42`、`data_seed=42`、`train_dataloader_shuffle=false`，但有效 loader 使用 Accelerate `SeedableRandomSampler`；实际 sampler 长度为 5。原生索引依次为 `[2,4]`、`[3,0]`、`[1]`，strict shim 为 `[4,2]`、`[3,1]`、`[0]`，且每步 `compute_loss` 实收 CUDA `input_ids`/mask 的哈希仍不相等。原 JSONL 有 6 行而 sampler 只迭代5个索引；其 data source 长度与哪条记录在 lazy dataset 构造时被排除尚未直接采集，作为未定位边界保留。两侧 sampler 输出的索引集合均为 `0..4`，顺序不同已足以解释批次身份不一致。

静态源码追踪显示 `swift.rlhf_trainers.reward_trainer.RewardTrainer` 的基类为 `RLHFTrainerMixin, SwiftMixin, HFRewardTrainer`；它没有继承定义自有 `train_dataloader_shuffle` sampler 的 `DataLoaderMixin`，所以 CLI 保存的 `false` 未阻止有效 loader 的随机采样。Accelerate `SeedableRandomSampler` 对两侧使用同一 data seed，并经 `torch.randperm(generator=...)` 取样；shim 的 `torch.Generator` stream 当前由 `numpy.random.default_rng` 提供，`randperm` 使用 `Generator.permutation`。这解释了同 seed 下实测索引排列差异。该路径同时暴露 ms-swift RewardTrainer 的 shuffle 参数未作用于此继承链，以及 shim `Generator`/`randperm` 未复刻 PyTorch 的 CPU 随机序列；本轮只记录责任边界，未修改 ms-swift 或 shim 源码。

以下是该次 sampler 追踪阶段的层级快照，不是当前结论：L0 partial（两侧公开 CLI 与 CUDA 参数/输入可运行，但缺完整两侧初态 dtype/device 清单）；L1 partial（2026-10-12 已证明三批输入身份一致，但没有保存可对拍的 logits）；L2 blocked（尚无逐 trainable 参数梯度、optimizer 状态/逐步更新审计，也没有按固定窗口验证两侧 loss 趋势；跨实现逐值容差不再是门槛）；L3 not-run；L4 入口已运行、功能未通过；L5 blocked。随后完成的各层证据与当前状态见本文顶部矩阵及后续分层记录。该固定配置结论不覆盖其它 RewardTrainer、RM 数据格式、采样设置或 ms-swift 训练面。

仓库门禁：Slurm 13785 的布局与 Manifest 生成/校验通过。完整 `tests/structure` 为 1383 passed、8 skipped、1 failed；唯一失败是 CPU matmul 测试需要 oneDNN v3，而 GPU worker 下载源码时到 `codeload.github.com` 连接被拒。Slurm 13720、13738 已在前序文档验证中遇到同一环境限制；本次未把完整结构门禁标记为通过。

运行键 `20261009-qwen2-reward-stock-cli-r1`、`20261009-qwen2-reward-stock-cli-r2`、`20261011-qwen2-reward-stock-cli-index-trace-v1` 与 `20261011-qwen2-reward-stock-cli-loader-trace-v1` 保存脚本、checkpoint、原始输入、索引和日志，均未版本化。r2 使用上一运行固定初始模型；追踪作业顺序复用与源码/ABI/工具链/设备架构相符的缓存。报告只确认上述单卡固定配置。

## 2026-10-12 当前 compat 基线 sampler 修复复验

修复 `torch.Generator`/CPU `torch.randperm` 兼容的提交 `5f70e7b33` 已进入当前树。新运行键 `20261012-qwen2-reward-stock-cli-randperm-fix-v2` / job17482 只运行 strict shim 的同一公开 CLI 配置；复用 job16186 中相同数据 SHA、初始权重 SHA、seed/data_seed、batch、优化器和三步原生 oracle，没有重跑 native。真实 RTX 4090 worker 为 cscg-qh04 / UUID `GPU-d813000b-1bfa-885e-cd61-0de97b365732`。strict shim CLI 实际完成 step 1–3，保存 `checkpoint-3`，trainer `global_step=3`；step loss 为 `0.63318640 → 0.61140037 → 0.57950783`。这只是三步轨迹记录，不替代预先固定的 L2 收敛窗口。

GPU-worker 只读工件审计 v3/job17523 记录 native 与 shim 的 loader 索引都为 `[[2,4],[3,0],[1]]`，三步 `input_ids` 和 `attention_mask` SHA256 逐项完全相等；strict CUDA 父子进程各记录 `use_cuda=1`、`fallback=0`。job17482 的训练本身到 checkpoint 保存和 CLI 正常结束，但 Slurm 退出码为 1，原因是训练后的脚本误把合法的 `start ... cuda=1 visible=0 fallback=0` 日志行当成格式错误。审计脚本另有已定位的格式断言错误：job17523 误把 4 条 `logging.jsonl`（3 个 step 加 1 条 summary）要求为 3；fail-closed job17528 明确报告仅该项不匹配，其他批次/输入、checkpoint、runtime 与 core SHA 核验通过。job17532 的审计脚本在日志字段检查处失败于其自身 jq 输入传参错误。以上 job/运行键和原始材料均保留，不作为兼容失败或完整验收通过；其独立输出支持本节限定的索引、输入哈希、step-3 结束及 fallback 观察。核心 `jittor_core` SHA 在 bootstrap 前后和事后审计中保持不变；JIT operator 增量不混同为 core 重编译。

此前两个短尝试也保留：job17480 在导入前因 worker `df` 参数不兼容终止；job17490 的只读审计引用了未使用的 `sampler-batches` 文件名。它们均未启动模型或改变缓存。所有运行产物均在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-reward-stock-cli-randperm-fix-v2/` 及 `...-artifact-audit-v*/`，未版本化。新的准入边界是：RewardTrainer 这组三批次 sampler/输入身份分叉已有当前版本证据显示消失；完整 L1/L2/L3 和更广的 Reward/ms-swift 覆盖仍未验。
## 2026-10-12 Reward 首批 logits 对拍

运行键 `20261012-qwen2-reward-stock-cli-l1-logits-v2` / Slurm job17601 修复了 v1/job17596 在模型前向前遇到的 harness 路径错误。v1 使用过长的持久 state 路径作为 `TMPDIR`，ms-swift 数据 `map(num_proc=1)` 创建 multiprocessing Unix socket 时失败于 `OSError: AF_UNIX path too long`；该键没有模型计算、没有原生 oracle，也未启动 shim。v2 将 `TMPDIR` 指向短路径 `/tmp/q2rm-l1-logits-$SLURM_JOB_ID`，该路径软链接仍指向持久 state 的 run `tmp/`；没有清理或挪动模型、缓存或实验产物。

job17601 在 `cscg-qh04` NVIDIA RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，driver `580.178.04`）按 native 后 strict shim 顺序，各执行公开 `swift rlhf --rlhf_type rm` 一步。两侧加载相同底模 SHA256 `9cd8fc8c…77cad` 与初始 safetensors SHA256 `415ca5dd…35a1b`，使用相同六行数据 SHA256 `dc072b5e…08eee4`、seed/data_seed 42、FP32/eager、batch 2/max length 32。采集器只将原有 `RewardTrainer.compute_loss` 调用改为 `return_outputs=True` 以记录其已计算的 chosen/rejected scores，随后按原调用约定只返回 loss；模型公式、输入和训练路径不变。双方首批 `input_ids` SHA256 `2a140ca4…361b87` 与 `attention_mask` SHA256 `c8b517ed…b7f6f8` 相同，且逐项匹配旧 native job16186 与 sampler 修复后 strict-shim job17482 的首批记录；输入 shape 均为 `[4,32]`、dtype `int64`、设备 `cuda:0`。

两侧模型 score logits 都是 `[4,1]`、CUDA `float32` 且 finite。Native 值为 `[-1.51895797, -0.02369708, -1.70691741, -0.08534986]`，strict shim 值为 `[-1.51895499, -0.02369499, -1.70691359, -0.08534890]`；`max_abs=3.8146973e-6`，以 `max(|native|,|shim|)` 为分母的 `max_rel=8.8034590e-5`。所有元素满足 `abs(diff) <= 1e-5 + 1e-4*abs(native)`。Strict-shim 两进程均 `use_cuda=1`、fallback delta=0；`jittor_core` SHA 前后完全相同。两端均完成 Trainer 单步且未写 checkpoint；该单次前向不构成 L2 收敛/全梯度验收，后续 job17714 的独立六步训练和审计才通过 L2，不改变 L0/L3/L4/L5 边界。原始结果、完整值与采集脚本未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-reward-stock-cli-l1-logits-v1/`、`...-v2/`；v1 保留为前向前的 harness 失败证据。

## 2026-10-12 Reward CLI 全参数 L2 与收敛窗口

新运行键 `20261012-qwen2-reward-stock-cli-l2-fullgrad-v1` / job17714 使用相同公开 RM CLI、Qwen2-0.5B 初始 safetensors、六行 preference 数据、FP32/eager、batch 2、SGD `lr=1e-5`、seed/data_seed 42，运行六步。收敛口径在作业前固定为 step 1–2 对 step 5–6 的平均 loss。Native 先运行，strict shim 后运行；外部插件只记录 `RewardTrainer.compute_loss` 收到的输入身份，并在 Transformers `on_pre_optimizer_step`/`on_step_end` 回调汇总梯度和参数差，不改 loss、数据或 optimizer 调用。原始逐参数记录、日志与 checkpoint 留在 state，未版本化。

两侧每步都有291个可训练参数；必需梯度均存在、位于 CUDA、有限且非零，逐步摘要覆盖全部参数。逐步更新均有限且非空，变化的参数张量数依次为 native/shim 各自的 `290, 290, 291, 291, 290, 289`。六步 `input_ids` 与 `attention_mask` 的 shape、dtype、device 和 SHA256 逐步一致。Native loss step1–6 为 `0.63318646, 0.61140025, 0.57950896, 0.45711076, 0.53535688, 0.55675769`；strict shim 为 `0.63318640, 0.61140001, 0.57950920, 0.45711112, 0.53535432, 0.55675828`。预锁 step1–2 均值分别为 `0.62229335` / `0.62229320`，step5–6 均值分别为 `0.54605728` / `0.54605630`；两个 runtime 各自窗口均下降。

step-6 checkpoint 的291个模型键均有限且相对初始模型均发生改变；每份权重共494,033,664个标量，native/shim 分别改变493,481,997/493,481,987个，最大绝对差分量均为 `2.5737472e-4`。两份 checkpoint 均有模型、optimizer、scheduler 与 `trainer_state` 文件，`global_step=6`。只读审计 job17756 使用 PyTorch `weights_only=True` 读取原生 optimizer，使用拒绝所有全局对象的受限 pickle reader 读取 strict-shim 便携状态；两边公共 state dict 完全相等，state 为空（SGD momentum=0 的预期），两组参数覆盖数为170/121，`lr=1e-5`、weight decay 为0.1/0.0且选项一致。optimizer 文件字节数不同（1768/894）对应不同序列化格式，不表示状态差异。

job17714 的最终 Slurm exit 1 来自训练后的首版比较器错误假设 optimizer 只有一个参数组；训练本身两侧均完成六步并保存 checkpoint。修正为检查两组公开参数选项后，独立 artifact audit job17739 通过全部 L2 检查，optimizer checkpoint 又由 job17756 通过。GPU worker 为 `cscg-qh04` RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，driver `580.178.04`）；strict-shim 父子进程 `use_cuda=1`、fallback delta=0。复用匹配的 CUDA JIT 缓存，`JITTOR_NO_BUILD=1` bootstrap 在2.9秒通过，`jittor_core` SHA256 前后均为 `caf12a34ff55c770651ef2b95c9f1ceda48599496bec59aeae8e36c6f994dbd5`。此结论仅验收当前固定 Reward RM CLI 的 L2 配置，不提升 L0/L3/L4/L5，也不代表更广 Reward 或 ms-swift 支持。

## 2026-10-12 Reward RM 构造阶段全参数清单

新运行键 `20261012-qwen2-reward-stock-cli-l0-construct-v1` / job17852 复用同一 Qwen2-0.5B 底模、确定性初始 safetensors、六行 preference 数据及 L2 固定参数。它直接构造 `SwiftRLHF`、template、数据集、全参数 tuner、RewardTrainer、SGD optimizer、scheduler 与训练 DataLoader，并在 step 0 停止；不执行前向、训练步骤或 checkpoint 保存。作业在 `cscg-qh04` RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，driver `580.178.04`）顺序运行独立 native 与 strict-shim Python 进程。两边模型类、processor/template 类、task type、device map、dataset/loader 结构与 scheduler 状态匹配。

291 个参数均位于 `cuda:0` 且 dtype 为 `torch.float32`；291 个 state 键也均在 `cuda:0`，两侧 state 名、shape、dtype、device 与值 SHA256 完全匹配。可训练标量数为494,033,664。逐项参数的名称、shape、dtype、device、requires_grad 及值 SHA256 完全匹配，说明两侧加载了同一初态。两组 optimizer 的参数成员顺序一致，组大小为170/121，显式 `lr=1e-5`、`initial_lr=1e-5`、weight decay分别为0.1/0.0；optimizer 初始 state 为空，scheduler 初态相同。strict-shim `sitecustomize` 作用域记录 `use_cuda=1`、父进程 fallback=0；匹配缓存下 `JITTOR_NO_BUILD=1` bootstrap通过且 `jittor_core` SHA前后不变。比较器不比较原生与 shim 的 optimizer 类名，也不把各自 `param_groups` 内部表示强求相同；既有 L2 optimizer 工件审计负责完整公共 optimizer 状态/选项。

job17852 的训练后比较器因比较 backend-specific optimizer 类/隐式默认及内部字段而以 exit 1 结束，但两侧构造和清单已写出。job17859 仅在 GPU worker 诊断组差异：两侧组成员和显式选项一致；原生把 momentum、dampening、nesterov、maximize 默认项放在参数组，shim 的 SGD 将默认值保存在 optimizer 实例上。job17862 在 GPU worker 用新运行键对原始清单作语义归一化比较并通过。为定位 job17852 比较器字段，曾在登录节点对 GPU 产出的 JSON 文件作一次只读字段诊断；未导入框架、未加载模型或运行计算，该诊断不计验收证据。最终验收比较由 Slurm GPU job17862 完成。原始 JSON、脚本与 Slurm 日志均保存在 state，未版本化。该段记录的是当时补齐 L0 后的状态快照；L3/L4 后续状态见顶部矩阵及后续记录，不改变本固定配置范围，也不代表其它 Reward/ms-swift 工作负载。

## 2026-10-12 Reward RM 中途 checkpoint 恢复

新运行键 `20261012-qwen2-reward-stock-cli-l3-resume-v1` / Slurm job17897 在 `cscg-qh04` RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，driver `580.178.04`）按 native 后 strict shim 顺序运行同一六行数据、初始模型、batch 2、FP32/eager、全参数 SGD `lr=1e-5` 配置。每侧先训练到 step 2 保存 epoch `2/3` 的中途 checkpoint，再在同一进程复用同一 RewardTrainer 恢复并继续至 step 6；随后由不同 CLI 进程从该 step-2 checkpoint 恢复至 step 6。恢复阶段关闭额外 checkpoint 保存，原始日志、逐参数梯度/更新、输入、RNG、callback 状态与 step-2 checkpoint 均留在 state，未版本化。

checkpoint 包含 `model.safetensors`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth` 和 `trainer_state.json`。恢复 callback 在训练继续前看到 `global_step=2`、`epoch=2/3`；两组 optimizer 参数数为170/121、lr均为`1e-5`、weight decay为0.1/0.0，scheduler `last_epoch=2`。SGD momentum为0，optimizer state slots 为空是该配置的预期。native 与 shim 的 step3–6 `input_ids`/`attention_mask` 哈希都与各自已完成的 L2 trace 逐项相同，且两侧互相相同；step 3 的 CUDA RNG 指纹在各自同进程与新进程恢复后相同。恢复后每一步291个 trainable 参数梯度均存在、有限，更新有限且非空。Native step1–2/5–6 平均 loss 为`0.62229335`/`0.54605728`；shim 为`0.62229320`/`0.54605645`（fresh process 末段`0.54605660`），每个运行时各自的末段低于初段。作业位于 `cscg-qh04` RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，driver `580.178.04`）；strict-shim 父子进程 `use_cuda=1`、fallback=0，匹配缓存下 bootstrap `--check` 通过，`jittor_core` SHA前后均为`caf12a34ff55c770651ef2b95c9f1ceda48599496bec59aeae8e36c6f994dbd5`。旧 L2 原生轨迹未重跑；native L3 为生成 step-2 oracle checkpoint 而按新运行键执行，恢复后批次与已存 L2 oracle 对照。

为了直接核实模型状态，补充运行键 `20261012-qwen2-reward-stock-cli-l3-state-audit-v1` 的 job17898 只恢复既有 step-2 checkpoint，并在 native、shim 的同进程/新进程恢复 callback 处逐项比较291个模型 state tensor 与 checkpoint；所有值均精确相等。唯一命名 buffer `model.rotary_emb.inv_freq` 位于`cuda:0`且不持久化，其 shape/dtype/value fingerprint 在每次恢复相同。job17898 的训练和这些比较均完成；其最终 scope 计数断言误把两次 shim CLI 的父子进程记录预期为2组，导致 Slurm exit 1。只读复核运行键 `20261012-qwen2-reward-stock-cli-l3-state-audit-check-v1` / job17899 将计数修正为实际4组后通过：四组 `use_cuda=1`、`fallback=0`，291项模型比较和buffer指纹均通过，`jittor_core` SHA前后未变。该失败属于验收脚本计数错误，不是训练或恢复失败；没有重跑模型训练来修复它。

L3 与本固定 stock CLI 的 L4 至此通过；后续固定 profile 的 L5 证据见下节。其它 RewardTrainer API、Reward 数据格式、采样配置和 ms-swift 任务均不在此结论范围内。

## 2026-10-12 固定 Reward RM CLI 稳态性能

新运行键 `20261012-qwen2-reward-stock-cli-l5-steady-v10` 的 strict-shim job17922 复用只读 native 基线 `20261012-qwen2-reward-stock-cli-l5-steady-v7` / job17916；没有重跑 native。job17916 的 native 阶段完整写出10次样本；该job后续strict-shim阶段因旧CLI守卫提前退出，故其整体Slurm状态失败，但native JSON仍由摘要锁定并只读复用。两作业先后运行于 `cscg-qh04` 的同一 RTX 4090 UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，但属于不同 Slurm allocation。相同固定模型、数据、公开 `swift rlhf --rlhf_type rm` 参数、初态与step 1–16输入轨迹均经摘要校验；双方各以step 1–6覆盖真实输入形状预热，计时step 7–16共10次。计时区间为同步后的 RewardTrainer 训练步及 optimizer callback；shim 预热后 JIT `.so` 数量保持154，`jittor_core` SHA256=`caf12a34ff55c770651ef2b95c9f1ceda48599496bec59aeae8e36c6f994dbd5`且未变。strict-shim 父子进程均为CUDA、fallback=0，tracked 参数确有更新，输入身份完全相同。

Native 延迟 median/mean/P95（最近秩）为 `79.177/79.125/79.831 ms`，吞吐为 `21.485 preference pairs/s` 与 `1291.623 nonpadding tokens/s`；strict shim 为 `62.754/104.384/478.475 ms`、`16.286 pairs/s` 与 `979.079 tokens/s`。Shim十次样本保留全部原值；step 11 的 `478.475 ms` 是明显离群值，故 P95 和均值包含它，未剔除。中位延迟比值 native/shim 为 `1.262`，shim/native pair吞吐比为 `0.758`。PyTorch CUDA allocator 峰值 allocated/reserved 分别为 native `5,960,430,592/6,341,787,648 bytes`、shim `5,928,867,840/6,430,916,608 bytes`；该统计口径不等同于Jittor总显存占用。结论只覆盖本报告锁定的固定 Reward RM CLI profile，不推广到其它 RewardTrainer 或 ms-swift 面。
