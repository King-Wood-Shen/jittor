# Qwen2-0.5B stock Reward Model CLI CUDA 入口审计

- 状态：原生与 strict shim 均完成公开 `swift rlhf --rlhf_type rm` 三步 CLI。2026-10-12 新 shim 复验的三批 sampler 索引与逐批输入哈希和既有原生 oracle 一致，确认先前顺序分叉在当前 compat 基线已消失；job17482 的 Slurm 终态因训练后格式断言失败，不能将整个运行键标为自动门禁通过。新单步键 job17601 对齐首批输入并通过同权重 CUDA reward logits 前向，故 L1 首批证据通过；L0 仍 partial、L2 blocked（缺全量参数梯度/更新与收敛窗口）、L3 not-run。L4 公开入口已运行并保存 step-3 checkpoint，但功能层仍未通过；L5 blocked。不代表 RewardTrainer 或 ms-swift 整体兼容。
- 日期：2026-10-09；当前基线补验：2026-10-11、2026-10-12。
- 基线：历史严格对拍 Jittor `dfde300e618dffcf38965b08777cae67c641476c`；2026-10-11 追踪运行 Jittor `487935592dd6b42cc9cde4212c7713906beee819`；2026-10-12 sampler 修复代码提交 `5f70e7b3321abe2e131a9bb0557b27cf041b037a`；验证时 Jittor HEAD `84ba71788b1bfeae461bd1d13e631ba6d1166c27`（文档提交后代；上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B `Qwen2ForSequenceClassification`、公开单卡 `swift rlhf --rlhf_type rm`、全参数 FP32/eager、SGD、六组固定 preference 数据、batch 2、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补齐两侧逐 trainable 梯度/真实更新和预先固定窗口 loss 证据，再评估恢复和性能层；若 sampler 实现改变，重新核对批次身份。

初始模型由原生 PyTorch 从固定 Qwen2 checkpoint 构造，确定性初始化 `score.weight` 后保存；同一 safetensors 被原生与候选 CLI 读取，SHA256 为 `415ca5ddc706cea55ef2a66850061291272c744bd5e1acdca6eeddc8d4035a1b`。底模文件 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。六条 preference JSONL SHA256 为 `dc072b5ee211afbb4c6e8f1a1f5689d7f4e5a98f4f79c86b70d4d39a4608eee4`。ms-swift 为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6。

Slurm 13761 在 `cscg-qh04` RTX 4090（UUID `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44`，driver `580.178.04`）先运行原生、再运行 strict shim。strict shim 使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；父子 CLI 进程均记录 `use_cuda=1`、fallback 0。两侧 CLI 均保存 `checkpoint-3`。全模型 291 个 state 键 shape 一致且 finite；末态最大绝对差 `1.0155141e-4`、相对 L2 `1.1141148e-6`。但第 2/3 步 loss 轨迹不一致，最大差 `0.0133114`。该运行没有保存实际批次输入，不能把 loss 分叉解释成模型数值误差。

Slurm 13784 在 `cscg-qh17` RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）复跑相同 checkpoint、数据与 CLI，并显式传入 `--dataset_shuffle false --train_dataloader_shuffle false`。外部审计插件仅记录 `RewardTrainer.compute_loss` 收到的 CUDA `input_ids` 与 `attention_mask`；两边各记录三步、记录中的张量都在 CUDA、输出 shape 结构相同，但逐步输入记录不相等。最终权重差和 loss 最大差与首轮基本相同（loss `0.0133115`）。候选进程仍为 strict CUDA 且 fallback 0。因为第二轮已确认数据身份未对齐，后续层的数值比较不能作为同输入对拍。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 同一确定性初始 safetensors 和 291 个末态键；shim 日志显示全参数 Qwen2ForSequenceClassification、`device_map=cuda:0`，审计到的批输入为 CUDA。2026-10-12 输入身份补验通过；仍缺两侧逐参数初态 dtype/device 清单。 |
| L1 | pass（首批） | job17601 在同一 RTX 4090 上先 native、后 strict shim，各跑公开 CLI 一步；首批 `input_ids`/`attention_mask` 的 shape、dtype、CUDA device 与 SHA 全等，也与旧 native 输入 oracle 和 sampler 修复后 shim 输入记录首批 SHA 一致。score logits shape `[4,1]`、CUDA `float32`、finite；`max_abs=3.8147e-6`、`max_rel=8.8035e-5`，满足预设 `atol=1e-5, rtol=1e-4`。该结论限定于此首批前向。 |
| L2 | blocked | 两侧已有三步训练日志，但没有全部 trainable 参数逐项梯度、optimizer 状态或逐步更新证据，也没有按新协议预先固定的初末 loss 窗口。旧输入顺序问题已消失；数值逐值对拍本身不再是门槛。 |
| L3 | not-run | 没有同进程或新进程恢复轨迹。 |
| L4 | 入口已运行，功能未通过 | 原生 job16186 与 2026-10-12 strict shim CLI 均完成三步并保存 checkpoint；新 shim 端 strict CUDA 父子进程均 `fallback=0`。该 shim 作业的最终 Slurm 失败来自训练后的日志断言格式错误；本次仍缺满足下层门槛的完整验收证据。 |
| L5 | blocked | 前级不通过；训练墙钟含冷 JIT，不作为性能结果。 |

## 2026-10-11 当前基线采样顺序追踪

在不重跑旧运行键的前提下，新运行键 `20261011-qwen2-reward-stock-cli-index-trace-v1` / job16181 和 `20261011-qwen2-reward-stock-cli-loader-trace-v1` / job16186，于 `cscg-qh04` 同一 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）再次依次运行 native 与 strict shim 公共 CLI。两次均完成三步；job16181 实收输入哈希与历史 r2 完全相同，job16186 同时记录 DataLoader 结构、实际 sampler index 和 loss hook 输入。候选父子进程均 `use_cuda=1`、`fallback=0`。v1 的首个 sampler hook 未挂到实际对象，没有产生索引日志；v2 从实际返回的 loader 对象包裹 batch sampler，采集有效索引。所有原始脚本、日志、JSONL 与 checkpoint 均在 state，未版本化。

job16186 中两侧 `seed=42`、`data_seed=42`、`train_dataloader_shuffle=false`，但有效 loader 使用 Accelerate `SeedableRandomSampler`；实际 sampler 长度为 5。原生索引依次为 `[2,4]`、`[3,0]`、`[1]`，strict shim 为 `[4,2]`、`[3,1]`、`[0]`，且每步 `compute_loss` 实收 CUDA `input_ids`/mask 的哈希仍不相等。原 JSONL 有 6 行而 sampler 只迭代5个索引；其 data source 长度与哪条记录在 lazy dataset 构造时被排除尚未直接采集，作为未定位边界保留。两侧 sampler 输出的索引集合均为 `0..4`，顺序不同已足以解释批次身份不一致。

静态源码追踪显示 `swift.rlhf_trainers.reward_trainer.RewardTrainer` 的基类为 `RLHFTrainerMixin, SwiftMixin, HFRewardTrainer`；它没有继承定义自有 `train_dataloader_shuffle` sampler 的 `DataLoaderMixin`，所以 CLI 保存的 `false` 未阻止有效 loader 的随机采样。Accelerate `SeedableRandomSampler` 对两侧使用同一 data seed，并经 `torch.randperm(generator=...)` 取样；shim 的 `torch.Generator` stream 当前由 `numpy.random.default_rng` 提供，`randperm` 使用 `Generator.permutation`。这解释了同 seed 下实测索引排列差异。该路径同时暴露 ms-swift RewardTrainer 的 shuffle 参数未作用于此继承链，以及 shim `Generator`/`randperm` 未复刻 PyTorch 的 CPU 随机序列；本轮只记录责任边界，未修改 ms-swift 或 shim 源码。

等级按当前训练协议复核：L0 partial（两侧公开 CLI 与 CUDA 参数/输入可运行，但缺完整两侧初态 dtype/device 清单）；L1 partial（2026-10-12 已证明三批输入身份一致，但没有保存可对拍的 logits）；L2 blocked（尚无逐 trainable 参数梯度、optimizer 状态/逐步更新审计，也没有按固定窗口验证两侧 loss 趋势；跨实现逐值容差不再是门槛）；L3 not-run；L4 入口已运行、功能未通过；L5 blocked。当前基线结论不覆盖其它 RewardTrainer、RM 数据格式、采样设置或 ms-swift 训练面。

仓库门禁：Slurm 13785 的布局与 Manifest 生成/校验通过。完整 `tests/structure` 为 1383 passed、8 skipped、1 failed；唯一失败是 CPU matmul 测试需要 oneDNN v3，而 GPU worker 下载源码时到 `codeload.github.com` 连接被拒。Slurm 13720、13738 已在前序文档验证中遇到同一环境限制；本次未把完整结构门禁标记为通过。

运行键 `20261009-qwen2-reward-stock-cli-r1`、`20261009-qwen2-reward-stock-cli-r2`、`20261011-qwen2-reward-stock-cli-index-trace-v1` 与 `20261011-qwen2-reward-stock-cli-loader-trace-v1` 保存脚本、checkpoint、原始输入、索引和日志，均未版本化。r2 使用上一运行固定初始模型；追踪作业顺序复用与源码/ABI/工具链/设备架构相符的缓存。报告只确认上述单卡固定配置。

## 2026-10-12 当前 compat 基线 sampler 修复复验

修复 `torch.Generator`/CPU `torch.randperm` 兼容的提交 `5f70e7b33` 已进入当前树。新运行键 `20261012-qwen2-reward-stock-cli-randperm-fix-v2` / job17482 只运行 strict shim 的同一公开 CLI 配置；复用 job16186 中相同数据 SHA、初始权重 SHA、seed/data_seed、batch、优化器和三步原生 oracle，没有重跑 native。真实 RTX 4090 worker 为 cscg-qh04 / UUID `GPU-d813000b-1bfa-885e-cd61-0de97b365732`。strict shim CLI 实际完成 step 1–3，保存 `checkpoint-3`，trainer `global_step=3`；step loss 为 `0.63318640 → 0.61140037 → 0.57950783`。这只是三步轨迹记录，不替代预先固定的 L2 收敛窗口。

GPU-worker 只读工件审计 v3/job17523 记录 native 与 shim 的 loader 索引都为 `[[2,4],[3,0],[1]]`，三步 `input_ids` 和 `attention_mask` SHA256 逐项完全相等；strict CUDA 父子进程各记录 `use_cuda=1`、`fallback=0`。job17482 的训练本身到 checkpoint 保存和 CLI 正常结束，但 Slurm 退出码为 1，原因是训练后的脚本误把合法的 `start ... cuda=1 visible=0 fallback=0` 日志行当成格式错误。审计脚本另有已定位的格式断言错误：job17523 误把 4 条 `logging.jsonl`（3 个 step 加 1 条 summary）要求为 3；fail-closed job17528 明确报告仅该项不匹配，其他批次/输入、checkpoint、runtime 与 core SHA 核验通过。job17532 的审计脚本在日志字段检查处失败于其自身 jq 输入传参错误。以上 job/运行键和原始材料均保留，不作为兼容失败或完整验收通过；其独立输出支持本节限定的索引、输入哈希、step-3 结束及 fallback 观察。核心 `jittor_core` SHA 在 bootstrap 前后和事后审计中保持不变；JIT operator 增量不混同为 core 重编译。

此前两个短尝试也保留：job17480 在导入前因 worker `df` 参数不兼容终止；job17490 的只读审计引用了未使用的 `sampler-batches` 文件名。它们均未启动模型或改变缓存。所有运行产物均在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-reward-stock-cli-randperm-fix-v2/` 及 `...-artifact-audit-v*/`，未版本化。新的准入边界是：RewardTrainer 这组三批次 sampler/输入身份分叉已有当前版本证据显示消失；完整 L1/L2/L3 和更广的 Reward/ms-swift 覆盖仍未验。
## 2026-10-12 Reward 首批 logits 对拍

运行键 `20261012-qwen2-reward-stock-cli-l1-logits-v2` / Slurm job17601 修复了 v1/job17596 在模型前向前遇到的 harness 路径错误。v1 使用过长的持久 state 路径作为 `TMPDIR`，ms-swift 数据 `map(num_proc=1)` 创建 multiprocessing Unix socket 时失败于 `OSError: AF_UNIX path too long`；该键没有模型计算、没有原生 oracle，也未启动 shim。v2 将 `TMPDIR` 指向短路径 `/tmp/q2rm-l1-logits-$SLURM_JOB_ID`，该路径软链接仍指向持久 state 的 run `tmp/`；没有清理或挪动模型、缓存或实验产物。

job17601 在 `cscg-qh04` NVIDIA RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，driver `580.178.04`）按 native 后 strict shim 顺序，各执行公开 `swift rlhf --rlhf_type rm` 一步。两侧加载相同底模 SHA256 `9cd8fc8c…77cad` 与初始 safetensors SHA256 `415ca5dd…35a1b`，使用相同六行数据 SHA256 `dc072b5e…08eee4`、seed/data_seed 42、FP32/eager、batch 2/max length 32。采集器只将原有 `RewardTrainer.compute_loss` 调用改为 `return_outputs=True` 以记录其已计算的 chosen/rejected scores，随后按原调用约定只返回 loss；模型公式、输入和训练路径不变。双方首批 `input_ids` SHA256 `2a140ca4…361b87` 与 `attention_mask` SHA256 `c8b517ed…b7f6f8` 相同，且逐项匹配旧 native job16186 与 sampler 修复后 strict-shim job17482 的首批记录；输入 shape 均为 `[4,32]`、dtype `int64`、设备 `cuda:0`。

两侧模型 score logits 都是 `[4,1]`、CUDA `float32` 且 finite。Native 值为 `[-1.51895797, -0.02369708, -1.70691741, -0.08534986]`，strict shim 值为 `[-1.51895499, -0.02369499, -1.70691359, -0.08534890]`；`max_abs=3.8146973e-6`，以 `max(|native|,|shim|)` 为分母的 `max_rel=8.8034590e-5`。所有元素满足 `abs(diff) <= 1e-5 + 1e-4*abs(native)`。Strict-shim 两进程均 `use_cuda=1`、fallback delta=0；`jittor_core` SHA 前后完全相同。两端均完成 Trainer 单步且未写 checkpoint；此单次前向不构成 L2 收敛/全梯度验收，也不改变 L0/L2/L3/L4/L5 边界。原始结果、完整值与采集脚本未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-reward-stock-cli-l1-logits-v1/`、`...-v2/`；v1 保留为前向前的 harness 失败证据。
