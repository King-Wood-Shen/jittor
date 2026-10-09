# Qwen2-0.5B 公开全参数 SFT 双卡三步逐参数梯度补证

- 状态：固定配置下原生 PyTorch 与严格 CUDA shim 完成三步公开训练；逐参数梯度、rank 间同步、初态参数哈希和末态权重已留证。两端输入采集点及 `use_logits_to_keep` 配置不一致，L0–L2 仍为 partial，不代表双卡训练面完整通过。
- 日期：2026-10-08。
- 基线：Jittor `6c104502e3474577705c49018bf57a3d12dabbb2`；同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B、公开 `python -m swift.cli.main sft`、双 RTX 4090 NCCL、全参数 FP32、eager attention、无动量 SGD、每卡 batch 2、三步、固定四条数据、最大长度 64、`seed=data_seed=1234`、学习率 `1e-5`、梯度裁剪范数 1.0。
- 维护者：ms-swift CUDA 适配。
- 复查条件：在两端同一 Trainer 输入边界记录逐 rank、逐步输入和样本身份，并固定或对齐 `use_logits_to_keep`；公开训练、torch shim、autograd、SGD 或 NCCL 语义变更时复查。

独立原生 oracle 为 Slurm 13142，strict shim 公开 CLI 为 13151，GPU worker 上的只读比较为 13153；均使用 `cscg-qh04` 两张 RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`、`GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）。模型文件 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，训练数据 SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。两侧公开 CLI 参数相同；rank 0/1 分别通过 Torchrun 与 Jittor NCCL 对应，候选两 rank 起止均记录 `use_cuda=1`、`fallback_count=0`。原生 rank 明确记录未安装 shim。

训练在每次 `SGD.step` 前、裁剪后采集 290 个参数的梯度。两 rank 各步清单均为 290 项；同一运行的 rank 0/1 每项梯度 SHA256 相同。跨运行时逐张量比较覆盖每步 494,032,768 个 FP32 梯度元素：

| 步 | 最大绝对差 | 相对 L2 |
| --- | ---: | ---: |
| 1 | `2.98023e-7` | `4.92789e-6` |
| 2 | `2.19792e-7` | `4.59096e-6` |
| 3 | `4.17233e-7` | `5.40135e-6` |

原生 / shim 三步 loss 分别为 `3.82736588 / 3.82736969`、`3.82289505 / 3.82289600`、`3.81842184 / 3.81841969`；三步 token accuracy 均为 `0.4375`。step-3 checkpoint 的 290 个权重键集合、shape、dtype 和有限性一致，最大绝对差 `7.45058e-9`、相对 L2 `6.54474e-11`。

在本报告最初版本中，候选逐 rank 批次和初态清单未留档。2026-10-09 的补采已补上候选两 rank 初态参数清单，但候选输入在 `Seq2SeqTrainer.compute_loss` 入口采集，原生输入在模型 `forward` 入口采集，两个观测点不同，不能据此证明批次身份。补采还发现 native/shim 自动选择了不同 `use_logits_to_keep`，且两侧均启用 `train_dataloader_shuffle`。故 L0–L2 仍不能升级为 PASS：L1 缺同权重、同批次、同 forward 配置的 logits/hidden 对拍；L2 的梯度和末态虽有诊断数字，仍缺同输入对拍及独立逐步 optimizer/update 状态核验。离散 token 输入不适用连续输入梯度。

| 层 | 状态 | 本配置缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型、tokenizer、trainer、optimizer 与两个 rank 构造；补采已证明两 rank 的 290 个候选参数名、shape、dtype、requires-grad、初态哈希与 CUDA device 对齐；tokenizer/template 的完整状态映射仍缺。 |
| L1 | partial | 原有 loss 和 token accuracy 可比；补采发现两端 `use_logits_to_keep` 不同，且输入观测点不同，缺同批次、同配置下的 logits/hidden 对拍。 |
| L2 | partial | 全参数三步梯度与末态 checkpoint 有诊断比较；补采的两端 sampler 顺序/输入身份未证明相同，缺同输入梯度对拍及逐步 optimizer 状态和直接更新轨迹。 |
| L3 | not-run | 本运行未做连续与恢复轨迹对照；旧运行的恢复证据保持其原运行键及基线边界。 |
| L4 | partial | 真实公开双卡 `swift sft` CLI 完成三步并保存 checkpoint；前序层未完整通过。 |
| L5 | blocked | 前置层未完整通过，未运行十次同步稳态性能协议。 |

故障与采集记录保留于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-ddp-grad3-{v5,v2,v3,v4,v6,v7}/`。13142 的原生 oracle 成功；其后 13142 shim 预热缺 Python 路径，13144 缺 `MASTER_ADDR`，13145 采集模块未进入搜索路径，13146 在文件准备阶段退出，13147 在模型阶段前取消，均不作为兼容结果。13150 在 forward 观测钩子开启时首步反向触发 `nano_vector.h:41: slice overflow`；其后去掉 forward 钩子的 13151 三步 strict shim 训练成功，比较器最初因错误要求 safetensors 键顺序相同失败，13153 改为比较键集合后通过。前述故障均保留日志，未据此修改兼容实现。

比较摘要为 `v7/comparison.json`，原生和候选梯度 manifest、张量、rank 日志及 checkpoint 均保留在对应运行目录。本报告只补强上述单个 Qwen2 配置的梯度证据；双卡 SFT 的完整 L0–L5、其他模型/训练器和 ms-swift 总体兼容仍未验收。

文档验收：Slurm 13156 的全量 `tests/structure` 结果为 `1385 passed, 6 skipped, 1027 subtests passed`，唯一失败是本报告尚未加入根 `MANIFEST.in`。补入生成器对应路径后，Slurm 13159 的布局检查、`tools/build/generate_manifest.py --check --project .` 与定向打包结构测试均通过（`1 passed`）。只复验了受影响的清单测试，未因文档清单修正重复运行完整结构套件。

## 2026-10-09 补充：候选初态与训练输入审计

候选补采运行键 `20261009-qwen2-causal-ddp-input-init-audit-v1`（Slurm 13794）在同一公开双卡 CLI 配置下完成 strict CUDA 三步训练并保存 `checkpoint-3`。候选两 rank 各自 290 个参数的名称、shape、dtype、requires-grad 与初态值哈希逐项匹配此前原生 oracle（运行键 `20261008-qwen2-causal-ddp-grad3-v5`）；全部参数在 CUDA。两 rank 每步梯度哈希一致，训练进程开始/结束均为 `use_cuda=1`、fallback `0`。GPU worker 只读审计为 Slurm 13795，结果位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-causal-ddp-input-init-audit-v1/forensics.json`。

这一补采**没有建立逐步输入同一性**：旧原生探针在 `Qwen2ForCausalLM.forward` 入口记录张量；候选补采在 `Seq2SeqTrainer.compute_loss` 入口、调用 `template.compute_sft_loss` 之前记录，因此两份 NPZ 处于不同观测点，不据其 shape/值差断言批次差异。训练日志另确认两侧都实际使用 `train_dataloader_shuffle=True`，且 ms-swift 的 `use_logits_to_keep` 自动判定为 native `False`、shim `True`。后者会改变 labels 的截断路径。相同 seed 的跨 runtime `RandomSampler` 排列不同已有独立探针证据（见 `2026-10-08-torch-randomsampler-generator.md`），但这不能单独证明本 CLI 的逐步样本顺序。初态匹配、参数更新差异很小和 rank 梯度同步均不能代替同一采样顺序、同一 model-forward 输入及相同 logits-to-keep 配置。

候选梯度与原生的诊断比较覆盖每步 290 个参数、494,032,768 个元素，最大绝对差依次为 `2.75671e-7`、`3.48315e-7`、`3.12924e-7`，相对 L2 为 `5.09947e-6`、`4.40784e-6`、`5.46803e-6`；末态 290 个权重键最大绝对差 `7.45058e-9`、相对 L2 `6.57411e-11`。由于采样身份和 logits-to-keep 配置未对齐，这些只作诊断，不升级 L1/L2。原生运行基线为 `6c104502e3474577705c49018bf57a3d12dabbb2`，候选为 `054163206571918d9969b8077d017731aefb0183`；两者之间 `python/jittor` 无差异。比较器因把不同观测点的输入直接要求逐值相等而退出，训练本身成功；没有重跑旧运行键。后续若继续该面，应在两端同一 Trainer 输入边界记录样本身份及完整 tensors，并明确固定或比较 `use_logits_to_keep` 后再做数值结论。

补充报告门禁 Slurm 13796：仓库布局、文档治理与 Manifest 生成及 `--check` 均通过；Torch 模式 `tests/structure` 为 `1386 passed, 6 skipped, 1027 subtests passed`。6 个 skip 中 2 个为声明类，4 个为本 worker 缺少已部署 Jittor 包和 pytest-xdist；日志保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-causal-ddp-input-audit-docgate-v1/slurm-13796.log`。
