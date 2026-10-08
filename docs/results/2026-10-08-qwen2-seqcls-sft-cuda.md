# Qwen2-0.5B 序列分类公开 SFT：CUDA 初态与首批前向

- 状态：本报告覆盖的 Qwen2-0.5B 序列分类全参数公开 SFT 配置，L0、L1、L2、L4 为 partial，L3、L5 未通过前置层级或尚未验证。三步逐参数梯度与更新已对拍；仍缺逐步样本身份/输入清单，且 scheduler checkpoint 结构不同。
- 日期：2026-10-08。
- 基线：三步扩展使用 Jittor `21cfdb570601325c256ca491717219528142a04f`，已整合 `origin/2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。原单步 Slurm 13067 使用 Jittor `24c36da105effd55826f67b8767756a387cf3ffc`，结果仍绑定其原 SHA。
- 范围：Qwen2-0.5B，公开 `python -m swift.cli.main sft --task_type seq_cls`，单 RTX 4090，FP32/eager，全参数 SGD，固定 8 行 JSONL、batch 4、三步扩展、最大长度 64。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补逐步样本身份与参数名映射，处理 scheduler 状态差异并完成新进程恢复及性能；Torch shim、Jittor CUDA 执行器或 Swift Trainer 变化时重验。

Slurm 13054 在 `cscg-qh04` 的 RTX 4090 上先执行原生公开 CLI，构造 494.0346M 参数的 `Qwen2ForSequenceClassification` 并保存 `checkpoint-1`。Slurm 13067 随后令原生与严格 shim 从该同一 checkpoint 各执行公开单步训练。Python 为 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、Jittor 2.0.0；GPU UUID 为 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。基础模型 `model.safetensors` SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，共享分类 checkpoint SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`，数据 SHA256 为 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`。候选使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；导入/结束均记录 `use_cuda=1`、fallback `0`。

在 SGD 首次 step 前，对 291 个 optimizer 参数（494,034,560 个元素）比较 group/index、shape、dtype、device、trainable 标记及逐项值 SHA256，全部相同；两侧参数都在 `cuda:0`。Trainer capture 的首批 `input_ids`、`attention_mask`、`labels` shape 分别为 `[4,15]`、`[4,15]`、`[4]`，值逐项相同且均在 CUDA。该批分类 logits shape `[4,2]`，最大绝对误差 `1.0729e-5`、相对 L2 `5.9512e-7`；loss 最大绝对误差 `9.5367e-7`、相对 L2 `2.3116e-7`，满足 `5e-3` 容差且两边有限。公开训练记录 loss 为原生 `4.12568569`、shim `4.12568665`；这是同一首批的训练日志，差值约 `9.54e-7`。两边均生成 step-1 checkpoint。

最初直接从相同基础模型、相同 seed 各自新建分类头时，native/shim 唯一不一致参数是 shape `[2,896]` 的分类 score 权重（Slurm 13054/13055），所以后续数值对拍改用共同原生 checkpoint。随后使用 `dataset_shuffle=false` 但未关闭 Swift 训练 DataLoader shuffle 的试验（13057–13059）捕获到两边首批 IDs/labels 不同；两侧 `RandomSampler` 的同 seed 次序也不同（有效探针 13063）。这是本轮测试配置没有设置正确参数造成的数据批次错配，不是相同输入上的前向差异。Slurm 13067 显式设置 `--train_dataloader_shuffle false` 后首批完全一致。

## 三步梯度与更新扩展

Slurm 13101 先在 `cscg-qh17` RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）以原生 PyTorch 完成公开 CLI 三步训练，并在每次 SGD step 前保存全部 291 个 CUDA 参数梯度。作业随后被 harness 的目录 glob 断言提前终止；三步原生证据完整保留。Slurm 13104 只补跑 strict shim：同一 checkpoint、数据、FP32/eager 配置、seed、`--train_dataloader_shuffle false`，三步梯度均捕获，导入至结束 `use_cuda=1`、fallback `0`。Slurm 13105 在 GPU worker 上完成两侧逐项比较。

三步梯度最大绝对差分别为 `7.49e-7`、`6.46e-7`、`1.49e-6`，相对 L2 分别为 `4.36e-6`、`3.91e-6`、`9.43e-6`；三步参数 loss 最大差分别为 `9.54e-7`、`3.58e-6`、`5.24e-6`。末步 checkpoint 的 291 个键/shape/dtype 一致，最大权重绝对差 `7.45e-9`、相对 L2 `6.06e-11`，171 个张量发生更新。两侧 SGD `optimizer.pt` 经 zip/pickle 双格式读取后，state 均为空、两个参数组逐项相同；这是无 momentum SGD 的预期状态。严格 shim 的每步梯度 device 均为 CUDA，fallback 始终为 0。

该扩展没有保存每步的 token/label 张量身份，所以三步相同数据顺序仅由固定数据文件、关闭训练 DataLoader shuffle、相同 seed/data seed 与一致 loss 记录支持，未作逐批逐值证明。另发现 `scheduler.pt` 持久化结构不一致：native 有 `verbose=False`，shim 有 `_is_initial=False`；本轮未继续恢复测试。Slurm 13103 的候选缺少审计环境变量；13104 完成 strict shim 三步训练和梯度采集后，因比较器变量名覆盖 NumPy 而以退出码 1 结束；13108/13109 的状态比较 harness 分别遇到错误加载器假设和 scheduler 键差异。相关日志和产物均保留，只有 13104 已捕获的训练证据及 13105/13110 有效比较计入结论。原始产物未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-seqcls-l2-grad3-v1/`。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开模型、tokenizer/template、dataset、全参数 tuner、optimizer、trainer 均构造；291 个参数初态一致且 CUDA。optimizer group/index 没映射回参数名/state-dict 键。 |
| L1 | partial | 公开训练首批固定输入的 logits/loss CUDA 对拍通过；仍缺完整输出结构与全批/独立前向清单。 |
| L2 | partial | 三步逐步比较全部 291 个 CUDA 梯度、loss、更新后权重及 SGD optimizer 状态通过；缺逐步样本身份/输入逐值清单，L0/L1 仍 partial。 |
| L3 | blocked | 未做同进程或新进程恢复；scheduler checkpoint 键不同（native `verbose`，shim `_is_initial`），optimizer/scheduler/RNG/游标续跑轨迹未验证。 |
| L4 | partial | 原生与 strict shim 均经公开 `swift sft` CLI 完成单步和三步训练并产出 checkpoint；前置 L0/L1/L2 未完整通过。 |
| L5 | blocked | 前置层未通过；没有十次预热后稳态协议。 |

Slurm 13055 曾因参数替换脚本错误在 shim 启动前退出；13058/13061 是未命中的 capture hook 与随后的精确输入独立前向诊断；13062 候选误先导入原生 `torch`，结果作废；13066 在 native 阶段因 capture hook 闭包变量错误退出。上述 harness 失败日志均保留，均不计作兼容通过或模型运行失败。Slurm 13067 的结果和原始 JSON/NPZ、脚本、checkpoint、缓存均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-seqcls-l0-v5/`；关联证据位于 `20261008-qwen2-seqcls-l0-v1/`、`20261008-qwen2-seqcls-l0-v2/` 与 `20261008-qwen2-seqcls-l1-lossdiag-v2/`。

该结论只适用于此 Qwen2 序列分类固定 checkpoint 与首批配置，不代表其他分类/embedding/reranker/reward 模型或整个 ms-swift。

文档与结构门禁 Slurm 13068 在 `cscg-qh04` 通过：产物有限性/shape/dtype 检查通过，`generate_manifest.py --check` 和 `tools/check_repo_layout.sh` 通过；Torch 模式 `tests/structure` 汇总 `1386 passed, 6 skipped, 1027 subtests passed`（183.13s）。

本次报告更新门禁 Slurm 13111 在 `cscg-qh17` 通过：`generate_manifest.py --check`、`tools/check_repo_layout.sh` 均成功；Torch 模式 `tests/structure` 为 `1386 passed, 6 skipped, 1027 subtests passed`（179.03s）。
