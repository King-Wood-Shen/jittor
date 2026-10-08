# Qwen2-0.5B 序列分类公开 SFT：CUDA 初态与首批前向

- 状态：固定同一分类 checkpoint 和首批输入的 CUDA 前向对齐；公开单步训练已完成。L0、L1 均为 partial，L2–L5 未通过前置层级或尚未验证。
- 日期：2026-10-08。
- 基线：Jittor `24c36da105effd55826f67b8767756a387cf3ffc`，已同步上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B，公开 `python -m swift.cli.main sft --task_type seq_cls`，单 RTX 4090，FP32/eager，全参数 SGD，固定 8 行 JSONL、batch 4、一步、最大长度 64。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补参数名与状态键映射、全量梯度和 optimizer 状态/更新轨迹、恢复与性能；Torch shim、Jittor CUDA 执行器或 Swift Trainer 变化时重验。

Slurm 13054 在 `cscg-qh04` 的 RTX 4090 上先执行原生公开 CLI，构造 494.0346M 参数的 `Qwen2ForSequenceClassification` 并保存 `checkpoint-1`。Slurm 13067 随后令原生与严格 shim 从该同一 checkpoint 各执行公开单步训练。Python 为 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、Jittor 2.0.0；GPU UUID 为 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。基础模型 `model.safetensors` SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，共享分类 checkpoint SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`，数据 SHA256 为 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`。候选使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；导入/结束均记录 `use_cuda=1`、fallback `0`。

在 SGD 首次 step 前，对 291 个 optimizer 参数（494,034,560 个元素）比较 group/index、shape、dtype、device、trainable 标记及逐项值 SHA256，全部相同；两侧参数都在 `cuda:0`。Trainer capture 的首批 `input_ids`、`attention_mask`、`labels` shape 分别为 `[4,15]`、`[4,15]`、`[4]`，值逐项相同且均在 CUDA。该批分类 logits shape `[4,2]`，最大绝对误差 `1.0729e-5`、相对 L2 `5.9512e-7`；loss 最大绝对误差 `9.5367e-7`、相对 L2 `2.3116e-7`，满足 `5e-3` 容差且两边有限。公开训练记录 loss 为原生 `4.12568569`、shim `4.12568665`；这是同一首批的训练日志，差值约 `9.54e-7`。两边均生成 step-1 checkpoint。

最初直接从相同基础模型、相同 seed 各自新建分类头时，native/shim 唯一不一致参数是 shape `[2,896]` 的分类 score 权重（Slurm 13054/13055），所以后续数值对拍改用共同原生 checkpoint。随后使用 `dataset_shuffle=false` 但未关闭 Swift 训练 DataLoader shuffle 的试验（13057–13059）捕获到两边首批 IDs/labels 不同；两侧 `RandomSampler` 的同 seed 次序也不同（有效探针 13063）。这是本轮测试配置没有设置正确参数造成的数据批次错配，不是相同输入上的前向差异。Slurm 13067 显式设置 `--train_dataloader_shuffle false` 后首批完全一致。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开模型、tokenizer/template、dataset、全参数 tuner、optimizer、trainer 均构造；291 个参数初态一致且 CUDA。optimizer group/index 没映射回参数名/state-dict 键。 |
| L1 | partial | 公开训练首批固定输入的 logits/loss CUDA 对拍通过；仍缺完整输出结构与全批/独立前向清单。 |
| L2 | blocked | 只做一步；未对 291 个参数梯度、optimizer 状态和至少三步更新轨迹对拍。 |
| L3 | blocked | 未测试恢复、RNG 和数据游标。 |
| L4 | blocked | 公开 CLI 单步已执行，但 L0/L1 未完整通过，依层级规则不标通过。 |
| L5 | blocked | 前置层未通过；没有十次预热后稳态协议。 |

Slurm 13055 曾因参数替换脚本错误在 shim 启动前退出；13058/13061 是未命中的 capture hook 与随后的精确输入独立前向诊断；13062 候选误先导入原生 `torch`，结果作废；13066 在 native 阶段因 capture hook 闭包变量错误退出。上述 harness 失败日志均保留，均不计作兼容通过或模型运行失败。Slurm 13067 的结果和原始 JSON/NPZ、脚本、checkpoint、缓存均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-seqcls-l0-v5/`；关联证据位于 `20261008-qwen2-seqcls-l0-v1/`、`20261008-qwen2-seqcls-l0-v2/` 与 `20261008-qwen2-seqcls-l1-lossdiag-v2/`。

该结论只适用于此 Qwen2 序列分类固定 checkpoint 与首批配置，不代表其他分类/embedding/reranker/reward 模型或整个 ms-swift。

文档与结构门禁 Slurm 13068 在 `cscg-qh04` 通过：产物有限性/shape/dtype 检查通过，`generate_manifest.py --check` 和 `tools/check_repo_layout.sh` 通过；Torch 模式 `tests/structure` 汇总 `1386 passed, 6 skipped, 1027 subtests passed`（183.13s）。
