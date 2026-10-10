# Qwen2-0.5B 序列分类公开 SFT：固定窗口收敛 CUDA 复验

- 状态：固定单卡 FP32/eager/SGD profile 的 L0、L1、L2、L4 通过；L3 未运行，L5 因前置层级未完成而未运行。本报告是更新后训练收敛协议下的新运行，保留旧版严格逐值报告及其原结论。
- 日期：2026-10-11。
- 基线：Jittor `3f1b2618d6c509538c927b4fa563dc8e3bae7311`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：公开 `python -m swift.cli.main sft --task_type seq_cls`，Qwen2-0.5B 全参数分类，单张 RTX 4090，FP32/eager，固定 8 行数据，batch 4、8 个 optimizer steps、SGD、固定前两步/末两步 loss 窗口。
- 维护者：ms-swift CUDA 适配。
- 复查条件：增加完整 checkpoint 恢复协议后再复核 L3；Jittor CUDA、Torch shim、Transformers 或 Swift Trainer 改动时重验。

原生 PyTorch oracle 在 Slurm job 16209 执行，strict shim 在 job 16210 执行；同一轮 GPU worker 只读比较 job 16211。三个作业均使用 NVIDIA RTX 4090、CUDA `cuda:0`。基础分类 checkpoint 的 SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`，固定数据 SHA256 为 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`。具体参数为 `--torch_dtype float32 --attn_impl eager --optim sgd --learning_rate 1e-5 --lr_scheduler_type constant --max_steps 8 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --dataset_shuffle false --train_dataloader_shuffle false --seed 1234 --data_seed 1234`。原生与候选训练都通过公开 SFT CLI 完成并保存 checkpoint；候选从导入到结果保存启用严格 CUDA scope 和 fallback 禁止，所有 shim 事件 `use_cuda=1`、`fallback_count=0`。

## 结果

两侧初始 state 均有 292 项（291 个参数和 1 个 buffer），键、shape、dtype、device 完全相同，所有项均为 CUDA FP32；完整初态逐项 SHA256 一致。291 个可训练参数的 `requires_grad` 标志一致。唯一的标志差异是 buffer `model.rotary_emb.inv_freq`：原生 `requires_grad=false`、shim `true`。它不改变本轮 buffer 值、训练参数集合或观测到的更新；仍明确记录该差异，不据此推断其他 buffer 或恢复语义。

8 个 optimizer step 的 `input_ids`、attention mask、labels 及逐样本身份哈希在两侧逐步相同。首批 `[4,2]` 分类 logits 最大绝对差为 `1.0729e-5`、相对 L2 为 `5.9512e-7`，两侧输出和 loss 均有限。所有 291 个可训练参数在每一步都产生有限、非零 CUDA FP32 梯度；两侧梯度总 L2 约为 1，跨运行逐参数梯度/更新差留作诊断，不作为新协议的数值门槛。

原生 loss 为 `4.12568569, 3.76338768, 4.11957598, 3.75764108, 4.11343718, 3.75190020, 4.10730362, 3.74615192`；shim loss 为 `4.12568665, 3.76338339, 4.11957645, 3.75764346, 4.11344528, 3.75189900, 4.10730457, 3.74615026`。按运行前固定的两步窗口，原生均值从 `3.944536685` 降至 `3.926727770`，shim 从 `3.944535020` 降至 `3.926727415`，分别满足收敛条件。

两侧均有 291 个 optimizer 参数、494,034,560 个元素发生有限且非空更新，各有 248 个张量改变；各自相对初态的更新 L2 均约 `0.00320158`，最大权重变化 `1.71065e-5`。原生和 shim 的末态跨运行差最大绝对值 `1.49012e-8`、相对 L2 `1.02106e-10`。两侧 SGD 状态项均为空（无 momentum 的预期状态），参数数目与两个参数组的学习率、momentum、weight decay、dampening 和 nesterov 相同；原生参数组另带 `initial_lr=1e-5`，shim 未发布该字段，作为 scheduler/恢复复核项保留，不掩盖差别。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | PASS | 两侧独立构造真实模型、tokenizer、数据、Trainer 与 SGD；292 项 state 的键/shape/dtype/device 和初态哈希相同。记录 `inv_freq.requires_grad` 与 `initial_lr` 字段差异。 |
| L1 | PASS | 8 步同批 CUDA 前向与有限 loss；首批分类 logits 形状、dtype、有限性及数值诊断已记录。 |
| L2 | PASS | 291 个可训练参数每步梯度存在且 finite/nonzero；两侧实际更新；固定两步初末 loss 窗口均下降。逐值差仅作诊断。 |
| L3 | not-run | 未检查同进程/新进程恢复、optimizer/scheduler/RNG/dataloader 游标及恢复后续训。 |
| L4 | PASS | 原生与 strict shim 均端到端运行公开 `swift sft` 并保存 checkpoint。 |
| L5 | blocked | L3 未完成，未做预热后至少十次同步稳态、吞吐/原生比和显存协议。 |

## 文档门禁

文档门禁运行键 `20261011-qwen2-seqcls-convergence-report-docgate-v2`、Slurm job 16213 已完成（exit 0）。GPU worker 上 `bash tools/check_repo_layout.sh` 通过，文档治理检查覆盖 237 个活动 Markdown；Torch 模式 `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed。此前 v1/job 16212 只因 manifest 顺序断言失败，已修正排序并以独立 v2 键重跑完整门禁；v1 结果保留，不属于模型兼容性失败。

## 审计边界

早期尝试中，job 16207/16208 的 harness 在原生入口或审计钩子阶段提前退出；job 16209 完成原生 oracle 后，曾因 JSON 审计器尝试序列化内部 JT Var 退出。job 16210 完成 strict shim 的 8 步训练后，旧比较器因把 buffer `requires_grad` 也当成跨实现硬条件而退出。以上运行目录均保留；没有把 harness 失败算作兼容失败，也没有重跑旧运行键。只读 GPU worker job 16211 按新协议完成比较并确认最终状态。原始脚本、checkpoint、逐步梯度/输入、日志和完整 JSON 留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-seqcls-convergence-{v1,v2,v3,v4,analysis-v1}/`。

本结论只适用于所列 Qwen2-0.5B 固定分类样本、全参数 FP32/eager、SGD 和单卡公开入口，不推广到其他分类数据、优化器、AMP、分布式训练、恢复或整个 ms-swift。旧报告 [三步严格对拍](2026-10-08-qwen2-seqcls-sft-cuda.md) 仍绑定原协议和原运行键，不由本报告改写。
