# Qwen2-0.5B stock `swift rlhf --rlhf_type simpo` CUDA 对拍

- 状态：固定单条 preference 数据、全参数 FP32、三步公开 SimPO CLI 下，native 与 strict shim 输入、首步前向、全部逐参数梯度和末态权重达到数值门槛；父子 shim 进程均为真实 CUDA 且 fallback=0。该受控场景 L0/L1/L2/L4 partial，L3 not-run，L5 blocked；不代表多样本 sampler、其他 CPO loss 或 ms-swift 整体兼容。
- 日期：2026-10-09。
- 基线：Jittor `6d20774d9f6e3d2fcc269905fcefbe5c62b09c7e`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM、公开单卡 `swift rlhf --rlhf_type simpo`，全参数 FP32/eager，beta 2.0、simpo_gamma 1.0、cpo_alpha 0、SGD、单条 preference 数据、batch 1、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：扩展到多条固定 preference 数据并显式核验样本顺序；补 optimizer 状态、恢复轨迹后再升级 L2/L3；ms-swift SimPO/CPO trainer、Torch sampler 或 shim 数值实现变化时复查。

Qwen2 权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；单行 preference 数据 SHA256 为 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0。Slurm 13939 预检和 13945 训练均在 `cscg-qh04` RTX 4090（GPU UUID `GPU-4b797e8c-3982-9b01-dc17-43d77960643c`）完成。ms-swift 的 SimPO 参数处理把公开 `rlhf_type=simpo` 转为 `CPOTrainer(loss_type='simpo')`；本次明确关闭 CPO 附加 NLL 项（`cpo_alpha=0`）。

运行键 `20261009-qwen2-simpo-fixedbatch-cli-v1` 中原生 oracle 先运行，strict shim 后运行。两侧均通过 stock CLI 完成三步并保存 `checkpoint-3`。三步 `input_ids`、`labels` 和 `attention_mask` 逐元素相同；全部 290 项初始参数名称、shape、dtype、trainable 标志和值 SHA256 相同且位于 CUDA。strict shim 父子进程开始和结束均记录 `use_cuda=1`、`fallback=0`。

首步 chosen/rejected logits shape 为 `[1,29,151936]`，最大绝对误差 `8.16137e-5`，相对 L2 分别 `2.57146e-6` 与 `2.53968e-6`；其他聚合前向输出最大绝对误差不超过 `6.20e-6`。三步 loss 为 native `1.84327781, 1.83792770, 1.83255649`，shim `1.84328341, 1.83793056, 1.83257294`，最大差 `1.64509e-5`。

每一步均比较全部 290 项梯度、共 494,032,768 个 FP32 元素。最大绝对差依次为 `3.98606e-7`、`5.79283e-7`、`7.45466e-7`；相对 L2 为 `6.11241e-6`、`8.03060e-6`、`1.34174e-5`。step-3 checkpoint 的 290 个权重键对齐，170 个张量更新，末态最大绝对差 `7.45058e-9`、相对 L2 `7.21459e-11`。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 真实模型、tokenizer、SimPO trainer/optimizer 经公开 CLI 构造；290 项初态与 CUDA 参数设备核验一致。完整 state/buffer 和 tokenizer/template 映射未审计。 |
| L1 | partial | 三步输入完全相同；首步 chosen/rejected 完整 logits 与聚合输出达容差。hidden state、后续步逐层前向未采集。 |
| L2 | partial | 三步全部 trainable 梯度和末态模型权重达容差；optimizer state 与逐步直接更新未单独比较。 |
| L3 | not-run | 未测试同进程/新进程恢复、optimizer/scheduler/RNG/dataloader 游标和续训轨迹。 |
| L4 | partial | native 与 strict shim 均经 stock `swift rlhf --rlhf_type simpo` 完成三步并保存 checkpoint；仅覆盖一条固定样本。 |
| L5 | blocked | 前置层未完整通过；shim 训练约 568.6 s 含首次 JIT，未执行预热后至少 10 次稳态性能与显存协议。 |

原始输入、梯度、比较 JSON、checkpoint 和日志未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-simpo-fixedbatch-cli-v1/`。本报告只覆盖上述固定单样本配置。

文档门禁 Slurm 13953 在 RTX 4090 worker 上通过：仓库布局检查成功，Torch 模式结构测试为 1386 passed、6 skipped、1027 subtests passed。6 个 skip 中 2 个因该测试解释器未安装 Jittor、2 个因未安装 pytest-xdist，其余由门禁分类记录。
