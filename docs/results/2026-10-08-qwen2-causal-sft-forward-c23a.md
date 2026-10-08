# Qwen2-0.5B 全参数 SFT：首批前向张量对拍

- 状态：同一固定训练 batch 的完整 logits、25 组 hidden states 和 loss 已逐张量比较；L0–L5 未全部通过。
- 日期：2026-10-08。
- 基线：Jittor `c23a8a75e6653cbcf8fc4adf4d7ec7b20f78dda3`（upstream `25700b208fe58680169312e214c9ee82f125a094`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 验证范围：公开 `python -m swift.cli.main sft`，Qwen2-0.5B 全参数 FP32/eager、SGD、单卡 RTX 4090、四条固定数据、batch 4、max length 64；显式关闭 train dataloader shuffle，固定 `data_seed=1234` 与 `seed=1234`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补全初始模型逐参数状态清单、在不改模型返回值的捕获路径复验训练反向，再执行三步更新和恢复；shim/CUDA 执行器或 ms-swift Trainer 前向改动时复查。

Slurm 12631 在 cscg-qh04 的 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，驱动 580.178.04）顺序运行独立原生 PyTorch 与严格 Jittor shim。两侧使用同一缓存模型 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`、数据 SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af` 及固定 CLI 参数。首批 `input_ids`、`attention_mask`、`labels` 均逐项相同，形状 `[4,31]`。捕获的参数为 494.0328M、全部可训练；输入和所有 27 个输出（loss、logits、25 组 hidden states）元数据均为 `cuda:0`，FP32 浮点结果有限。

Slurm 12632 在 GPU worker 比较：完整 logits shape `[4,31,151936]`，最大绝对差 `1.12038e-4`、相对 L2 `3.17019e-6`，每个位置的 argmax 一致；25 组 hidden shape 均为 `[4,31,896]`，总体最大绝对差 `4.65393e-4`、相对 L2 最大值 `3.91260e-6`；loss 最大绝对差 `9.53674e-7`。候选 `fallback_count=0`，原生进程确认未安装 torch shim。此数值结论限于该 batch 和模型配置。

前向捕获钩子为记录 hidden states，在候选 forward 中请求 `output_hidden_states=True`。前向、保存结果和数值对拍均完成；随后 Trainer 调用 backward 时，Slurm 12631 以 `nano_vector.h:41: slice overflow` 退出，堆栈落在 `torch.Tensor.backward -> jittor.compat.torch` 的 `core.grad_optional`。这发生在本次观测钩子改变模型输出选项的诊断运行中，不据此推断未加钩子的普通训练失败。此前不带此钩子的公开三步训练完成记录见 `2026-10-08-qwen2-causal-sft-grad3-4a7e.md`。

早期捕获试验亦保留：12626 的钩子只匹配 CLI 父进程，原生单步实际执行但未保存张量，作业壳层因缺产物退出；try2 的 12627 保存了张量，但两个独立进程的训练 sampler 顺序不同，12629 比较因此失败。12630 在 worker 确认两侧样本顺序和 padding mask 对应不同记录。try3 显式设置 `train_dataloader_shuffle=false` 后输入逐项相同。三次运行和比较原始产物均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-fullparam-l1capture-c23a-try3/`、`...-c23a-try2/` 与 `...-c23a/`。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 SFT CLI 构造真实模型、Trainer 与固定数据；没有逐项核对初始 290 个状态键、参数 dtype/device 和数值。 |
| L1 | partial | 前向数值对拍达到容差、输入逐项相同、输出形状/dtype/有限性/设备一致；按层级规则，L0 尚非完整 PASS。 |
| L2 | blocked | 此捕获配置的 backward 在诊断钩子开启 hidden states 后触发内部断言；未取得本配置完整逐参数梯度、三步 optimizer 状态与更新轨迹。 |
| L3 | blocked | 未做同进程或新进程恢复、RNG 与数据游标检查。 |
| L4 | blocked | 公共命令已进入前向，但候选因上述诊断运行 backward 退出码 1；本配置未完成公开入口端到端。 |
| L5 | blocked | 前级未通过，未运行预热后十次同步稳态协议。 |
