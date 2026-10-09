# Qwen2-0.5B 公开 CPO LoRA `swift rlhf` CUDA 对拍

- 状态：原生与 strict CUDA shim 均可完成三步公开 CPO LoRA 训练，但固定配置的训练指标和末态 adapter 明显不同；数值兼容未通过。初始 adapter 和实际 batch 身份尚未独立对照，根因未定。
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

### 静态调用链复核（2026-10-09）

在 ms-swift `88d727951203256baa564c643c651b6f8d90fd7e` 检查 `CPOTrainer` 的继承链后，`RLHFTrainerMixin.get_train_dataloader` 调用 `super()`，最终进入 `SwiftMixin.get_train_dataloader`。该实现将 `args.train_dataloader_shuffle` 传入 `BatchSamplerShard` 的 `shuffle` 参数；本次 CLI 两侧记录的值均为 `false`。因此现有源码没有支持“Swift CPO dataloader 忽略关闭 shuffle”这一解释的证据。此静态检查不能证明两次运行实际读取了相同的 tokenized batch，也不能解释指标分歧；原始运行没有保存 batch 内容与 ID、初始 adapter 仍未比对，根因继续保持未定。下次诊断须先固定相同 adapter，再记录每步 `input_ids`、mask 与 labels，之后才比较 logits/loss、全部梯度及更新轨迹。

| 层 | 状态 | 证据与缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型、tokenizer、偏好数据、CPO Trainer 与 LoRA 构造并完成训练；未对比初始 adapter 全量值、所有参数 device 清单和 batch ID。 |
| L1 | not-run | 未做同初始 adapter、同输入的 chosen/rejected logits 与 CPO loss 张量对拍。 |
| L2 | partial | 两侧三步训练、loss/grad norm 和末态 adapter 已比较且存在明显分歧；缺逐参数梯度、optimizer 状态和初始 adapter 等价证据。 |
| L3 | not-run | 未验证新进程恢复与后续轨迹。 |
| L4 | blocked | 公开 CLI 两侧均完成并保存 checkpoint；前层门槛未通过，按层级规则不升级。 |
| L5 | blocked | 前层未通过，且首次 JIT 混入候选时长；未做预热后至少 10 次稳态测量。 |

这是 Qwen2-0.5B、固定小数据、LoRA CPO 的单配置结果，不代表 CPO 所有实现、其他偏好算法或整个 ms-swift。
