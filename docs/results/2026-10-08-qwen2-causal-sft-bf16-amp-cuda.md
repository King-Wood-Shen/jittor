# Qwen2-0.5B 公开 SFT BF16 autocast CUDA 复验

- 状态：已验证当前 shim 对 FP32 模型请求 BF16 autocast 时会降为 FP16，故该精度合同不兼容；固定三步训练可完成，但完整 L0–L5 未通过。
- 仓库：Jittor `baa73bfdc3e3af949722659405e31ee70e5bb380`（2.0.0）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：实现并测试真实 BF16 autocast 后，补齐同输入前向、逐参数梯度/优化器状态、恢复及性能协议。

Slurm 13516 在 `cscg-qh04` RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，驱动 580.178.04）依次运行原生 PyTorch 与 strict CUDA shim。环境为 Python 3.11.15、CUDA 12.2；模型为 Qwen2-0.5B，模型权重 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，固定四条数据 SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。配置为公开 `python -m swift.cli.main sft`、全参数、FP32 权重、`--bf16 true`、eager attention、固定数据顺序、batch 4、最大长度 64、SGD、学习率 `1e-5`、三步。原生独立 oracle 先完成并保存 checkpoint-3，之后才启动 shim。

Shim 的 AMP 实现打印警告：当操作数全为 FP32 时，`torch.autocast(dtype=torch.bfloat16)` 被近似为 FP16；其源头在 `python/jittor/compat/torch/amp.py::_amp_register_for`。strict shim bootstrap 的父子进程均记录 `use_cuda=1`、Torch shim 标记为真、fallback 计数 0；Jittor 日志确认 CUDA 已启用。两侧均成功完成公开训练入口并写出 checkpoint-3。

| step | 原生 loss | shim loss | 原生 grad norm | shim grad norm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.81937957 | 3.82080078 | 448.09659 | 444.98901 |
| 2 | 3.81178713 | 3.81982422 | 444.11020 | 445.67569 |
| 3 | 3.76624942 | 3.82519531 | 458.02768 | 444.40701 |

Slurm 13521 在 GPU worker 上比较完整 safetensors checkpoint：290 个键、494,032,768 个参数元素一致；最大绝对权重差 `3.72529e-7`，全量相对 L2 `9.80753e-9`。这只表示该小学习率三步后的末态权重接近，不能抵消逐步 loss/梯度范数差异或 BF16 autocast 的语义偏差。比较器随后无法使用原生 `torch.load` 读取 shim 的 `optimizer.pt`：原生文件为 PyTorch zip 容器，shim 文件是协议 4 pickle 数据。因此 optimizer 状态未完成对照，不能据末态权重宣称 L2 通过。

训练日志的 shim 首步耗时 399 秒，后续步因首次 JIT 编译完成而显著缩短；它不能与原生约 4 秒的整段训练时间作性能比较。日志内存读数也不是按同一规范测量的峰值，故不作为 L5 证据。原始命令、日志、缓存、checkpoint 与比较器保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-sft-bf16-amp-r1/`。

| 层 | 状态 | 证据与缺口 |
| --- | --- | --- |
| L0 | partial | 公开 CLI 构造真实模型、tokenizer、数据与 Trainer 并训练；完整初态键、dtype、逐参数设备清单未对拍。 |
| L1 | not-run | 未在独立同权重、同输入前向中比较 logits/hidden/loss 张量。 |
| L2 | partial | 两侧完成三步并比较 loss、grad norm 与末态全量权重；没有逐参数梯度、optimizer 状态对拍，且 FP32 输入区域的 BF16 请求实际被降为 FP16。 |
| L3 | not-run | 未验证新进程恢复及后续轨迹。 |
| L4 | blocked | 实际公开 SFT CLI 已完成并保存 checkpoint；因前层证据未通过，按层级门槛不升级。 |
| L5 | blocked | 前层未通过；首次 JIT 混入 shim 总耗时，未执行预热加至少 10 次稳态测量。 |

本结果仅覆盖此 Qwen2-0.5B FP32 权重、三步 SFT 配置；不代表已加载 BF16 权重、其他 AMP dtype 或整个 ms-swift 兼容。
