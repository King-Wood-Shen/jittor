# Qwen2-0.5B IA3 adapter：公开 infer CLI 新基线复验

- 状态：同一 IA3 adapter 权重下，两条公开 `swift infer` 输出逐字段一致；本配置 L0/L1 partial，完整等级未验收。
- 日期：2026-10-07。
- 基线：Jittor `1abb3629cbf228dc9a9be1ed9f95d091832cb6e4`（上游 `ffeb7bd80447ca78735e7cb96628ab88c1e9360b`；本运行期间源码与 `24483d2327f5ef00ea2d30de953ed81ea2bea5ef` 相同），ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`，Python 3.11.15。
- 验证范围：Qwen2-0.5B、公开 `python -m swift.cli.main infer`、PEFT IA3 adapter、FP32/eager、batch 2、两条短 prompt、greedy 每条最多生成 2 token、单 RTX 4090。
- 维护者：ms-swift CUDA 适配。
- 复查条件：改动 PEFT/IA3 加载、Qwen2 前向、Torch shim 或 ms-swift infer 入口；补齐逐参数设备与全 logits 捕获时。

Slurm 11352 原生阶段和 11355 shim 阶段在同一节点 qh09、同一 RTX 4090（UUID `GPU-84923f1e-ce04-5893-0e80-a1bf47897553`）顺序运行。两侧读取同一模型目录和**同一份** adapter 文件 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261004-qwen2-ia3-export/native-adapter`；Qwen2 基础模型 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；adapter 权重 SHA256 为 `e7de90795307c4ff2d8d928d5fa43c11bd6ff07174c174d9bb549ff9b85e4348`，adapter config SHA256 为 `df9caeb0cf0cbe62682a14e90a260197a7ca1d92e3758867ef29113ec3d8c340`。比较作业 11357 在 GPU worker 上逐字段比较 JSONL，两侧均返回相同的两条非空 response，合计生成 4 token。

候选入口通过 `sitecustomize` 在 CLI 主进程和推理子进程启动/退出时记录 `jittor=2.0.0`、shim 标记真、`use_cuda=1`、fallback=0。Jittor 日志明确报告 `CUDA enabled`，ms-swift 参数将模型映射到 `cuda:0`，并加载 `IA3Model`。本测试没有逐项导出全部模型/adapter 参数、输入 ID 和 logits 的 device/dtype/shape 元数据；因此全参数设备驻留与张量级数值误差仍未证明。旧运行 11313 使用了不同 SHA 的 native/shim adapter，虽输出相同，不计入同权重 oracle 证据。11352 的 shell 在原生 CLI 成功后因 `SRC` 未定义退出；11355 复用该原生结果，仅执行候选阶段。原始脚本、日志、JSONL 和审计保存在未版本化的 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-ia3-infer-same-adapter-ffeb/`，不同 adapter 的旧运行及审计日志保存在同级 `20261007-qwen2-ia3-public-infer-ffeb/`。

| 层 | 本配置状态 | 缺口 |
| --- | --- | --- |
| L0 | partial | 真实 Qwen2、tokenizer、PEFT IA3 模型及公开入口完成构造；尚无所有参数/缓冲的键、dtype、device 清单。 |
| L1 | partial | 同一 adapter、输入和 CUDA runtime 下公开生成结果逐字段相同；未保存同一 forward 的完整 logits/hidden states 与设备元数据。 |
| L2 | not-applicable | 本项是推理配置，不做反向或更新。 |
| L3 | not-run | 未验证 adapter/model 状态恢复及恢复后生成。 |
| L4 | blocked | CLI 端到端执行成功，但 L0/L1 完整验收与 L3 尚缺。 |
| L5 | blocked | 前级未通过；冷启动/JIT 与本次单次生成耗时不作为稳态性能数据。 |

该结果只适用于此 Qwen2-0.5B IA3 adapter 固定配置。PEFT LoRA、QLoRA、LoHa、LoKr、其他 tuner、训练/导出、长上下文、流式及性能均未由本项升级；先前 Jittor 1.3.11 的 adapter 运行也不继承到 Jittor 2.0.0 新基线。
