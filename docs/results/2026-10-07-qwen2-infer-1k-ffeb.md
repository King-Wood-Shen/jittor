# Qwen2-0.5B 公开 infer CLI：新基线千 token 非流式复验

- 状态：限定配置的公开 infer CLI 输出逐字段一致，首次 forward 最后位置 logits 对齐；按逐级门槛 L4/L5 blocked。
- 日期：2026-10-07。
- 基线：Jittor `af2cbfa3c`（父提交包含上游 `ffeb7bd80`），ms-swift `88d7279`，隔离 Python 3.11.15。
- 验证范围：真实缓存 Qwen2-0.5B、公开 `python -m swift.cli.main infer`、单 RTX 4090 CUDA、FP32、eager attention、非流式 greedy、最多 32 新 token、两条提示。
- 维护者：ms-swift CUDA 适配。
- 复查条件：改动 Qwen2 加载、tokenizer/template、Torch shim、CUDA executor 或公开推理入口；补完整 logits/hidden、构造状态和恢复审计时。

Slurm 作业 10968 在同一 GPU 上依次跑原生 PyTorch 与启用严格 CUDA/禁止 fallback 的 Jittor shim，之后用独立审计脚本复核产物；作业 10991 在 NVIDIA worker 另证原生环境为 PyTorch 2.6.0+cu124、CUDA 可用且无 shim 标记。两条真实 tokenizer 提示长分别为 1377、1286 token；公开入口统计合计 2701 prompt tokens，实际生成 49 tokens。两侧保存的完整 JSONL 两行逐字段相等，响应非空且回填到 messages。两侧日志都显示模型映射到 `cuda:0`；候选父子两个进程的启动与正常退出标记均为 `use_cuda=1`、shim 标记真、`fallback=0`。这证明该固定输入和配置的公开入口完成，但不替代前级逐张量验收。

| 层 | 状态 | 现有证据与缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型与公开入口构造并驻留 CUDA；初始状态键、逐项 dtype/device 未直接审计。 |
| L1 | partial | greedy 输出逐字段一致；完整序列 logits 已直接对拍，数值相符；hidden 未直接比较。 |
| L2 | not-applicable | 此配置仅推理，不涉及反向与优化器更新。 |
| L3 | not-run | 未审计同进程及新进程的推理状态恢复。 |
| L4 | blocked | 两侧公开 CLI 已完成；L0/L1/L3 尚未全部验收。 |
| L5 | blocked | 前级未通过；没有合格的同步稳态性能对比。 |

本作业包括模型加载和首次 JIT，候选日志中的约 615 秒不构成 L5 稳态性能数据。更长上下文、stream、批量变化、训练与恢复均不由这项结果升级；约 1.3k-token 的 Python 流式完整性能协议仍受 `KI-EXEC-011` 阻断。原始命令、节点/GPU UUID、日志、两份 JSONL 及审计结果位于未版本化的 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-infer-long-context-ffeb/`。

## 公开入口首次 forward 的张量诊断

同一代码基线（新增运行时 Jittor 报告提交 `ccf25cb6f`，无源码变化）的 Slurm 11221，在同一张 RTX 4090（UUID `GPU-a83573c3-9e4c-50a6-0e76-e85127edcaaf`）顺序执行独立原生 PyTorch 与严格 shim 的公开 `swift infer`。沿用上述两条真实提示，FP32/eager、batch 2、非流式 greedy，每条最多生成 1 token；运行时只读捕获两侧首次 `Qwen2ForCausalLM.forward` 的 CUDA 输入 ID 与 logits。候选父子进程启动和正常退出均记录 shim 标记、`use_cuda=1`、fallback 0。11222 在独立 worker 对拍：输入 ID 逐项相同，形状 `[2,1396]`；完整 logits 的形状均为 `[2,1396,151936]`，dtype 均为 FP32，设备均为 `cuda:0`。捕获的最后位置 `[2,151936]` logits 最大绝对差 `5.38826e-5`、相对 L2 `2.93652e-6`，argmax 相同；两侧公开结果 JSONL 逐字段相等。捕获未保存完整 logits 各位置的值，也未要求返回 hidden，因此 L1 仍是 partial。

首次运行键 `20261007-qwen2-public-infer-logits-ffeb` 的原生 oracle 在数据预处理阶段因 UNIX socket 路径过长退出，未执行模型 forward；日志保留。第二运行键 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-infer-logits-r2-ffeb/` 使用指向该运行目录的短 `TMPDIR` 路径后完成；其中保存了两侧原始日志、JSONL、首次 forward 的 NPZ/元数据、比较 JSON 及精确脚本，均未版本化。该诊断只覆盖首次 forward 的最后位置数值，不提升本功能面的 L4/L5 等级。

## 全部序列 logits 与参数元数据

Slurm 11233 在 Jittor `0c9810603d83b2aa53dd04051436a1265957ece7`（包含上游 `ffeb7bd80447ca78735e7cb96628ab88c1e9360b`）、ms-swift `88d727951203256baa564c643c651b6f8d90fd7e` 上，在 qh17 的 RTX 4090（UUID `GPU-dc9aeebd-1eae-0eee-47d1-97050e5f3252`）先跑独立 PyTorch oracle，后跑严格 shim 的同一公开 CLI、同一两条提示与 Qwen2-0.5B 权重；11234 在 worker 上比较结果。两侧输入均为 `[2,1396]`，首次 forward 完整 logits 均为 `[2,1396,151936]` FP32、`cuda:0`；290 个参数和 1 个 buffer 的名称、形状、dtype、device 元数据相同，模型参数全在 CUDA。全部位置 logits 最大绝对差 `2.44141e-4`，相对 L2 `1.14350e-6`；所有 token 位置 argmax 相同，公开 JSONL 两行逐字段相等。严格候选父子进程起止均记录 `use_cuda=1`、shim 标记真、fallback 0。运行键、NPZ、JSON、日志和比较脚本未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-infer-full-logits-ffeb/`。本采集只保留完整 logits 与参数元数据，没有捕获 hidden，也未审计初始状态的数值键，因此 L0/L1 仍为 partial，L4/L5 不升级。
