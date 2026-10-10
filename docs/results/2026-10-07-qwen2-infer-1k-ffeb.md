# Qwen2-0.5B 公开 infer CLI：新基线千 token 非流式复验

- 状态：2026-10-11 当前基线新增的固定非流式 CLI profile：L0/L1/L4 通过，L2/L3 不适用，L5 未运行；历史运行记录及各自当时的结论保留在下文。
- 日期：2026-10-07（历史运行）；2026-10-11（当前基线复验）。
- 基线：Jittor `af2cbfa3c`（父提交包含上游 `ffeb7bd80`），ms-swift `88d7279`，隔离 Python 3.11.15。
- 验证范围：真实缓存 Qwen2-0.5B、公开 `python -m swift.cli.main infer`、单 RTX 4090 CUDA、FP32、eager attention、非流式 greedy、最多 32 新 token、两条提示。
- 维护者：ms-swift CUDA 适配。
- 复查条件：改动 Qwen2 加载、tokenizer/template、Torch shim、CUDA executor 或公开推理入口；补完整 logits/hidden、构造状态和恢复审计时。

Slurm 作业 10968 在同一 GPU 上依次跑原生 PyTorch 与启用严格 CUDA/禁止 fallback 的 Jittor shim，之后用独立审计脚本复核产物；作业 10991 在 NVIDIA worker 另证原生环境为 PyTorch 2.6.0+cu124、CUDA 可用且无 shim 标记。两条真实 tokenizer 提示长分别为 1377、1286 token；公开入口统计合计 2701 prompt tokens，实际生成 49 tokens。两侧保存的完整 JSONL 两行逐字段相等，响应非空且回填到 messages。两侧日志都显示模型映射到 `cuda:0`；候选父子两个进程的启动与正常退出标记均为 `use_cuda=1`、shim 标记真、`fallback=0`。这证明该固定输入和配置的公开入口完成，但不替代前级逐张量验收。

| 层 | 状态 | 现有证据与缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型与公开入口构造并驻留 CUDA；初始状态键、逐项 dtype/device 未直接审计。 |
| L1 | partial | 完整序列 logits、25 组 hidden states 与 greedy 输出均已直接对拍并满足 CUDA 前向容差；该功能面的 L0 状态值与完整 buffer 元数据尚未审计，因此不提升整体层级。 |
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

## 全层 hidden-state 对拍

Slurm 11282 在 Jittor `0c9810603d83b2aa53dd04051436a1265957ece7`（上游基线 `ffeb7bd80447ca78735e7cb96628ab88c1e9360b`）、ms-swift `88d727951203256baa564c643c651b6f8d90fd7e` 上，在 qh10 的 RTX 4090（UUID `GPU-b36c5582-63c5-8f1f-0e9b-62d4465cc743`）先运行独立原生 PyTorch，再运行严格 shim 的同一 `swift infer` 入口、checkpoint、输入与 FP32/eager 配置；11283 在 worker 上比较捕获。两侧同一首次 forward 输入均为 `[2,1396]`、CUDA；logits 均为 `[2,1396,151936]` FP32 CUDA。25 组 embedding/decoder hidden states 均为 `[2,1396,896]` FP32 CUDA、无非有限值且形状/dtype 一致；最大绝对差出现在第 24 层，为 `5.98907e-4`，全层最大相对 L2 为 `1.44909e-6`，logits 最大绝对差 `2.44141e-4`、相对 L2 `1.14350e-6`。逐位置 argmax 及两条公开结果 JSONL 均相同；候选主进程和推理子进程起止均为 `use_cuda=1`、shim 标记真、fallback 0。提示、权重 SHA256 分别为 `bdb99f9e1fe097c595db2fba7c81cefcb49300e6dbac2595cf075fd6160abcaa`、`9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。未版本化的 NPZ、元数据、结果、日志和脚本位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-infer-hidden-ffeb/`。本次前向张量覆盖完整，但构造阶段的 buffer 元数据与初始状态值没有独立核对，故不把单次配置提升为整个推理面的 L0/L1 完成；L3/L4/L5 仍未验收。

## 2026-10-11 当前基线完整状态与非流式公开入口复验

新收敛协议独立复验键为 `20261011-qwen2-public-infer-1k-l01-v2`，Slurm job 16255（cscg-qh04，RTX 4090 UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）；ms-swift HEAD `88d727951203256baa564c643c651b6f8d90fd7e`，Jittor HEAD `007e97bf883c5f27986a227e612f7f445356f21b`。沿用固定提示 SHA256 `bdb99f9e1fe097c595db2fba7c81cefcb49300e6dbac2595cf075fd6160abcaa` 与模型权重 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。原生 PyTorch 先跑，严格 shim 随后在同一物理 GPU、FP32/eager、batch 2、greedy 非流式公开 `python -m swift.cli.main infer` 跑同一输入，每条最多生成 32 token。完整输入、参数/状态及 logits/hidden 证据保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-public-infer-1k-l01-v2/`；分析 job 16257 对已完成产物做只读全量比较。

两侧首次 forward 输入 ID 完全相同，形状 `[2,1396]`；290 个参数与 1 个 buffer 的键、shape、dtype、device、值 SHA256 一致，参数 `requires_grad` 标志也一致。两侧输入、logits 与 25 组 hidden 均为 CUDA FP32 且有限；logits shape `[2,1396,151936]`，最大绝对差 `2.44140625e-4`、相对 L2 `1.1434963e-6`，所有位置 greedy argmax 一致。hidden shape 均为 `[2,1396,896]`，最大绝对差 `5.9890747e-4`、最大相对 L2 `1.4490903e-6`，各层满足同一比较阈值。两条公开 JSONL 响应及对应 messages 逐字段相等，共生成 49 token。候选公开 CLI 父/推理子进程的 start/end 均记录 strict shim、`use_cuda=1`、`fallback_count=0`；产物比较进程也记录零 fallback。此证据只覆盖本节固定的非流式 profile。

| 层 | 当前固定 profile | 证据边界 |
| --- | --- | --- |
| L0 | PASS | 公开 CLI 完成真实模型/tokenizer/template/dataset 构造；两侧完整参数与 buffer 键、shape、dtype、device、参数可训练标志和值 hash 一致。 |
| L1 | PASS | 同输入 CUDA forward；全 logits、25 组 hidden 与确定性 greedy argmax 均通过，完整公开输出相等。 |
| L2 | not-applicable | 推理配置，不做反向或 optimizer 更新。 |
| L3 | not-applicable | 推理配置没有训练状态续训合同；此项不声称验证独立 server/session 恢复。 |
| L4 | PASS | 两侧均通过公开 `swift infer` CLI 端到端完成，并产出逐字段相同 JSONL。 |
| L5 | not-run | 本次有模型采集开销且非稳态协议；不使用其运行时间作性能结论。 |

采集器自身的失败尝试也保留：job 16254（键 `20261011-qwen2-public-infer-1k-l01-v1`）在原生 CLI 产出结果后因未将 native capture 钩子导入实际推理子进程而退出；job 16255 完成原生与 strict shim 两侧及张量采集，但首个比较器因在 shim 环境内、atexit 前读取自身尚未闭合的 runtime 事件而退出；job 16256 的只读比较另因过时地要求 `native-capture.json` 退出，均未得出兼容结论。job 16257（键 `20261011-qwen2-public-infer-1k-analysis-v2`）改为读取实际的 `native-forward.json/.npz` 并在独立进程退出后比较，完成并通过上述审计。旧运行键未重跑，原有各历史严格对拍结论均未覆盖或改写。此项不覆盖 streaming/KI-EXEC-011、其他上下文长度或 batch，也不代表整个 ms-swift 已兼容。

文档门禁 job 16258（cscg-qh04）通过：`bash tools/check_repo_layout.sh` 报告 237 个活动 Markdown、布局合格；Torch 模式 `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed（182.75 秒；session 收集 1392 项、执行 2409 项）。8 项跳过为 2 accelerator、2 declared、4 other；other 是解释器内未安装 Jittor 的 2 项和未安装 pytest-xdist 的 2 项。`git diff --check` 通过。
