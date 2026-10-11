# Qwen2-0.5B PEFT IA3 公开 `swift infer` 当前基线 CUDA 复验

- 状态：此固定 profile 的 L0/L1 通过；公开 CLI 已在原生与 strict shim 上端到端运行且输出相同，但因 L3 未运行，正式 L4 blocked；L2 不适用，L5 blocked。
- 日期：2026-10-11。
- Jittor 基线：`7c110a68e54e02d625a7d87c62a88b035e829489`；同步的 `origin/2.0-refactor`=`7a18abf295668d9b19da5fa1657f5606e84b65a0`，为提交祖先。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`（4.6.0.dev0）；Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1。
- 模型与 adapter：Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；IA3 `adapter_model.safetensors` SHA256 `e7de90795307c4ff2d8d928d5fa43c11bd6ff07174c174d9bb549ff9b85e4348`；配置 SHA256 `df9caeb0cf0cbe62682a14e90a260197a7ca1d92e3758867ef29113ec3d8c340`。
- 范围：公开 `python -m swift.cli.main infer`、Transformers backend、PEFT IA3、FP32/eager、单卡 RTX 4090、两条固定短 prompt、batch 2、greedy、每条生成最多 2 token。结论仅适用于此模型、adapter、请求与当前环境。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor CUDA/Torch shim、Qwen2/PEFT IA3 注入、Transformers generation 或 ms-swift `infer` 入口变化；补齐保存及新进程恢复后重新评估 L3/L4/L5。

## 执行与证据

运行键 `20261011-qwen2-ia3-public-infer-l01-v2`（Slurm 16270）在 `cscg-qh04` 的 NVIDIA RTX 4090 上先运行独立原生 PyTorch，再运行 strict CUDA shim；GPU UUID 为 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`。两侧使用相同模型、adapter、输入文件和 CLI 参数。模型/adapter/配置/prompt 哈希在 worker 上核验。完整采集器在实际 infer 子进程截取首个 prefill。

L0 比较了完整的 362 个命名参数和 1 个 buffer；每项的名称、shape、dtype、device、`requires_grad` 与值 SHA256 两侧完全相同。72 个 IA3 参数也都在清单中。两侧的全部参数与 buffer 均为 FP32、`cuda:0`，输入 IDs 和 attention mask shape 均为 `[2,26]` 且逐元素相同。

L1 保存并比较了 28 个数组：IDs、mask、完整 logits 与 25 层 hidden states。张量均在 CUDA、dtype/shape 一致且有限。Logits shape 为 `[2,26,151936]`，最大绝对误差 `1.05381012e-4`、相对 L2 `1.96029835e-6`，所有位置 greedy argmax 相同；hidden states 最大绝对误差 `4.04357910e-4`、最大相对 L2 `2.96811737e-6`。

公开 CLI 两侧均完成两条请求，各生成 2 token；JSONL 逐字段相同（响应分别为 `The passage` 与 `The expression`）。strict shim 的父/推理子进程 start/end 事件均记录 shim 标记、`use_cuda=1`、`fallback_count=0`；运行时明确记录 `CUDA enabled`。这证明入口已执行和输出一致，但按本 Skill 的逐层门槛，L3 未验收前不授予正式 L4 PASS。

首次键 `20261011-qwen2-ia3-public-infer-l01-v1` / job16259 的原生 CLI 完成，但 strict shim 在单独冷缓存首次初始化期间 55 分钟超时；该缓存仅生成 `jit_utils_core`，没有 shim 模型计算。其日志与锁文件保留，未清除或重跑。v2 改用既有 Qwen2 CUDA 缓存并完成；该缓存的锁元数据指向已终态比较进程，提交前队列无活动使用者。正确性作业允许缺失的 JIT 库增量编译，因此本次不声称预热后缓存稳定，也没有性能测量。

原始日志、JSON、NPZ、manifest、脚本及编译缓存未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-ia3-public-infer-l01-v2/`；失败键证据在同级 `...-l01-v1/`。此前 `349d387` 基线的直接 PEFT 前向结果与更早 CLI 记录保留在对应历史报告中，不被本次结论覆盖或外推。

## 仓库文档门禁

唯一门禁键 `20261011-qwen2-ia3-public-infer-docgate-v2`（Slurm 16274）在 `cscg-qh04` 完成：`tools/check_repo_layout.sh` 报告 238 个活动 Markdown 文件且布局通过；Torch-mode `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed（184.39 秒）。skip 原因是 2 个 accelerator、2 个 declared 和 4 个环境相关项（解释器缺 Jittor 安装解析目标与 pytest-xdist）。节点有 RTX 4090，但该结构测试运行时 Jittor 日志显示没有 nvcc 并继续 CPU 路径；因此此门禁只证明文档/打包/结构契约，不作为 CUDA 计算证据。门禁 v1/job16273 的两个失败分别发现本报告未列入结果 toctree、生成的根 `MANIFEST.in` 未更新；已补齐结果 toctree 并在 v2 worker 中运行 manifest 生成器及 `--check`，之后全门禁通过。两轮原始日志和 manifest 均保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-ia3-public-infer-docgate-v1/` 与 `...-v2/`，未版本化。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | PASS | 362 参数、1 buffer 元数据与值哈希完全相同；72 个 IA3 参数及全部状态驻留 CUDA。 |
| L1 | PASS | 同权重与输入下完整 logits、25 层 hidden states 有限、设备/dtype/shape 一致并在误差门槛内，greedy argmax 相同。 |
| L2 | not-applicable | 此 profile 是推理，不含反向或更新。 |
| L3 | not-run | 未测试同进程和新进程的模型/adapter 保存与恢复。 |
| L4 | blocked | 公开 `swift infer` 已执行且输出匹配；按逐层门槛等待 L3。 |
| L5 | blocked | L3/L4 未完整通过，未按协议进行十次稳态性能测量。 |
