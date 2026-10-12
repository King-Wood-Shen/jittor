# Qwen2-0.5B PEFT IA3 公开 `swift infer` 当前基线 CUDA 复验

- 状态：此固定 profile 的 L0/L1/L3/L4 在 `7c110a68e54e02d625a7d87c62a88b035e829489` 基线通过；L5 在当前 `a0bf19b817ea530ccebac925732ad70b0ba22da0` 基线通过；L2 不适用。公开 CLI 原生与 strict shim 输出一致。
- 日期：2026-10-12。
- Jittor 基线：正确性证据为 `7c110a68e54e02d625a7d87c62a88b035e829489`；L5 运行在 `a0bf19b817ea530ccebac925732ad70b0ba22da0`，`python/jittor` tree=`b066dc9e303d2d3e7eb2c4bd4a824ab68657e835`。同步的 `origin/2.0-refactor`=`7a18abf295668d9b19da5fa1657f5606e84b65a0`，为提交祖先。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`（4.6.0.dev0）；Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1。
- 模型与 adapter：Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；IA3 `adapter_model.safetensors` SHA256 `e7de90795307c4ff2d8d928d5fa43c11bd6ff07174c174d9bb549ff9b85e4348`；配置 SHA256 `df9caeb0cf0cbe62682a14e90a260197a7ca1d92e3758867ef29113ec3d8c340`。
- 范围：公开 `python -m swift.cli.main infer`、Transformers backend、PEFT IA3、FP32/eager、单卡 RTX 4090、两条固定短 prompt、batch 2、greedy、每条生成最多 2 token。结论仅适用于此模型、adapter、请求与当前环境。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor CUDA/Torch shim、Qwen2/PEFT IA3 注入、Transformers generation 或 ms-swift `infer` 入口变化；补齐保存及新进程恢复后重新评估 L3/L4/L5。

正确性层仍绑定到报告所列的 `7c110a68` 基线，没有重跑。该基线之后的 compat 代码差异限于显式 `torch.Generator` 随机流、CPU `randperm` 与对应测试；本报告固定请求为 `temperature=0`、`do_sample=false`，不进入随机采样/数据 shuffle 路径。当前 HEAD 的 L5 另行核验参数数目、输入身份、greedy 输出、CUDA 放置、fallback 与稳态 JIT 清单；该适用性判断不把旧运行键重新标记为当前 HEAD 的运行结果。

## 执行与证据

运行键 `20261011-qwen2-ia3-public-infer-l01-v2`（Slurm 16270）在 `cscg-qh04` 的 NVIDIA RTX 4090 上先运行独立原生 PyTorch，再运行 strict CUDA shim；GPU UUID 为 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`。两侧使用相同模型、adapter、输入文件和 CLI 参数。模型/adapter/配置/prompt 哈希在 worker 上核验。完整采集器在实际 infer 子进程截取首个 prefill。

L0 比较了完整的 362 个命名参数和 1 个 buffer；每项的名称、shape、dtype、device、`requires_grad` 与值 SHA256 两侧完全相同。72 个 IA3 参数也都在清单中。两侧的全部参数与 buffer 均为 FP32、`cuda:0`，输入 IDs 和 attention mask shape 均为 `[2,26]` 且逐元素相同。

L1 保存并比较了 28 个数组：IDs、mask、完整 logits 与 25 层 hidden states。张量均在 CUDA、dtype/shape 一致且有限。Logits shape 为 `[2,26,151936]`，最大绝对误差 `1.05381012e-4`、相对 L2 `1.96029835e-6`，所有位置 greedy argmax 相同；hidden states 最大绝对误差 `4.04357910e-4`、最大相对 L2 `2.96811737e-6`。

公开 CLI 两侧均完成两条请求，各生成 2 token；JSONL 逐字段相同（响应分别为 `The passage` 与 `The expression`）。strict shim 的父/推理子进程 start/end 事件均记录 shim 标记、`use_cuda=1`、`fallback_count=0`；运行时明确记录 `CUDA enabled`。

L3 使用新键 `20261011-qwen2-ia3-restore-l3-v3` / job16277，基于 job16270 捕获的同一输入，不重跑公开 infer。原生 oracle 先导出 IA3 adapter，再在同进程通过新建模型对象加载导出物，随后另起进程从相同导出目录恢复；strict shim 执行相同步骤。两侧恢复前后完整 362 参数、1 buffer、72 IA3 参数的名称、shape、dtype、device、`requires_grad` 与值 SHA256 均相同。恢复前后输入 IDs/mask 及两 token greedy 结果相同；logits/25 层 hidden 均有限，恢复内误差符合本 profile L1 门槛且 argmax 相同。原生与 shim 的新进程恢复状态哈希完全相同；logits 最大绝对误差 `1.05381012e-4`、相对 L2 `1.96029835e-6`，hidden 最大绝对误差 `4.04357910e-4`、最大相对 L2 `2.96811737e-6`，greedy token IDs 相同。六条 shim runtime 事件均为 shim 标记、`use_cuda=1`、`fallback_count=0`。只读产物审计键 `20261011-qwen2-ia3-restore-l3-audit-v1` / job16278 PASS；它没有导入模型、JIT 或重算。

L3 的早期尝试也保留：job16275 因 harness 漏传路径变量在模型加载前退出；job16276 的状态哈希相同，但 harness 错将新进程浮点 logits 要求为 bitwise 相等，未保存输出，因此不作兼容判定。job16277 的原生和 shim 恢复阶段均完成并保存全部产物，但作业末尾比较器错误地要求原生侧也有 shim runtime 事件，故作业终态为 FAILED；只读审计 job16278 修正为只检查 shim 事件，完成所有已保存证据的比较并 PASS。L0/L1 的旧证据不被覆盖。

首次键 `20261011-qwen2-ia3-public-infer-l01-v1` / job16259 的原生 CLI 完成，但 strict shim 在单独冷缓存首次初始化期间 55 分钟超时；该缓存仅生成 `jit_utils_core`，没有 shim 模型计算。其日志与锁文件保留，未清除或重跑。v2 改用既有 Qwen2 CUDA 缓存并完成；该缓存的锁元数据指向已终态比较进程，提交前队列无活动使用者。正确性作业允许缺失的 JIT 库增量编译，因此当时未声称预热后缓存稳定，也未测性能。

L5 使用新运行键 `20261012-qwen2-ia3-public-infer-l5-v1`：worker 预检 job17931 与性能 job17932 均在 `cscg-qh04` 完成，RTX 4090 UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`。两侧模型均有 362 个 FP32 参数与 1 个 buffer，驻留 `cuda:0`；实际输入为 `[2,26]`，IDs SHA256=`303cce0bcca6c91b6b88790db22cfa382ba60510d53012360c453348885263ec`，mask SHA256=`8f67e95bfe90e219fe86c8e68eca68fb44527ef113e823be6be70a99ac14c531`。两侧各执行 2 次真实形状预热、10 次 CUDA 同步计时；每次均生成相同的 2×2 个 token，输出序列 SHA256 均为 `e17bb355218e7a0ddfa535776709e9c8d5c55c2ed665b1171a0c5249b1c7a082`，公开 CLI JSONL 的 response、messages、labels 与 dataset 字段一致。

原生 median/mean/P95 为 `47.762/47.755/48.222 ms`，吞吐 `83.748` 个生成 token/s，CUDA allocator peak allocated/reserved 为 `1,992,917,504/2,067,791,872 bytes`。Strict shim 为 `50.939/51.126/52.064 ms`、`78.526 token/s`、`1,982,381,056/3,077,570,560 bytes`；median latency shim/native 比值 `1.067`，吞吐比值 `0.938`。shim 父/推理子事件 `use_cuda=1`、fallback delta `0`。JITTOR_HOME 的 311 个 `.so` 清单在预检前后及整个性能作业前后逐项相同，no-build bootstrap 检查 2.8 秒通过，`jittor_core` 哈希无变化；首次 warmup 的 5.333 秒不计入测量。此 profile 最多生成 2 token，数字只描述两条短 prompt 的固定请求，不能外推到长上下文、流式生成或其它模型。

存储预检 job17931 与性能 job17932 都核实 NFS 源 `10.0.0.20:/home/xinshen`、64 KiB fsync 探针通过；预检可用 `20,219,691,008 bytes`、空闲 inode `39,489,869`，高于估算峰值 2 GiB 的 `2×峰值+2 GiB` 门槛。quota 命令无适用用户记录。两个进程的 multiprocess 临时目录清理均打印 `.nfs*` EBUSY 警告；输出、事件和计时结果已完整保存，未手动删除临时产物；之后只读扫描未发现残留 `.nfs*` 文件。

原始日志、JSON、NPZ、manifest、脚本及编译缓存未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-ia3-public-infer-l01-v2/`；失败键证据在同级 `...-l01-v1/`。此前 `349d387` 基线的直接 PEFT 前向结果与更早 CLI 记录保留在对应历史报告中，不被本次结论覆盖或外推。

## 仓库文档门禁

唯一门禁键 `20261011-qwen2-ia3-public-infer-docgate-v2`（Slurm 16274）在 `cscg-qh04` 完成：`tools/check_repo_layout.sh` 报告 238 个活动 Markdown 文件且布局通过；Torch-mode `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed（184.39 秒）。skip 原因是 2 个 accelerator、2 个 declared 和 4 个环境相关项（解释器缺 Jittor 安装解析目标与 pytest-xdist）。节点有 RTX 4090，但该结构测试运行时 Jittor 日志显示没有 nvcc 并继续 CPU 路径；因此此门禁只证明文档/打包/结构契约，不作为 CUDA 计算证据。门禁 v1/job16273 的两个失败分别发现本报告未列入结果 toctree、生成的根 `MANIFEST.in` 未更新；已补齐结果 toctree 并在 v2 worker 中运行 manifest 生成器及 `--check`，之后全门禁通过。两轮原始日志和 manifest 均保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-ia3-public-infer-docgate-v1/` 与 `...-v2/`，未版本化。L3 原始产物及只读审计输出保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-ia3-restore-l3-v3/` 与 `...-audit-v1/`；无效轮次 v1/v2 日志也保留，均未版本化。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | PASS | 362 参数、1 buffer 元数据与值哈希完全相同；72 个 IA3 参数及全部状态驻留 CUDA。 |
| L1 | PASS | 同权重与输入下完整 logits、25 层 hidden states 有限、设备/dtype/shape 一致并在误差门槛内，greedy argmax 相同。 |
| L2 | not-applicable | 此 profile 是推理，不含反向或更新。 |
| L3 | PASS | IA3 adapter 导出；同进程 fresh-object 与新进程均恢复完整模型/adapter 状态，输入与 greedy token 相同，logits/hidden 符合 L1 数值阈值；native/shim 状态哈希相同，strict CUDA 零 fallback。推理无 optimizer、scheduler、训练 RNG 或 dataloader 游标。 |
| L4 | PASS | 公开 `swift infer` 两侧均完成，JSONL 逐字段相同；适用 L3 已通过。 |
| L5 | PASS | 在 `a0bf19b817ea530ccebac925732ad70b0ba22da0` 基线完成 2 warmup + 10 CUDA-synchronized samples；输入与贪心输出两侧相同，shim 零 fallback、测量期 JIT 清单稳定；限本报告固定两条短 prompt 和 max_new_tokens=2。 |
