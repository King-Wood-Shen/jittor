# Qwen2-0.5B ms-swift `swift deploy` 服务端/客户端 CUDA 复验

- 状态：当前基线固定 Qwen2-0.5B FP32/eager 配置下，公开 Transformers 服务的 L0/L1/L4/L5 通过；L2/L3 不适用。结论限于本报告固定服务 profile，不代表其它模型、后端或 ms-swift 整体兼容。
- 日期：2026-10-08；当前基线补验：2026-10-11。
- 历史基线：`d11887358eb17daa8abe2ddf22f6ff1d2c8c5882`；当前 Jittor 基线：`d66fece475851205cd6413e2aa341a649042b4bb`；上游 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：`swift deploy`、TransformersEngine、HTTP generation、Jittor shim 生命周期或请求批处理变化时；改变服务模型、dtype、设备或并发时。

运行键 `20261008-qwen2-service-infer-361bf-r5`。Slurm 13324 在 cscg-qh15 RTX 4090（UUID `GPU-43cb8a5a-a194-7846-232c-b460bee1ad9e`，driver `580.178.04`）先启动未设置 shim 的原生 `python -m swift.cli.main deploy`，再启动严格候选。两侧均为本地 Transformers backend、FP32/eager、Qwen2 causal LM、同一模型权重和 served model name。客户端使用 OpenAI 风格 `/v1/models` 与 `/v1/chat/completions`，固定请求 “What is the capital of France?”、temperature 0、seed 1234、max_tokens 8；每侧共 12 请求，前 2 请求作为预热，其后 10 个串行请求计稳态。

原生与候选 12 个响应的文本、finish reason、prompt/completion/total token 用量逐项完全相同，每个返回 8 个 completion tokens，文本为 “The capital of France is Paris.”。稳态 HTTP 请求中位时延原生 `175.554 ms`、shim `169.300 ms`，shim/native 为 `0.9644`；按固定 8-token 返回折算吞吐分别为 `45.57` 和 `47.25 tokens/s`。shim 首个预热请求耗时 `106.569 s`，与 JIT 首次请求有关，不包含在 10 次稳态中位数。没有记录进程显存或 allocator 峰值，性能结果不满足完整 L5 显存口径。

候选服务端和其承载请求的进程均在启动时记录 `use_cuda=1`、shim marker 存在、fallback 0；完成服务请求的 Uvicorn 子进程在 SIGINT 正常关闭时记录 `fallback=0`。外层 CLI 父进程记录了 start，但实验 harness 结束它时没有 end 行；因此只把子服务进程的结束 fallback 记为零。底层公开 `TransformersEngine` 在相同 Qwen2 checkpoint、ms-swift 版本和 Jittor 源码上已有 L0/L1 逐参数设备/metadata 与 31 步 logits/token 证据，见 [`2026-10-08-qwen2-causal-transformersengine-api.md`](2026-10-08-qwen2-causal-transformersengine-api.md)。该报告基线到本报告 HEAD 的 `python/`、`compat/`、`adapters/`、`backends/`、`src/` 差异为空；此处服务端本身未重新采集完整 state-dict device 清单或逐步 logits，故仍保守记 partial。

实验脚本、服务日志、bootstrap 记录、12 组 JSON 响应与比较结果未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-service-infer-361bf-r5/`。期间保留了失败 harness 尝试：13307 shell 语法错误且未启动服务；13308 因 worker `HTTP_PROXY` 未绕过 localhost 而停在健康检查；13309 bootstrap 文件漏拷，候选响应未启严格 fallback 审计，结果不采信；13323 在开跑前发现清理函数递归错误并取消。13324 的 ms-swift `DeployArguments` 将候选请求端口 `31324` 改绑到 `31325`，原健康检查仍轮询旧端口；随后在同一 GPU worker 上直接从实际监听端口完成严格候选 12 个请求，并逐进程发 SIGINT 收尾。该 batch 作业退出码非零，响应与审计文件仍保留；比较作业 13328 成功。Slurm 13331 在 GPU worker 上通过布局门禁与 `tests/structure/test_packaging_structure.py`（8 passed、8 subtests passed）；Slurm 13333 通过仓库布局门禁及完整 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`（1386 passed、6 skipped、1027 subtests passed，11 分 30 秒）。6 项跳过中，2 项为声明跳过，另有 2 项因该解释器未安装 Jittor、2 项因未安装 pytest-xdist；pytest 汇总另计 4 项其他跳过原因。完整运行日志保存在上述运行目录。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS | job16188 原生服务与 job16189 strict shim 服务实际加载的 Qwen2ForCausalLM，各有 291 个参数/缓冲项；名称、shape、dtype、device 元数据逐项一致，FP32 且全 CUDA、有限。两个作业在 cscg-qh13 RTX 4090（UUID `GPU-c050abb2-8bea-6b3a-b793-236e283fa373`）顺序运行；模型权重 SHA256 与固定 checkpoint 一致。 |
| L1 | PASS | 同一个 HTTP 请求的 `input_ids`/mask shape `[1,26]` 逐项相同；生成 8 步各自捕获完整 `[151936]` CUDA FP32 logits，均有限，最大绝对差 `2.72989e-5`、相对 L2 `8.40818e-7`，逐步 argmax 100% 一致；生成 IDs、公开文本与 finish reason 相同。 |
| L2 | not-applicable | 无状态纯推理服务，不执行反向或 optimizer 更新。 |
| L3 | not-applicable | 无训练/续训状态的生成服务请求。 |
| L4 | PASS（固定配置） | 公开 `swift deploy` 服务通过 OpenAI 风格 HTTP endpoint 完成固定请求；job16189 以单请求补齐当前服务进程 L0/L1 审计，job16250 在同一 profile 下完成 12 次 HTTP 请求。job16189 外层作业因独立收尾 awk 检查引用缺失的 `shim-bootstrap.log` 而 exit 2；服务请求及模型/生成审计产物完整，shim 服务进程 start/end 事件均记录 fallback 0。 |
| L5 | PASS（固定配置） | job16250 在同一 RTX 4090、相同模型/输入与服务配置下，两次预热后各测 10 次串行稳态请求；响应相同、shim 所有进程 strict CUDA 且 fallback 0、预热后 JIT `.so` 清单稳定。HTTP 中位时延 native/shim 为 169.686/165.015 ms，P95 为 170.657/166.105 ms，中位延迟比 0.9725，按固定 8-token 输出折算吞吐 47.21/48.45 token/s。`nvidia-smi` 进程采样峰值 native/shim 为 2428/3382 MiB（样本 30/212）；shim峰值包含独立 CLI 父进程和承载模型的 Uvicorn 子进程，不与 Torch allocator 数字混用。首次 JIT/启动不计入稳态。 |

## 2026-10-11 当前基线 L0/L1 服务进程补证

新运行键 `20261011-qwen2-service-infer-l01-audit-v1` / job16188 先以原生 PyTorch 启动公开 `swift deploy` 服务，在服务实际加载模型返回处采集全部参数和缓冲区元数据，并在 `lm_head` 前向 hook 中只读捕获生成 logits、服务接收的 prompt IDs/mask 与生成 IDs。原生服务成功完成同一固定请求并留下完整产物。该作业随后因 harness 的 oracle 文件 glob 将真实 `state.<pid>.json` 写成了预期 `state-*.json` 而退出，shim 未启动；原生产物保留且被下一个运行只读复用。

新的 shim-only 运行键 `20261011-qwen2-service-infer-l01-audit-v2` / job16189 在同一 cscg-qh13 RTX 4090 上启动 strict shim 公开服务并发送相同 HTTP 请求。候选服务进程事件为 shim marker=true、`use_cuda=1`，完整状态清单 291 项全 CUDA/FP32/有限。原生与候选状态的名称、shape、dtype、device 逐项相同；输入 ID/mask 完全相同，8 个解码步的 `[8,151936]` logits 均有限，最大绝对差 `2.7298927e-5`、相对 L2 `8.4081810e-7`、逐步 argmax 一致率 100%；生成序列、响应文本及 finish reason 相同。候选服务进程正常退出事件记录 `fallback=0`。

job16189 的模型计算、HTTP 请求和比较器均成功；Slurm exit 2 来自脚本末尾残留的 awk 检查尝试读取本 v2 worker 未配置的 `shim-bootstrap.log`。其余服务进程审计事件及比较 JSON 完整，故本报告只依据已落盘且通过的逐项证据升级 L0/L1/L4，不把外层 harness exit 误作 shim 兼容失败。job16188/16189 原始日志、状态 JSON、NPZ logits、HTTP 响应和脚本均未版本化，分别位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-service-infer-l01-audit-v1/` 与 `20261011-qwen2-service-infer-l01-audit-v2/`。本次 logits hook 会同步并复制 logits 到 CPU，故只作正确性审计，不用于性能测量；L5 沿用独立旧运行数字并保留显存缺项。

## 2026-10-11 L5 进程显存补证尝试

运行键 `20261011-qwen2-service-infer-l5-v1` / Slurm job16191 在 `cscg-qh17` RTX 4090 上先完成 native 固定请求的 2 次预热与 10 次测量外加全部 12 个请求，`nvidia-smi` 采样得到 36 行。strict shim CLI 记录 `use_cuda=1`、shim marker 存在、fallback 0，并在日志中完成 Uvicorn startup/监听；但 harness 未观察到 `/health` 响应便走 readiness 超时分支并向服务发送 SIGINT。候选没有 HTTP 请求/响应文件、显存采样文件或收尾事件，所以不能计算 native/shim 性能比，也不能把这次启动事件算作服务 L4/L5 证据。与 job16189 同类成功 shim 服务相较，16191 日志缺少任何 health 访问记录；现有证据无法区分 shim 启动时间压线、health 请求未成功或其它 readiness 等待问题，尚未定位成 shim 实现缺陷。原始 worker、server/bootstrap 日志、native 响应与显存采样未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-service-infer-l5-v1/`。该次失败当时未改变 L5 partial 状态；后续完整 L5 证据见 v2 部分。原运行键/job 不重跑。

## 2026-10-11 L5 服务稳态与进程显存复验

新运行键 `20261011-qwen2-service-infer-l5-v2` / Slurm job16250 在 `cscg-qh04` RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`）同一 allocation 中顺序运行 native 和 strict shim 公开 `swift deploy`。两侧均使用 Qwen2-0.5B、FP32/eager、Transformers backend、`cuda:0`、同一固定提示与确定性 8-token 请求；各自完成 2 次预热和 10 次计时请求。worker 从真实服务监听端口探测 readiness，native 与 shim 分别在端口 47250 和 47251 收到 `/health` 200、`/v1/models` 200 及 12 个 `/v1/chat/completions` 200。原生与 shim 的完整响应对象按比较器判定相同，12 个返回均为 8 个 completion tokens。

native 与 shim 的 10 次稳态 HTTP 中位时延分别为 `169.686 ms` 和 `165.015 ms`，P95 分别为 `170.657 ms` 和 `166.105 ms`，shim/native 中位数比为 `0.9725`；按 8-token 固定返回折算吞吐为 `47.21` 与 `48.45 tokens/s`。`nvidia-smi` 进程采样分别记录 30 与 212 个样本，观测进程峰值为 `2428 MiB` 与 `3382 MiB`。shim 峰值由 CLI 父进程 `408 MiB` 与承载模型的 Uvicorn 子进程 `3382 MiB` 组成；这是进程级物理显存采样，不是 Torch allocator 峰值。采样脚本和进程 PID 清单均随 state 原始产物保留。两侧 UUID 相同；shim bootstrap 的父、子进程 start/end 事件均为 `use_cuda=1`、marker=true、fallback=0。预热请求后与稳态请求后的 `.so` 清单和内容 hash 相同，因此本测量区间无增量编译；冷启动与 JIT 不计入稳态数值。

job16250 完成并 exit 0；原始 worker、两侧完整服务日志、readiness 探测、HTTP 响应、进程显存采样、PID、shim fallback 事件、缓存快照及比较 JSON 未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-service-infer-l5-v2/`。v1 readiness 失败与 v2 新运行键相互独立：v1 不重跑，v2 的 L5 结论只适用于上述单卡、固定请求与并发配置。

报告门禁运行键 `20261011-qwen2-service-infer-l5-report-docgate-v1` / Slurm job16251 于 cscg-qh04 完成并 exit 0。`bash tools/check_repo_layout.sh` 和文档治理检查通过（237 个活动 Markdown）；Torch 模式 `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed（180.84 秒）。8 项跳过由 2 项 accelerator、2 项声明跳过、4 项其它跳过组成；后者含 2 项当前测试解释器未安装 Jittor、2 项未安装 pytest-xdist。门禁日志未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-service-infer-l5-report-docgate-v1/`。
