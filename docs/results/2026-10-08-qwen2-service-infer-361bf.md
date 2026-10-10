# Qwen2-0.5B ms-swift `swift deploy` 服务端/客户端 CUDA 复验

- 状态：当前基线固定 Qwen2-0.5B FP32/eager 配置下，公开 Transformers 服务的 L0/L1/L4 通过；L2/L3 不适用。L5 有 10 次稳态时延/吞吐，但因缺进程显存口径仍 partial。不代表其它模型、后端或 ms-swift 整体兼容。
- 日期：2026-10-08；当前基线补验：2026-10-11。
- 历史基线：`d11887358eb17daa8abe2ddf22f6ff1d2c8c5882`；当前 Jittor 基线：`5f5ee433dc002b1f5b76154a6e8e8055961a31b8`；上游 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：`swift deploy`、TransformersEngine、HTTP generation、Jittor shim 生命周期或请求批处理变化时；改变服务模型、dtype、设备、并发，或补齐服务进程显存口径时。

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
| L4 | PASS（固定配置） | 公开 `swift deploy` 服务通过 OpenAI 风格 HTTP endpoint 完成 12 次请求；job16189 又以单个确定性请求补齐当前服务进程 L0/L1 审计并正常启动、服务、收尾。job16189 外层作业因独立收尾 awk 检查引用缺失的 `shim-bootstrap.log` 而 exit 2；服务请求及模型/生成审计产物完整，shim 服务进程 start/end 事件均记录 fallback 0。 |
| L5 | partial | 两次预热后 10 次串行稳态时延/吞吐及原生比已测，strict shim fallback 为 0；没有可比较的 shim 进程显存峰值，因此不满足完整 L5。job16191 的新采样尝试没有进入 HTTP 请求阶段，不提供性能或显存数据。首次请求包含冷 JIT 启动耗时，不纳入稳态。 |

## 2026-10-11 当前基线 L0/L1 服务进程补证

新运行键 `20261011-qwen2-service-infer-l01-audit-v1` / job16188 先以原生 PyTorch 启动公开 `swift deploy` 服务，在服务实际加载模型返回处采集全部参数和缓冲区元数据，并在 `lm_head` 前向 hook 中只读捕获生成 logits、服务接收的 prompt IDs/mask 与生成 IDs。原生服务成功完成同一固定请求并留下完整产物。该作业随后因 harness 的 oracle 文件 glob 将真实 `state.<pid>.json` 写成了预期 `state-*.json` 而退出，shim 未启动；原生产物保留且被下一个运行只读复用。

新的 shim-only 运行键 `20261011-qwen2-service-infer-l01-audit-v2` / job16189 在同一 cscg-qh13 RTX 4090 上启动 strict shim 公开服务并发送相同 HTTP 请求。候选服务进程事件为 shim marker=true、`use_cuda=1`，完整状态清单 291 项全 CUDA/FP32/有限。原生与候选状态的名称、shape、dtype、device 逐项相同；输入 ID/mask 完全相同，8 个解码步的 `[8,151936]` logits 均有限，最大绝对差 `2.7298927e-5`、相对 L2 `8.4081810e-7`、逐步 argmax 一致率 100%；生成序列、响应文本及 finish reason 相同。候选服务进程正常退出事件记录 `fallback=0`。

job16189 的模型计算、HTTP 请求和比较器均成功；Slurm exit 2 来自脚本末尾残留的 awk 检查尝试读取本 v2 worker 未配置的 `shim-bootstrap.log`。其余服务进程审计事件及比较 JSON 完整，故本报告只依据已落盘且通过的逐项证据升级 L0/L1/L4，不把外层 harness exit 误作 shim 兼容失败。job16188/16189 原始日志、状态 JSON、NPZ logits、HTTP 响应和脚本均未版本化，分别位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-service-infer-l01-audit-v1/` 与 `20261011-qwen2-service-infer-l01-audit-v2/`。本次 logits hook 会同步并复制 logits 到 CPU，故只作正确性审计，不用于性能测量；L5 沿用独立旧运行数字并保留显存缺项。

## 2026-10-11 L5 进程显存补证尝试

运行键 `20261011-qwen2-service-infer-l5-v1` / Slurm job16191 在 `cscg-qh17` RTX 4090 上先完成 native 固定请求的 2 次预热与 10 次测量外加全部 12 个请求，`nvidia-smi` 采样得到 36 行。strict shim CLI 记录 `use_cuda=1`、shim marker 存在、fallback 0，并在日志中完成 Uvicorn startup/监听；但 harness 未观察到 `/health` 响应便走 readiness 超时分支并向服务发送 SIGINT。候选没有 HTTP 请求/响应文件、显存采样文件或收尾事件，所以不能计算 native/shim 性能比，也不能把这次启动事件算作服务 L4/L5 证据。与 job16189 同类成功 shim 服务相较，16191 日志缺少任何 health 访问记录；现有证据无法区分 shim 启动时间压线、health 请求未成功或其它 readiness 等待问题，尚未定位成 shim 实现缺陷。原始 worker、server/bootstrap 日志、native 响应与显存采样未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-service-infer-l5-v1/`。原运行键/job 不重跑；本次结果不改变 L5 partial 状态。
