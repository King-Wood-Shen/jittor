# Qwen2-0.5B ms-swift `swift deploy` 服务端/客户端 CUDA 复验

- 状态：固定 Qwen2-0.5B FP32/eager 场景下，原生与严格 CUDA shim 的公开服务端均响应 12 次 OpenAI 风格 HTTP 请求，文本、结束原因与 token 用量相等，服务路径稳态请求时延可测；该服务面 L0/L1/L4/L5 仍记 partial，不代表所有部署后端或 ms-swift 整体兼容。
- 日期：2026-10-08。
- Jittor 基线：`d11887358eb17daa8abe2ddf22f6ff1d2c8c5882`；上游 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 为其祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：`swift deploy`、TransformersEngine、HTTP generation、Jittor shim 生命周期或请求批处理变化时；补齐服务进程完整状态审计与显存口径时。

运行键 `20261008-qwen2-service-infer-361bf-r5`。Slurm 13324 在 cscg-qh15 RTX 4090（UUID `GPU-43cb8a5a-a194-7846-232c-b460bee1ad9e`，driver `580.178.04`）先启动未设置 shim 的原生 `python -m swift.cli.main deploy`，再启动严格候选。两侧均为本地 Transformers backend、FP32/eager、Qwen2 causal LM、同一模型权重和 served model name。客户端使用 OpenAI 风格 `/v1/models` 与 `/v1/chat/completions`，固定请求 “What is the capital of France?”、temperature 0、seed 1234、max_tokens 8；每侧共 12 请求，前 2 请求作为预热，其后 10 个串行请求计稳态。

原生与候选 12 个响应的文本、finish reason、prompt/completion/total token 用量逐项完全相同，每个返回 8 个 completion tokens，文本为 “The capital of France is Paris.”。稳态 HTTP 请求中位时延原生 `175.554 ms`、shim `169.300 ms`，shim/native 为 `0.9644`；按固定 8-token 返回折算吞吐分别为 `45.57` 和 `47.25 tokens/s`。shim 首个预热请求耗时 `106.569 s`，与 JIT 首次请求有关，不包含在 10 次稳态中位数。没有记录进程显存或 allocator 峰值，性能结果不满足完整 L5 显存口径。

候选服务端和其承载请求的进程均在启动时记录 `use_cuda=1`、shim marker 存在、fallback 0；完成服务请求的 Uvicorn 子进程在 SIGINT 正常关闭时记录 `fallback=0`。外层 CLI 父进程记录了 start，但实验 harness 结束它时没有 end 行；因此只把子服务进程的结束 fallback 记为零。底层公开 `TransformersEngine` 在相同 Qwen2 checkpoint、ms-swift 版本和 Jittor 源码上已有 L0/L1 逐参数设备/metadata 与 31 步 logits/token 证据，见 [`2026-10-08-qwen2-causal-transformersengine-api.md`](2026-10-08-qwen2-causal-transformersengine-api.md)。该报告基线到本报告 HEAD 的 `python/`、`compat/`、`adapters/`、`backends/`、`src/` 差异为空；此处服务端本身未重新采集完整 state-dict device 清单或逐步 logits，故仍保守记 partial。

实验脚本、服务日志、bootstrap 记录、12 组 JSON 响应与比较结果未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-service-infer-361bf-r5/`。期间保留了失败 harness 尝试：13307 shell 语法错误且未启动服务；13308 因 worker `HTTP_PROXY` 未绕过 localhost 而停在健康检查；13309 bootstrap 文件漏拷，候选响应未启严格 fallback 审计，结果不采信；13323 在开跑前发现清理函数递归错误并取消。13324 的 ms-swift `DeployArguments` 将候选请求端口 `31324` 改绑到 `31325`，原健康检查仍轮询旧端口；随后在同一 GPU worker 上直接从实际监听端口完成严格候选 12 个请求，并逐进程发 SIGINT 收尾。该 batch 作业退出码非零，响应与审计文件仍保留；比较作业 13328 成功。Slurm 13331 在 GPU worker 上通过布局门禁与 `tests/structure/test_packaging_structure.py`（8 passed、8 subtests passed）；Slurm 13333 通过仓库布局门禁及完整 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`（1386 passed、6 skipped、1027 subtests passed，11 分 30 秒）。6 项跳过中，2 项为声明跳过，另有 2 项因该解释器未安装 Jittor、2 项因未安装 pytest-xdist；pytest 汇总另计 4 项其他跳过原因。完整运行日志保存在上述运行目录。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | partial | 公开 deploy 服务构造真实 Qwen2、tokenizer/template 与 Transformers backend；shim CLI/服务进程 strict CUDA 启用。底层同源 engine 有 290 参数 metadata/device 对拍，但本服务运行没有另存完整参数设备清单。 |
| L1 | partial | 同一请求下 12 次服务端响应文本、finish reason、token usage 完全一致；服务协议未返回/捕获 logits 或 hidden states。底层 engine 的 logits 证据来自已引用报告。 |
| L2 | not-applicable | 无状态纯推理服务，不执行反向或 optimizer 更新。 |
| L3 | not-applicable | 无训练/续训状态的生成服务请求。 |
| L4 | partial | 当前公开 `swift deploy` 服务端与客户端完成 12 次真实 HTTP 推理；端到端请求证据有效，但原 batch 健康检查端口选择错误使调度脚本未正常收尾，且 L0/L1 为 partial。 |
| L5 | partial | 两次预热后 10 次串行稳态时延/吞吐及原生比已测，strict child-server fallback 为 0；显存统计缺失，首次请求包含冷 JIT 启动耗时。 |
