# Qwen2-0.5B 训练后全参数 checkpoint：公开流式 deploy 限定 L5

## 判定与范围

此前该真实训练后 checkpoint 的公开 swift.cli.main deploy SSE 流式服务已达到限定 L4。沿用 ms-swift 88d7279、Jittor 81839d98f1e32a437747cdeb7503bfdc8879fe32 和同一 Qwen2-0.5B 纯 FP32/SGD/梯度累积 2 三步训练后原生/候选 checkpoint-3，补充同节点真实 RTX 4090 的流式稳态性能。固定单条 What is 1+1? 提示、greedy、每次 8 个生成 token、单客户端串行 SSE 请求的限定 L5 证据完成。完整 ms-swift 矩阵仍未完成。

运行键 20261007-qwen2-public-trained-deploy-stream-l5；原始 worker.sh、严格 CUDA bootstrap、请求、逐次 SSE 事件/计时/显存、比较 JSON 和服务日志在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-deploy-stream-l5/。原生 Slurm 10742、严格 CUDA 候选 10743、独立比较 10745 均 COMPLETED(0)。服务顺序运行在 cscg-qh09 的 NVIDIA RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8，驱动 580.178.04；各服务启动前整卡约 1 MiB 已用。没有并发复用 JIT 缓存。

## 测量口径与结果

每侧先预热 2 次，再完成 10 次稳态。worker 内 time.perf_counter() 从发起 HTTP 请求前起计，首个非空 SSE 内容 chunk 作为首内容事件边界，收到 [DONE] 作为完整响应边界；包含本机 HTTP、服务处理、SSE 解析和客户端读取。首次 JIT 不计入稳态。吞吐为 10 次共 80 个 completion token / 10 次完整响应耗时之和，不是并发吞吐。每次响应后才通过 nvidia-smi 采样整卡 memory.used 与 compute-app 进程用量，采样不计入延迟；不是进程内峰值。

| 后端 | 首内容事件中位延迟 | 首内容范围 | 完整响应中位延迟 | 完整响应范围 | 串行 completion token/s | 整卡显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 PyTorch | 0.060476 s | 0.058653–0.066351 s | 0.202852 s | 0.197257–0.207351 s | 39.5166 | 2382 MiB |
| Jittor shim 严格 CUDA | 0.076775 s | 0.072821–0.086386 s | 0.179645 s | 0.174771–0.227327 s | 43.4502 | 4817 MiB |

候选/原生首内容中位延迟比 1.269503，完整响应中位延迟比 0.885596，串行 token 吞吐比 1.099541。候选存在一次 0.227327 s 的完整响应样本；只报告观察值，不将中位结果解释为稳定的普遍速度优势。整卡显存差 +2435 MiB；原生 compute-app 2372 MiB、1 个服务进程，候选 compute-app 合计 4798 MiB，其中分发器约 408 MiB、服务约 4390 MiB。这是整套服务的真实显存口径，不能当成单模型张量差。

10745 在 NVIDIA worker 验证各侧预热 2、稳态 10、GPU UUID 一致和所有统计公式。24 个流式响应的归一化 chunk、累积文本“The expression 1+1 is a”、length、26/8/34 token 用量及单个 [DONE] 在两侧逐次一致。候选 CLI 分发器和实际服务进程启动与请求后共 4 条探针均 use_cuda=1、shim marker 真、fallback=0；严格 backend_fallback=error 和 forbid_backend_fallbacks() 生效。

## 未覆盖范围

该 L5 只适用于固定训练后 checkpoint、短文本、8-token、单客户端本机 HTTP SSE。没有验证并发负载、长上下文、流式中断/重连、长时间稳态、其他模型/tuner、不同 dtype/GPU、跨机网络或峰值显存，均为 not-run。历史 TinyLlama streaming logprob 问题没有重启，也不能由本结果推断通过。
