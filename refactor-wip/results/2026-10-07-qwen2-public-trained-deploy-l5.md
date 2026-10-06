# Qwen2-0.5B 训练后全参数 checkpoint：公开 deploy HTTP 服务限定 L5

## 判定与范围

同一 ms-swift 88d7279、Jittor babce9288cea7d827509ec4fde1ebce492268913、RTX 4090 上，复用已在公开 deploy L4 报告验证的原生与严格 CUDA 候选三步全参数纯 FP32/SGD 训练后 checkpoint-3，通过真正的 swift.cli.main deploy 和本机 HTTP 客户端测量。限定串行、非流式、greedy、单个固定 prompt、每次 8 个生成 token；该服务合同的响应数值一致，L5 性能证据完成。完整 ms-swift 兼容矩阵仍未完成。

运行键：20261007-qwen2-public-trained-deploy-l5。原始脚本、请求、逐请求延迟与显存、服务日志和比较 JSON 在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-deploy-l5/。ms-swift checkpoint 来自 20261007-qwen2-public-causal-fullparam-gradaccum-fp32 的原生/候选各自 checkpoint-3。Python 3.11 venv 为 _state/ms-swift-cuda/20261003-python311/venv，模型权重为缓存的 Qwen2-0.5B；FP32、eager attention、Transformers engine，max_new_tokens=8。

## 方法与原始运行

- 原生 Slurm 10716、候选 10717 均 COMPLETED(0)；独立比较 10719 COMPLETED(0)。两次服务顺序运行于 cscg-qh09 的同一 NVIDIA GeForce RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8，驱动 580.178.04。每次服务启动前 nvidia-smi 报 1 MiB 已用。没有并发复用 JIT 缓存。
- 请求经 POST /v1/chat/completions；前 2 次预热，之后连续 10 次稳态，完整读取每次 HTTP 响应为同步边界。计时用 worker 上 time.perf_counter()，包含本机 HTTP 传输、服务排队与解码、读完响应；JSON 解析与 nvidia-smi 显存查询在计时区外。首次 JIT/构建没有纳入稳态统计。
- 吞吐定义为 10 次请求共 80 个 completion token / 10 次响应耗时之和，不是并发吞吐。显存口径为每次稳态响应完成后同一 GPU 的 nvidia-smi memory.used，另记 compute-app 进程用量之和，单位 MiB；均非进程峰值/训练显存。

| 后端 | 稳态中位响应延迟 | 稳态延迟范围 | 串行 completion token/s | 整卡 memory.used | compute-app 用量之和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 PyTorch | 0.176526 s | 0.174420–0.180415 s | 45.3154 | 2382 MiB | 2372 MiB，1 进程 |
| Jittor shim 严格 CUDA | 0.166794 s | 0.165656–0.168125 s | 47.9532 | 4817 MiB | 4798 MiB，2 进程 |

候选/原生的中位延迟比为 0.944871，串行生成 token 吞吐比为 1.058210；整卡显存差为 +2435 MiB。候选的两进程为 CLI 分发器约 408 MiB 与实际 Uvicorn 服务约 4390 MiB，原生只有服务进程约 2372 MiB。整卡显存是真实部署开销，但进程架构差异是该数值的组成部分，不能解释为单个模型张量的显存差。

10719 在 NVIDIA worker 逐条复核 24 个响应：文本“The expression 1+1 is a”、finish_reason=length、prompt/completion/total tokens 为 26/8/34，两侧全部一致；原生和候选各 10 个稳态样本，GPU UUID 一致，统计公式复算一致。候选 CLI 与实际服务进程在启动和请求后各采样一次，共 4 条 use_cuda=1、shim marker 真、fallback=0；严格 backend fallback error 生效。10719 日志以 COMPARE_PASS 结束。

## 限制与后续

此结果只对已通过 L0–L4 的这一训练后 checkpoint、本机 HTTP 固定 8-token 服务合同构成限定 L5；它没有证明其他模型、tuner、上下文长度、batch、并发、stream、BF16、其他 GPU 或跨机器网络性能。单次测量、未控制温度/功耗与服务进程内的显存峰值，0.94 延迟比仅为观察值，不作普遍速度结论。长上下文、并发、stream、生产认证与峰值显存均为 not-run。历史已跳过问题仍按原报告保持 blocked/not-run。
