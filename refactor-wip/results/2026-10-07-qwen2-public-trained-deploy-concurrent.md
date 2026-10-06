# Qwen2-0.5B 训练后全参数 checkpoint：公开 deploy 双客户端请求

## 判定

基于 ms-swift 88d7279、Jittor 07c1068f636b572bd313a74402ba35697412e76c，复用此前已经通过单客户端公开服务的真实 Qwen2-0.5B 纯 FP32 全参数三步训练后原生/候选 checkpoint-3。原生 PyTorch 和严格 CUDA 候选经 swift.cli.main deploy 本机 HTTP 服务各完成三轮两客户端同时发起的非流式 greedy 请求，回复逐字段与原生单客户端参考及彼此一致。仅判定这组双客户端公开入口的限定 L4；没有十次稳态计时与显存采样，L5 not-run。请求区间交叠只证明客户端并发，不证明服务内部确实合批。

运行键 20261007-qwen2-public-trained-deploy-concurrent；原始 worker.sh、bootstrap、concurrent_client.py、compare.py、响应与服务日志在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-deploy-concurrent/。模型为缓存 Qwen2-0.5B 训练后 checkpoint-3，FP32、eager attention、Transformers engine，服务 max_batch_size=2；请求分别为 Hello 与 What is 1+1?，temperature=0、max_tokens=8、stream=false。未修改 ms-swift 产品源码。

## 证据

- 原生 Slurm 10754、候选 10756 均 COMPLETED(0)，顺序运行于 cscg-qh09 同一 RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8，服务启动前显存 1 MiB。独立比较 10759 COMPLETED(0)。
- 每轮两个客户端线程经 barrier 同时发起 POST /v1/chat/completions。两客户端请求时间区间在原生与候选的三轮中全部交叠；每轮均获 HTTP 200、length、8 completion token。对 12 条响应逐字段检查 model、message、finish_reason、usage，与此前原生单客户端响应一致。
- 第一轮包含预热/首次 JIT：原生每请求约 0.568 秒，候选约 10.178 秒。此轮不能计入稳态性能；后两轮原生每请求约 0.172 秒，候选约 0.187–0.194 秒，但只有两轮，不满足 L5 至少十次的要求。
- 候选 CLI 分发器与实际 Uvicorn 服务进程在启动和三轮请求后各采探针一次，共 4 条均 use_cuda=1、shim marker 真、fallback=0；严格 backend_fallback=error 和 forbid_backend_fallbacks() 生效。

## 未覆盖范围

没有证明服务内部动态合批、队列公平性、过载、取消/超时、更多客户端、stream 与并发组合、长上下文、长期稳定性或十次稳态性能；这些均为 not-run。历史已跳过的 TinyLlama streaming logprob、GKD BF16、packing 等问题未重启。完整 ms-swift 功能矩阵仍未完成。
