# Qwen2-0.5B 训练后全参数 checkpoint：公开 deploy 双客户端限定 L5

## 判定

基于此前双客户端公开 deploy 限定 L4，对 ms-swift 88d7279、Jittor 0a08b1fab07e4ad98ef0044b43db55873dbac658 的真实训练后 Qwen2-0.5B checkpoint-3 补充 2 次预热和 10 次稳态双客户端请求。原生 PyTorch 与严格 CUDA 候选的所有响应逐字段一致、客户端区间交叠、候选零 fallback；同节点同 GPU 的限定双客户端 HTTP 性能证据达到 L5。完整 ms-swift 功能矩阵仍未完成；客户端并发不证明服务内部动态合批。

运行键 20261007-qwen2-public-trained-deploy-concurrent-l5。原始 worker.sh、concurrent_perf.py、compare.py、每轮请求时间与显存、服务及探针日志在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-deploy-concurrent-l5/。复用 20261007-qwen2-public-causal-fullparam-gradaccum-fp32 中各自的真实三步全参数纯 FP32/SGD 训练后 checkpoint-3；缓存 Qwen2-0.5B，eager attention、Transformers engine，服务 max_batch_size=2。两提示为 Hello 和 What is 1+1?，greedy、每条请求 8 completion token、stream=false。

## 运行与口径

- 原生 Slurm 10769、严格 CUDA 候选 10770、独立比较 10771 均 COMPLETED(0)。两服务顺序运行在 cscg-qh09 同一 RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8，驱动 580.178.04；各自启动前整卡约 1 MiB。无并发 JIT 缓存复用。
- 每轮两个客户端线程在 barrier 后同时发出 POST /v1/chat/completions；计时取最早发送到最后完整响应的区间，包含本机 HTTP 与解码。完整读取响应是同步边界，JIT/预热前 2 轮不计入稳态。客户端区间在全部 12 轮交叠。稳态 10 轮共 20 请求、160 completion token；吞吐按 160 / 十个双请求区间耗时之和计算，不是单请求延迟之和。
- 每轮请求后才采样 nvidia-smi 整卡 memory.used 和 compute-app 进程用量；采样不计时，单位 MiB，非峰值。每轮两条回复的 model/message/finish_reason/usage 均与此前原生单客户端参考逐字段相同，原生和候选彼此一致。

| 后端 | 双请求区间中位耗时 | 稳态范围 | completion token/s | 整卡 memory.used | compute-app 用量之和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 PyTorch | 0.174866 s | 0.172833–0.176301 s | 91.5359 | 2382 MiB | 2372 MiB，1 进程 |
| Jittor shim 严格 CUDA | 0.172835 s | 0.171675–0.177358 s | 92.3955 | 4817 MiB | 4798 MiB，2 进程 |

候选/原生中位双请求耗时比为 0.988381，吞吐比为 1.009390；差异很小，单次实验不支持普遍速度优势。整卡显存差 +2435 MiB，候选包含约 408 MiB CLI 分发器与约 4390 MiB 服务进程，不能当成单模型张量差。

10771 在 NVIDIA worker 逐轮复核 2 次预热、10 次稳态、请求区间交叠、20 条稳态响应、GPU UUID、统计公式与双方数值。候选 CLI 分发器及实际服务进程在启动和请求后共 4 条探针均 use_cuda=1、shim marker 真、fallback=0；严格 backend_fallback=error 与 forbid_backend_fallbacks() 生效。

## 未覆盖范围

服务内部是否合批、并发数超过 2、队列公平性、过载/取消/超时、并发流式、长上下文、不同模型/tuner/dtype/GPU、长期稳定性及峰值显存均为 not-run。此 L5 只适用于固定双客户端短文本非流式本机 HTTP 合同；历史已跳过问题未重启。
