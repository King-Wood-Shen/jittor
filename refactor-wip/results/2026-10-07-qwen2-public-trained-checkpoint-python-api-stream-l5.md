# Qwen2-0.5B 训练后全参数 checkpoint：公开 Python API 流式生成限定 L5

## 判定

此前 ms-swift 88d7279 的 TransformersEngine.infer(RequestConfig(stream=True)) Python 生成器路径已在 Jittor 38c6e8de1927e0ce2add59fc9c2126b7cb3efc43 达到固定 Qwen2-0.5B 训练后 checkpoint 限定 L4。本次在真实 NVIDIA RTX 4090 上补充原生与严格 CUDA 的 2 次预热及 10 次稳态单请求，完成首内容事件、完整生成耗时、串行吞吐及显存同口径比较；归一化生成事件一致、候选 fallback0。只判定该固定短文本 Python 迭代器合同的限定 L5，不代表通用 ms-swift 性能。

运行键 20261007-qwen2-public-trained-checkpoint-python-api-stream-l5；原始 api_perf.py、worker.sh、bootstrap、逐次事件/计时/显存和比较 JSON 在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-checkpoint-python-api-stream-l5/。原生和候选分别加载 20261007-qwen2-public-causal-fullparam-gradaccum-fp32 中各自真实三步全参数纯 FP32/SGD 训练后 checkpoint-3；缓存 Qwen2-0.5B、eager attention、qwen 模板，固定 What is 1+1?，temperature=0、max_tokens=8。

## 方法和结果

- 原生 Slurm 10804、严格 CUDA 候选 10805、独立比较 10811 均 COMPLETED(0)，顺序运行于 cscg-qh09 同一 NVIDIA RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8；各次运行前整卡约 1 MiB 已用，没有并发复用 JIT 缓存。模型参数均在 cuda:0。
- 同一 Python 进程构造模型一次，预热 2 次后计时 10 次。每次调用前后执行 torch.cuda.synchronize()；从开始 engine.infer 到首个非空 Python 内容事件计首事件时间，从调用前到完全迭代生成器并同步计完整时间。首次 JIT/编译不计入稳态。每次迭代完成后才用 nvidia-smi 采样整卡 memory.used 和 compute-app 用量，采样不计时；非进程内峰值。
- 吞吐定义为 10 次共 80 个 completion token / 10 次完整迭代时间之和，不是并发吞吐。

| 后端 | 首内容中位延迟 | 首内容范围 | 完整迭代中位耗时 | 完整范围 | completion token/s | 整卡 memory.used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 PyTorch | 0.062971 s | 0.057661–0.065785 s | 0.204408 s | 0.193767–0.213313 s | 39.2530 | 2382 MiB |
| Jittor shim 严格 CUDA | 0.076348 s | 0.073410–0.080699 s | 0.181156 s | 0.176412–0.192263 s | 43.9195 | 4404 MiB |

候选/原生首内容中位延迟比 1.212430，完整迭代中位耗时比 0.886248，串行 token 吞吐比 1.118882；整卡显存差 +2022 MiB。compute-app 用量原生 2372 MiB、候选 4390 MiB，均为单进程。HTTP deploy 的候选整卡显存为 4817 MiB，包含另一个 408 MiB CUDA 分发器进程，不能与本 Python 单进程口径直接当作模型显存变化比较。单次实验中的时间比仅为观察值。

10811 在 NVIDIA worker 复核每侧 2 次预热、10 次稳态、GPU UUID、统计公式及 24 次响应的归一化事件逐项一致。文本均为“The expression 1+1 is a”，length、prompt/completion/total=26/8/34。候选从 Python 入口导入前到退出使用 jt.runtime.scope(use_cuda=1, backend_fallback=error) 与 forbid_backend_fallbacks()；start/end 两条记录均 use_cuda=1、shim marker 真、fallback=0。

## 未覆盖范围

本结果是 Python 生成器，不是 HTTP SSE；并发迭代器、长上下文、logprobs、工具调用、其他 checkpoint、adapter、dtype、GPU、跨机网络、峰值显存和长期稳定性均为 not-run。历史 TinyLlama streaming logprob 问题未重启，不能据此推断通过。完整 ms-swift 兼容矩阵仍未完成。
