# Qwen2-0.5B 训练后全参数 checkpoint：公开 Python API 流式生成

## 判定

基于 ms-swift 88d7279、Jittor 0996b39d3608ca28f4a185e61d93c5f9a6c1d6b4，复用真实 Qwen2-0.5B 三步全参数纯 FP32/SGD/梯度累积 2 训练后 checkpoint-3，经公开 TransformersEngine.infer 与 RequestConfig(stream=True) 的 Python 生成器路径。原生 PyTorch 与严格 CUDA 候选在两条短文本请求的归一化生成事件、累积文本、结束原因和 token 用量逐项一致，候选参数在 cuda:0 且全程 fallback0。仅计该固定 Python 流式生成路径限定 L4；这是 Python 迭代器，并非 HTTP SSE 服务。L5 十次稳态性能 not-run，完整 ms-swift 兼容矩阵仍未完成。

运行键 20261007-qwen2-public-trained-checkpoint-python-api-stream。原始 api_stream.py、worker.sh、严格 CUDA bootstrap、两侧原始事件及独立比较在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-checkpoint-python-api-stream/。两侧分别加载 20261007-qwen2-public-causal-fullparam-gradaccum-fp32 各自的 checkpoint-3；缓存 Qwen2-0.5B、torch.float32、eager attention、qwen 模板、greedy、max_tokens=8。未修改 ms-swift 产品源码。

## 证据

- 原生 Slurm 10788、严格 CUDA 候选 10796、独立比较 10799 均 COMPLETED(0)，顺序运行于 cscg-qh09 同一 RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8。加载日志确认各自训练后 checkpoint 与 device_map=cuda:0；首个模型参数均在 cuda:0。
- 两提示为 Hello 和 What is 1+1?。逐次读取 Python 生成器而非一次取完整响应：原生与候选分别产生 8 个、5 个非空响应事件；排除随机生成的 id、created 时间戳后，每个事件的 model/object/choices/usage 逐项相同。累积文本分别为“The passage you provided is not related to”与“The expression 1+1 is a”；finish_reason=length，最终 prompt/completion/total token 为 20/8/28 与 26/8/34，也与已通过的非流式 Python API 结果一致。
- 候选进程在导入公开 API 之前由 _state 测试 bootstrap 启用 jt.runtime.scope(use_cuda=1, backend_fallback=error) 和 forbid_backend_fallbacks()；start/end 两条记录均 use_cuda=1、shim marker 真、fallback=0。

## 未覆盖范围

Python 生成器流式首 token 延迟、十次稳态吞吐与显存 L5，长上下文、logprobs、工具调用、adapter、多模型/dtype、取消/重入及并发生成器均为 not-run。历史 TinyLlama streaming logprob 问题未重启，不能从本结果推断其通过。
