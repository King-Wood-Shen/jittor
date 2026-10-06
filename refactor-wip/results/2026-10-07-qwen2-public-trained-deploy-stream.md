# Qwen2-0.5B 训练后全参数 checkpoint：公开 deploy 流式 HTTP 服务

## 判定

复用已验证的真实 Qwen2-0.5B 三步全参数纯 FP32/SGD/梯度累积 2 训练后 checkpoint-3，通过 ms-swift 88d7279 的真正 swift.cli.main deploy 公开入口，在 Jittor 27bc6036f737b7119119c393773c38fba28391ac 上完成两条短文本 greedy 流式聊天请求。原生 PyTorch 与严格 CUDA 候选的每个 SSE chunk 的 model、choices、usage、object 字段一致，累积文本和此前非流式原生响应一致。仅判定这组请求的限定 L4；L5 流式首 token 延迟、稳态吞吐和显存未测。

运行键 20261007-qwen2-public-trained-deploy-stream；原始 worker.sh、bootstrap、stream_client.py、compare.py、原始事件和比较结果在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-deploy-stream/。原生与候选分别加载 20261007-qwen2-public-causal-fullparam-gradaccum-fp32 中各自的 checkpoint-3，模型权重为缓存 Qwen2-0.5B。FP32、eager attention、Transformers engine，生成上限 8 token。测试期间未修改 ms-swift 产品源码。

## 运行和数值证据

- 原生 Slurm 10724、候选 10725 均 COMPLETED(0)，同一 cscg-qh09 的 RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8。两作业顺序执行；启动前显存约 1 MiB。独立比较作业 10730 COMPLETED(0)。
- 本机客户端向 POST /v1/chat/completions 发送两个 stream=true 请求，提示分别为 Hello 和 What is 1+1?，temperature=0、max_tokens=8。客户端逐行解析真正的 text/event-stream、JSON chunk 和 [DONE] 终止标记；每条至少两个非空内容 chunk。
- 请求 1 两侧各 8 个内容事件，累积文本“The passage you provided is not related to”；请求 2 两侧各 5 个内容事件，累积文本“The expression 1+1 is a”。两侧 finish_reason 均为 length，均只有一个 [DONE]。两个事件流逐事件的 model/choices/usage/object 相同，仅排除按请求生成的 id 与 created 时间戳。最终 token 用量分别为 prompt/completion/total=20/8/28 与 26/8/34，同此前原生非流式响应相同。
- 候选 CLI 分发器与实际 Uvicorn 服务进程，在启动和两次请求完成后分别记录 start/probe，共 4 条均 use_cuda=1、shim marker 真、fallback=0；服务使用 backend_fallback=error 和 forbid_backend_fallbacks()。bootstrap 覆盖子进程，避免历史 10708/10710 的只检查分发器问题。

## 边界

该结果只覆盖固定训练后 checkpoint、两条短提示、单客户端、非流式已验证路径之外的 SSE 流式公开服务。没有验证并发、客户端中断/重连、长上下文、工具调用、图片、多轮大批量、鉴权、其他模型或 tuner。流式性能 L5、峰值显存及长期稳定性均为 not-run。历史 TinyLlama streaming logprob 问题是不同模型与 logprob 路径，未重启，也不据本结果推断其通过。完整 ms-swift 功能矩阵仍未完成。
