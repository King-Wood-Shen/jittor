# Qwen2-0.5B 训练后全参数 checkpoint：公开 Python 推理 API

## 判定

在 ms-swift 88d7279、Jittor 94a0ab77a1991f6c1feb8c2aeb355960a61918ae 上，真实 Qwen2-0.5B 三步全参数纯 FP32/SGD/梯度累积 2 训练后 checkpoint-3 经公开 Python API 的 TransformersEngine、InferRequest、RequestConfig 构造和生成。原生 PyTorch 与严格 CUDA 候选的两个响应逐字段一致，且与此前公开 deploy 原生响应一致；候选模型参数在 cuda:0、全程 fallback0。仅计固定 checkpoint、两条短文本非流式 greedy Python API 限定 L4；L5 十次稳态性能 not-run。完整 ms-swift 兼容矩阵仍未完成。

运行键 20261007-qwen2-public-trained-checkpoint-python-api；脚本、环境、模型路径、结果、严格探针与独立比较在 /home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-checkpoint-python-api/。原生/候选各加载 20261007-qwen2-public-causal-fullparam-gradaccum-fp32 中各自 checkpoint-3。缓存 Qwen2-0.5B，torch.float32、eager attention、qwen 模板、max_batch_size=2，RequestConfig(max_tokens=8, temperature=0)。未修改 ms-swift 产品源码。

## 运行和验证

- 原生 Slurm 10777、严格 CUDA 候选 10781、独立比较 10783 均 COMPLETED(0)，顺序运行于 cscg-qh09 同一 NVIDIA RTX 4090，UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8。两侧加载日志均显示相应训练后 checkpoint-3 与 device_map=cuda:0；首个模型参数的 device 均为 cuda:0。
- 请求为 Hello 和 What is 1+1?。两侧回复分别为“The passage you provided is not related to”和“The expression 1+1 is a”，finish_reason 均为 length，用量分别为 prompt/completion/total=20/8/28、26/8/34。10783 在 NVIDIA worker 对两侧 Python API 的 message、finish_reason、usage 逐字段比较，也与此前原生 HTTP 服务结果逐字段一致。
- 候选进程在 Python 入口导入之前由测试 bootstrap 启用 jt.runtime.scope(use_cuda=1, backend_fallback=error) 与 forbid_backend_fallbacks()；start/end 两条标记均 use_cuda=1、shim marker 真、fallback=0。bootstrap 只属于 _state 测试环境。

## 未覆盖范围

该结果不证明 Python API 的 stream、长上下文、logprobs、工具调用、LoRA/adapters、多模型、不同 dtype、输入/中间 logits 对齐或性能。Python API L5、恢复及长期稳定性均为 not-run。历史已跳过的 TinyLlama streaming logprob、GKD BF16 等问题未重启。
