# Qwen2 全参数训练 checkpoint 的公开 infer 加载与生成

基线：Jittor 集成分支 `cf39d86dfdde539df833f78e15c8fff36139fb30`，ms-swift `88d7279`。复用已通过的真实 Qwen2-0.5B 全参数纯 FP32、SGD、梯度累积 2、三步公开 SFT checkpoint：原生与候选分别从各自 `checkpoint-3/model.safetensors` 加载，未重新训练。两侧均经公开 `python -m swift.cli.main infer`、Transformers engine、eager、同一两条离线提示、batch2、greedy 8 token、非流式，输出 JSONL。

Slurm NVIDIA worker 原生 10686 和候选 10690 均 COMPLETED0，均在 cscg-qh06 的同一 RTX 4090（UUID GPU-e0c8764d-4b51-62e5-f0d3-4aa27a8ddb68）运行；独立 worker 10703 COMPLETED0。两份结果各两行，`dataset/labels/logprobs/messages/response` 五字段逐值一致，均生成总计 16 token。日志证实原生/候选分别指向其自身训练后的 checkpoint-3、`device_map='cuda:0'` 和 FP32；候选主进程及 CLI 子进程 start/end 四条标记均 `use_cuda=1, marker=True, fallback=0`。先前训练报告证实该两份 checkpoint 的全模型权重数值对齐及非零更新。

候选 10689 首次启动遇到 Jittor `jit_utils was rebuilt and cannot be reloaded in this process`，退出码 3；没有执行模型推理。按提示在无并发占用的同一缓存重启为 10690 后通过，JIT 重编译时间不计性能。结论只覆盖这一真实模型、训练后全参数 checkpoint 的公开非流式 greedy 推理及原生/严格 CUDA 输出对齐，可记该限定持久化加载加推理 L4；没有核对生成过程中每层 logits、其他 checkpoint 或训练状态恢复。L5 十次预热同步稳态 not-run，服务、stream、其他模型和 tuner 不外推。

原始 worker、提示、两侧日志/JSONL、bootstrap、comparison.json 与 Slurm 输出位于 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-fullparam-checkpoint-infer`。没有产品源码改动；独立集成树干净，原始 30 个修改加 2 个未跟踪路径保留。
