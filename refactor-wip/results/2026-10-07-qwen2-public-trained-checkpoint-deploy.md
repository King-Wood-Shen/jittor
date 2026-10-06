# Qwen2 训练后全参数 checkpoint 的公开 deploy 服务

基线 Jittor `1017d8857ab2ea3723a7a3083278188bdd543c70`、ms-swift `88d7279`。复用已通过三步数值对齐的真实缓存 Qwen2-0.5B 全参数纯 FP32/SGD/梯度累积 2 checkpoint-3；未重复训练。原生与候选分别经 `python -m swift.cli.main deploy` 启动 Transformers engine，绑定 worker 的 `127.0.0.1`；同一 Slurm worker 内请求 `/health`、`/v1/models` 和两次 `/v1/chat/completions`，固定 greedy 8 token。服务进程结束后没有对外开放端口。

原生作业 10707 COMPLETED0，严格 CUDA 候选作业 10711 COMPLETED0；两者均在 cscg-qh09 同一 RTX 4090（UUID GPU-335d6370-259b-bb0c-b471-930c0b4516a8）。独立 worker 10712 COMPLETED0：模型列表均返回 `qwen2-trained-checkpoint`；两条聊天回复的 `model/choices.message/finish_reason/usage` 逐字段一致，分别生成 8 token，prompt tokens 为 20/26。原生和候选日志确认各自加载训练后的 checkpoint-3。

严格性边界：初版候选 10708 的测试 bootstrap 只匹配 `swift.cli.main` 分发器，未覆盖它创建的 `swift/cli/deploy.py` Uvicorn 服务进程；虽然 HTTP 响应一致，不能拿来认定严格 CUDA。10710 试图补采，但仍只记录一个 start，按测试协议失败。修正测试 bootstrap 的子进程匹配后，10711 在分发器和实际服务进程各记录 `start` 与请求后的 `probe`：四条均 `use_cuda=1, marker=True, fallback=0`，并在两进程全程保持 `jt.runtime.scope(use_cuda=1, backend_fallback="error")` 与 `forbid_backend_fallbacks()`。HTTP 两次 200 且返回字段对齐；10712 复核标记、checkpoint 路径和响应。该修正仅在 _state 测试 bootstrap，没有产品源码改动。

结论：该真实 Qwen2 训练后全参数 checkpoint 的公开 `swift deploy` 本机 HTTP 非流式聊天服务限定 L4 通过。只覆盖两条短文本 greedy 请求；并发、多客户端、stream、认证、长上下文、服务稳定性、其他模型或 tuner 仍 not-run。没有十次预热同步稳态计时，L5 not-run；服务退出日志的快速中断堆栈不计推理失败，作业与请求均成功且请求后 fallback 为零。

原始脚本、bootstrap、请求/响应、服务与 Slurm 日志、`comparison-strict.json` 位于 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-trained-checkpoint-deploy`。原工作树 30 个修改加 2 个未跟踪路径保留，完整适配矩阵未完成。
