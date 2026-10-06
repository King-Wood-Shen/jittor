# 真实 Qwen2-0.5B 公开 infer 分发器 CUDA 验证

- 状态：限定 L4 PASS，覆盖 `python -m swift.cli.main infer` 官方分发器及其子进程；全 ms-swift CLI、L5 与功能面矩阵未完成。
- 基线：Jittor `760561bcc`，ms-swift `88d7279`，Python 3.11.15；真实本地缓存 Qwen2-0.5B FP32/eager，Qwen 模板、Transformers 推理引擎，完全离线。
- 协议：固定 JSONL 的 `Hello` 与 `What is 1+1?`，batch 上限 2，greedy、8 个新 token、非流式。原生与候选在独立 Slurm NVIDIA worker 运行相同 CLI 参数，保存官方 `result_path` JSONL。候选从父进程和子进程导入开始使用 `JITTOR_TORCH_SHIM=1`、`use_cuda=1`、`backend_fallback="error"` 与 `forbid_backend_fallbacks()`，独立 `JITTOR_HOME`。
- 原生 10047 COMPLETED0：cscg-qh15 RTX 4090 GPU-afd56a2e-3deb-3bd7-f122-1075e6de8956，driver 580.178.04。子进程实际加载 `device_map=cuda:0`，两条结果、总计 16 个新 token。
- 候选 10055 COMPLETED0：cscg-qh13 RTX 4090 GPU-0a3bba05-32f4-4794-0fd0-8267f4bb390e，driver 580.178.04。父/子进程两个不同 PID 的 bootstrap 起止记录均为 shim 标记真、`use_cuda=1`、fallback 增量 0；子进程实际加载 `device_map=cuda:0`，运行时 GPU 占用约 3382 MiB。两条官方 JSONL 逐字段与原生完全相同，独立审计 10074 COMPLETED0（cscg-qh04 RTX 4090 GPU-4b797e8c-3982-9b01-dc17-43d77960643c）。
- 首轮候选 10049 在 bootstrap 导入 Jittor 时，使内部 `python -m jittor_utils.query_cuda_cc` 查询进程再次执行同一 sitecustomize，形成递归等待，约 4 分钟后取消；Slurm worker 进程树和 `attempt1-root-cause.txt` 留证。第二轮仅在测试环境的 sitecustomize 中把 Jittor 启动限定于公开 CLI 父/子进程，未改 Jittor 或 ms-swift 产品源码。

## 可复核结果和限制

两条响应分别为 `The passage you provided is not related to` 与 `The expression 1+1 is a`。审计比较完整 JSONL（含消息、标签和响应）以及父子进程起止记录，输出 `QWEN2_PUBLIC_CLI_PASS 2 2 fallback=0`。本次 CLI 未请求 logprobs，生成 token ID 未单独捕获；仅计入文本与总生成 token 数的公开路径对齐。真实模型的逐层 logits/缓存解码另有独立 L1 证据，不能由此报告替代。

10055 的 338 秒生成时间包含首次 JIT 和缓存构建，不能作为 L5 稳态性能。已安装的 `swift` console script 在本隔离 venv 中不存在，故该二进制包装入口维持 not-run；stream、server/client、其他模型、BF16 和训练入口同样不由此结果覆盖。

原始数据、`prompts.jsonl`、worker、bootstrap、审计脚本、Slurm 日志和父子进程 fallback 记录均在 `/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261006-qwen2-public-infer-cli`。原工作树 30 个脏补丁及 2 个未跟踪路径未触碰。

## 流式文本分支（独立配置）

10086 原生和 10087 候选在 cscg-qh13 RTX 4090 GPU-3b3bba8a-5ea1-f84a-beb4-a9443a4f5e13（driver 580.178.04）顺序运行同一公开分发器，只把 `--stream` 改为 `true`。两侧均 COMPLETED0，各保存两条官方 JSONL，总共 16 个新 token。10092 在 Slurm worker 独立审计两份 JSONL 全字段一致、候选父/子进程各自起止 `use_cuda=1`、shim 标记真、fallback0，输出 `QWEN2_PUBLIC_CLI_STREAM_PASS 2 2 fallback=0`。流式响应文本与上方非流式固定提示相同，但本分支的独立结论来自流式原生/候选直接对照。

此处仅覆盖流式文本，不请求 logprobs，不触及已跳过的 TinyLlama streaming logprob。候选 140 秒窗口包含此前未编译的流式 CUDA kernel，不能列为 L5。原始证据在同一未版本化运行目录的 `*-stream*` 文件中；其他模型、并发、server/client 和 token ID 仍未验证。
