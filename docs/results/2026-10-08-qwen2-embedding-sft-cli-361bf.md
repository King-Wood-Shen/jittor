# Qwen2-0.5B ms-swift Embedding `swift sft` CLI CUDA 复验

- 状态：当前 Jittor 基线下固定 Qwen2-0.5B/FP32/eager/InfoNCE 场景通过公开 `swift sft` CLI 完成原生与严格 shim 三步训练和 checkpoint 保存；CLI 入口层记 partial，不能据此宣布 embedding CLI 或 ms-swift 整体兼容。
- 日期：2026-10-08。
- Jittor 基线：`361bfac511b468d4b65745b43df122a8591c35d1`；上游 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 为其祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：embedding CLI 参数、InfoNCE/梯度或 torch shim CUDA 路径变化时；补齐全量参数/输入审计、恢复和性能协议时。

运行键 `20261008-qwen2-embedding-sft-cli-361bf-r2`。Slurm 13283 在 cscg-qh13 RTX 4090（UUID `GPU-5331cb2a-7ec6-ad9a-5890-cb43fd723398`，driver `580.178.04`）按先原生后严格 shim 的顺序执行。两侧使用同一公开入口 `python -m swift.cli.main sft`、本地模型与 tokenizer、Qwen template、FP32/eager、batch 4、长度 64、固定四条离线 anchor/positive/negative 数据、无 shuffle、seed 1234、InfoNCE、全量 tuner 但只训练 `model.norm.weight`、SGD lr `0.01`、三步。模型和数据 SHA 与 worker 记录一致；无网络下载。严格候选从启动到退出启用了 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`。父进程和子进程均记录 start/end `use_cuda=1`、shim marker 存在、fallback 0。

原生 loss 为 `[2.03235769, 2.03235316, 2.03234959]`，严格 shim 为 `[2.03236127, 2.03235745, 2.03235340]`；最大逐步 loss 绝对差 `4.29154e-6`。grad norm 最大相对差 `5.34611e-6`。保存 checkpoint 的 290 个 state 键完全相同；被冻结的 `model.layers.0.self_attn.q_proj.weight` 逐值相等，唯一训练参数 `model.norm.weight` 末态最大绝对差为 0、相对 L2 为 0，两侧相对原模型的更新 L2 均为 `5.85909e-4`。这组 CLI 证据未保存逐参数梯度/逐步权重、完整参数 device 清单或训练批次 token 身份；不能替代直接 EmbeddingTrainer API 报告中更细的 forward/gradient 证据，也未执行恢复。

Slurm 13281 是首个尝试，在原生数据预处理开始前因过长 `TMPDIR` 导致 `OSError: AF_UNIX path too long`，后续 `EOFError`，未进入模型训练。保留原日志于运行键 `20261008-qwen2-embedding-sft-cli-361bf-v1`。13283 改用短 worker 临时目录后，两侧模型训练与 checkpoint 保存完成；尾部两次只读比较作业 13297/13298 分别因比较器指向旧 checkpoint/bootstrap 文件名而失败/修正后通过，没有重跑模型。checkpoint、日志、比较器及结果 JSON 未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-embedding-sft-cli-361bf-r2/`；失败尝试在同目录前缀 `...-v1/`。

13283 shim 三步 runtime 为 `373.018 s`，首步含本轮 JIT 首次编译；原生 runtime 为 `4.293 s`。未做两次预热及至少 10 次同步稳态测量，二者不能作为 L5 性能对比。Slurm 13298 的比较只读已有产物并成功。Slurm 13299 在 cscg-qh17 RTX 4090 上通过 `bash tools/check_repo_layout.sh` 与 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure/test_packaging_structure.py`（8 passed、8 subtests passed）；发布清单检查通过，共 220 份 active Markdown。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | partial | CLI 真实构造并训练 Qwen2 embedding 模型、processor/template、dataset、trainer 与 SGD；保存 state 键相等，shim CUDA scope/fallback 已审计，但没有完整逐参数 device/dtype/requires-grad 清单。 |
| L1 | not-run | CLI 日志有 loss；未捕获同一输入下的 embedding、logits 或中间张量，不能把 loss 对齐推广为完整前向对齐。 |
| L2 | partial | 三步 loss/grad norm 对齐，唯一训练参数最终更新相同；没有逐步保存该参数梯度和更新，也没有 optimizer 状态逐项比较。 |
| L3 | not-run | 未做同进程或新进程 checkpoint 恢复。 |
| L4 | partial | 当前公开 `swift sft` 命令在原生和严格 CUDA shim 下均完成三步并写 checkpoint；因 L0-L3 证据缺口，不标完整层级通过。 |
| L5 | blocked | L0-L3 未完整通过，且没有满足预热与稳态次数的性能数据。 |
