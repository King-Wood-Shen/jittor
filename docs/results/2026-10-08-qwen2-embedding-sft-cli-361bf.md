# Qwen2-0.5B ms-swift Embedding `swift sft` CLI CUDA 复验

- 状态：固定 Qwen2-0.5B/FP32/eager/InfoNCE 场景已在公开 `swift sft` CLI 上完成原生与严格 shim 构造、三步训练、checkpoint 保存及中途恢复；当前源码基线下 L0-L4 对本配置通过，L5 未运行。该结果不代表其他 embedding 配置或 ms-swift 整体兼容。
- 日期：2026-10-08；L1/L2 补验：2026-10-09；L3/L0/L4 复核：2026-10-10。
- Jittor 基线：`361bfac511b468d4b65745b43df122a8591c35d1`；上游 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 为其祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：embedding CLI 参数、InfoNCE/梯度或 torch shim CUDA 路径变化时；补齐全量参数/输入审计、恢复和性能协议时。

运行键 `20261008-qwen2-embedding-sft-cli-361bf-r2`。Slurm 13283 在 cscg-qh13 RTX 4090（UUID `GPU-5331cb2a-7ec6-ad9a-5890-cb43fd723398`，driver `580.178.04`）按先原生后严格 shim 的顺序执行。两侧使用同一公开入口 `python -m swift.cli.main sft`、本地模型与 tokenizer、Qwen template、FP32/eager、batch 4、长度 64、固定四条离线 anchor/positive/negative 数据、无 shuffle、seed 1234、InfoNCE、全量 tuner 但只训练 `model.norm.weight`、SGD lr `0.01`、三步。模型和数据 SHA 与 worker 记录一致；无网络下载。严格候选从启动到退出启用了 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`。父进程和子进程均记录 start/end `use_cuda=1`、shim marker 存在、fallback 0。

原生 loss 为 `[2.03235769, 2.03235316, 2.03234959]`，严格 shim 为 `[2.03236127, 2.03235745, 2.03235340]`；最大逐步 loss 绝对差 `4.29154e-6`。grad norm 最大相对差 `5.34611e-6`。保存 checkpoint 的 290 个 state 键完全相同；被冻结的 `model.layers.0.self_attn.q_proj.weight` 逐值相等，唯一训练参数 `model.norm.weight` 末态最大绝对差为 0、相对 L2 为 0，两侧相对原模型的更新 L2 均为 `5.85909e-4`。这组 CLI 证据未保存逐参数梯度/逐步权重、完整参数 device 清单或训练批次 token 身份；不能替代直接 EmbeddingTrainer API 报告中更细的 forward/gradient 证据，也未执行恢复。

Slurm 13281 是首个尝试，在原生数据预处理开始前因过长 `TMPDIR` 导致 `OSError: AF_UNIX path too long`，后续 `EOFError`，未进入模型训练。保留原日志于运行键 `20261008-qwen2-embedding-sft-cli-361bf-v1`。13283 改用短 worker 临时目录后，两侧模型训练与 checkpoint 保存完成；尾部两次只读比较作业 13297/13298 分别因比较器指向旧 checkpoint/bootstrap 文件名而失败/修正后通过，没有重跑模型。checkpoint、日志、比较器及结果 JSON 未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-embedding-sft-cli-361bf-r2/`；失败尝试在同目录前缀 `...-v1/`。

13283 shim 三步 runtime 为 `373.018 s`，首步含本轮 JIT 首次编译；原生 runtime 为 `4.293 s`。未做两次预热及至少 10 次同步稳态测量，二者不能作为 L5 性能对比。Slurm 13298 的比较只读已有产物并成功。Slurm 13299 在 cscg-qh17 RTX 4090 上通过 `bash tools/check_repo_layout.sh` 与 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure/test_packaging_structure.py`（8 passed、8 subtests passed）；发布清单检查通过，共 220 份 active Markdown。

## 当前源码基线的 L1 补验

运行键 `20261009-qwen2-embedding-sft-cli-l1-v1`，Slurm 14204 在 cscg-qh04 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`）先运行独立原生 PyTorch，再运行严格 shim。Jittor HEAD 为 `a1a083bf80527dfcfb7850f512fcbdcd68070e40`，ms-swift SHA `88d727951203256baa564c643c651b6f8d90fd7e`；模型 SHA 与上节一致，四行数据 SHA 为 `6b69fd53a87c54615855f0eef31179855c0ea7eb1a99e38b9213108d6cf39d92`。严格 shim 进程全程 `use_cuda=1`，父子进程均有 shim 标记且 `fallback_count=0`。

三步公开 CLI 均完成并保存 checkpoint。训练首批在两侧捕获到完全相同的 `input_ids`、`attention_mask`（均 `[16,28]`）及 labels（`[12]`）；290 个初始参数的名称、shape、dtype、可训练状态和逐值 SHA 完全一致，且均在 CUDA。Qwen2 embedding 输出 `last_hidden_state` 在两侧均为 CUDA FP32 `[16,896]`，无非有限值；最大绝对差 `1.99676e-6`、相对 L2 `4.00853e-6`。首步 loss 为 native `2.03235769`、shim `2.03236127`，绝对差 `3.58e-6`。因此只将该固定模型、数据和公开 CLI 配置的 L1 标为通过；不外推到其他 embedding 模型、pooling 配置或其他 ms-swift 入口。

运行键 `20261009-qwen2-embedding-sft-cli-l2-v1`，Slurm 14216 在同一 RTX 4090、源码 HEAD `1cf2e2893fa979a23bc1deeff3eeca23fb6d15bc`、模型/数据、batch 顺序和 CLI 配置下重新运行原生与严格 shim 三步训练。三步完整 `input_ids`、attention mask 和 labels 逐值一致，shape 分别为 `[16,28]`、`[16,28]`、`[12]`；290 个参数的名称、shape、dtype、requires-grad 和初始哈希一致，全部在 CUDA。唯一 trainable 参数 `model.norm.weight` 的三步 FP32 CUDA 梯度相对 L2 分别为 `9.39e-6`、`1.04e-5`、`1.08e-5`，最大绝对差分别为 `1.34e-7`、`1.36e-7`、`1.35e-7`。每一步更新前后权重两侧均逐值相同；SGD 的两组超参数一致（lr `0.01`、momentum `0`，权重衰减组 `0.1/0.0`），内部状态均为空；末态 290 键 checkpoint 最大绝对差为 0。loss 最大差 `4.29e-6`。所有候选进程为真实 CUDA、`use_cuda=1`、`fallback_count=0`。该运行满足本固定配置 L2 的三步、全可训练参数梯度和更新门槛；不推广到其他 tuner、优化器或 embedding 任务。

运行键 `20261009-qwen2-embedding-sft-cli-l3-train-v1` / Slurm 14221 在同一 RTX 4090 上用公开 `swift sft` CLI 分别执行原生与严格 shim 连续三步，并从各自 step-1 checkpoint 新进程恢复至 step 3；batch size 为 2，四条固定记录关闭 shuffle，因此 step 1 checkpoint 位于 epoch 中途。GPU-worker 只读对拍 14222 确认恢复后的 step 2/3 输入与连续训练逐 tensor 完全相同，且 native/shim 的 step 1–3 输入也一致（CUDA `input_ids`/mask，shape `[8,28]` 或 `[8,27]`，labels `[6]`）。连续与恢复轨迹均到达 global step 3 / epoch 1.5；native step 2/3 loss 完全一致，shim 最大差 `4.77e-7`。同一 runtime 内，连续/恢复的 290 键 step-3 权重完全相同；`optimizer.pt`、`scheduler.pt`、`rng_state.pth` 分别逐字节相同，证明该固定中途游标和后续状态恢复对齐。native 与 strict shim 的连续末态及恢复末态 290 键权重也逐值相同。四种运行的父/子进程均记录 CUDA，shim marker 与模式相符，fallback=0。该结果仅支持此固定 trainer、SGD、数据顺序及中途 checkpoint 配置；不外推到其他 sampler、scheduler 或模型。

14221 日志在数据 map 子进程退出清理时出现 `multiprocess` manager 删除 `.nfs*` 临时文件的 `EBUSY` traceback；Map 随后完成，四条 CLI 返回成功，捕获的输入、checkpoint 与状态文件均完整。该清理告警保留在运行日志，不影响上述恢复轨迹证据。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS（固定配置） | Slurm 14216 保存的 native/shim 初态清单各有 290 个参数，name、shape、dtype、device、requires-grad、SHA 逐项相同；全部 CUDA，唯一可训练参数为 `model.norm.weight`。公开 CLI 日志与训练、optimizer、checkpoint 产物证明本模型、Qwen template、数据、EmbeddingTrainer 和 SGD 均已构造；shim fallback=0。 |
| L1 | PASS（固定配置） | Slurm 14204 在当前源码基线下确认首批输入和 290 个初始参数逐项一致；`last_hidden_state` 为 CUDA FP32 `[16,896]`，最大绝对差 `1.99676e-6`、相对 L2 `4.00853e-6`，首步 loss 差 `3.58e-6`，fallback=0。仅适用于本模型、数据及 CLI 配置。 |
| L2 | PASS（固定配置） | Slurm 14216 的三步输入完全相同；唯一 trainable 参数梯度相对 L2 最大 `1.08e-5`、max abs `1.36e-7`，逐步更新前后权重一致；SGD 参数组相同、momentum=0 且状态均为空，末态 290 键 checkpoint 完全相同，fallback=0。仅适用于该模型、数据和 CLI 配置。 |
| L3 | PASS（固定配置） | Slurm 14221/14222 的新进程 checkpoint-1 中途恢复：后续批输入逐值一致；global step/epoch、loss、290 键权重及每个 runtime 内 optimizer/scheduler/RNG 文件均与连续轨迹相同；strict CUDA、fallback=0。仅适用于本 trainer、SGD 和固定四行数据。 |
| L4 | PASS（固定配置） | 原生与严格 shim 均通过公开 `swift sft` CLI 完成三步训练、checkpoint 保存及新进程中途恢复；L0-L3 对该配置均通过。 |
| L5 | not-run | 未执行真实尺寸预热和至少 10 次同步稳态测量。 |
