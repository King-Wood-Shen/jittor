# Qwen2 序列分类公开 SFT 双卡 NCCL CUDA 对拍

- 状态：保留旧协议下固定配置的三步逐值结论。2026-10-11 的均衡标签双卡 profile 在第六步 backward 触发 Jittor slice overflow；另一组八行固定数据的当前基线双卡 profile 按新协议完成 L0–L5。各结论仅适用于对应配置，不代表所有分类/embedding/reranker/reward 模型或整个 ms-swift。
- 日期：2026-10-08。
- 基线：Jittor `d2ea2dde9635ff7a5f4e9a5b7ecde86667a51546`；本轮最终重新 fetch 的 `origin/2.0-refactor` SHA 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。开工时 HTTPS fetch 与 SSH `ls-remote` 曾分别遇到连接关闭和公钥认证失败，结束前 HTTPS fetch 成功并确认上游 SHA 与缓存相同。
- 范围：公开 `python -m swift.cli.main sft --task_type seq_cls`，Qwen2-0.5B `Qwen2ForSequenceClassification`，双 RTX 4090、FP32/eager、全参数 SGD、固定 8 行数据、每卡 batch 2、三步，NCCL 2.21.5。
- 维护者：ms-swift CUDA 适配。
- 复查条件：本报告的 L0–L5 仅适用于所列固定 profile；扩展到其他分类数据、精度、优化器或 tuner 时须另行验收。

Slurm 13122 先运行原生 PyTorch oracle；三步公开训练、checkpoint、rank 0/1 每步输入/输出及 291 个全参数梯度均已保留。随后 Slurm 13123 运行 strict shim，先串行预热 Jittor/NCCL，再以公开 CLI 完成三步训练。两作业均在 `cscg-qh04`，GPU UUID 为 `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea` 与 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。双 rank 的 `RANK/WORLD_SIZE/LOCAL_RANK` 与 Jittor rank 变量一致，NCCL world size 为 2；候选每个 rank 导入/结束均 `use_cuda=1`、fallback `0`。初始 291 个参数名称、shape、dtype、值哈希相同，全部 trainable 且在 CUDA。基础 checkpoint SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`，数据 SHA256 为 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`。

比较器逐 rank、逐步验证全部捕获输入张量逐元素一致，并比较 CUDA logits/loss。logits 最大绝对差为 `3.10e-6` 至 `1.39e-5`；loss 最大绝对差不超过 `7.63e-6`。rank 1 的 loss 绝对值接近零，因此其相对 L2 不作为误差判断依据。每个 rank 的每一步都比较 291 个参数梯度，最大绝对差依次为 `6.55e-7`、`1.32e-7`、`3.45e-7`，相对 L2 为 `3.64e-6`、`2.21e-6`、`4.92e-6`。最终 checkpoint 的 291 个键/shape/dtype 一致，最大权重绝对差 `7.45e-9`、相对 L2 `5.91e-11`，168 个张量有更新。训练日志两侧 loss 分别为 native `4.12568808 / 3.76338148 / 4.11957359`、shim `4.12568474 / 3.76338148 / 4.11957836`。

GPU worker 对 `optimizer.pt` 做 zip 或 pickle 格式识别后逐字段/张量比较，两边相同；SGD 的 state 为空，参数组相同。`scheduler.pt` 不同：native 包含 `verbose=False`，shim 含 `_is_initial=False`。因此未尝试宣称恢复轨迹等价。原始命令、日志、rank JSON、梯度、NPZ、checkpoint、逐字段状态审计和比较结果未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-seqcls-ddp-nccl-v1/`。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 CLI 构造模型、tokenizer、dataset、tuner、optimizer、trainer；291 参数名称、元数据、值哈希和 CUDA 设备一致；尚缺 tokenizer/template 与完整参数到 optimizer 状态键映射。 |
| L1 | partial | 两 rank 三步输入逐值一致，logits/loss 有限且满足 CUDA 前向容差；没有保存完整模型输出结构与 hidden-state 清单。 |
| L2 | partial | 三步均比较全部 291 个 CUDA 梯度、更新后的全量权重和 optimizer 状态；但前置 L0/L1 仍 partial。 |
| L3 | blocked | 未进行同进程或新进程恢复；scheduler 持久化字段不同，未验证 RNG、数据游标、后续输入与续训轨迹。 |
| L4 | partial | native 与 strict shim 均经公开 `swift sft` CLI 端到端完成三步并产出 checkpoint；由于前置层未完整通过，不升级为 PASS。 |
| L5 | blocked | 前置层未通过，且没有预热后至少 10 次同步稳态双卡性能协议。 |

Slurm 13121 是启动脚本指向缺失 NCCL 目录的 harness 错误，模型未运行。13122 在原生阶段成功后，首次 shim 预热触发 Jittor 一次性 reload guard，故保留并复用原生结果；13123 修正预热顺序后完成候选。首步 JIT 期间共享缓存发生锁等待，rank 逐个编译不同 CUDA 算子，产物持续推进，最终训练正常完成；不把该编译等待计作训练失败。Slurm 13130 的状态审计错误地用 `torch.load` 读取 shim 原始 pickle，修正 zip/pickle 识别后，13131 在 worker 上完成只读状态比较。上述原始记录均保留。

文档收尾：Slurm 13133 完整结构测试暴露环境缺少 cmake、oneDNN 默认启用而导致 CPU matmul 用例失败；Slurm 13135 在 `use_mkl=0` 下对此用例定向复验通过。Slurm 13137 按仓库 Torch 模式命令并设 `use_mkl=0` 通过清单检查、布局检查及 `tests/structure`（`1384 passed, 8 skipped, 1019 subtests passed`）。8 个 skip 由 accelerator/声明依赖及未安装的 pytest-xdist 等检查依赖造成；不额外设置 `JITTOR_TEST_REQUIRE_EXECUTION`。Slurm 13136 使用该额外严格变量时虽所有断言通过，但因报告上述环境 skip 而退出 1，不作为最终门禁结果。`git diff --check` 通过。

## 2026-10-11：新收敛协议下的双卡复验

本节保留上方历史严格逐值结果，只记录更新后的训练收敛协议所需的新运行。运行键 `20261011-qwen2-seqcls-ddp-balanced-convergence-training-v1` / Slurm 16252 在 Jittor HEAD `7bb1562f8c15be6adc68b581c2b006cb616e3e86`、ms-swift SHA `88d727951203256baa564c643c651b6f8d90fd7e` 上执行。仍为公开 `swift sft --task_type seq_cls`、Qwen2-0.5B FP32/eager、全参数 SGD、双 RTX 4090/NCCL、每 rank batch 4、8 步；使用同一组八条固定样本，仅将顺序调整为 `[0,4,1,5,2,6,3,7]`，使标准分布式采样下每 rank 的每批标签均为 2:2。新数据 SHA256 为 `fdf366615e7d9dd82dc6e58843cdaf3f7f46e8164d0df308cb3841705a0c1c01`。GPU UUID 仍为 `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea` 和 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。

独立 native oracle 完成全部 8 步；两 rank 标签平衡，loss 前两步均值分别为 `4.12376833`、`3.76302028`，末两步均值分别降至 `4.10072970`、`3.74212742`。严格 shim 的双 rank 均映射到相应 CUDA ordinal/UUID，NCCL world size 为 2，`use_cuda=1`、`fallback_count=0`；实际完成前 5 步，每步采集到 291 个有限、非空梯度。第 6 步 `loss.backward()` 在 `autograd_api.py::_backward -> jt.core.grad_optional` 触发 `nano_vector.h:41: slice overflow: 93893295758975 0 1`，rank 0 退出并终止另一个 rank。shim 未达到预先固定的 7–8 步末段窗口，没有 checkpoint-8，也没有完成原生/shim 全轨迹比较；前五步观测不作为收敛 PASS。

| 层 | 此次均衡双卡新 profile | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 CLI、模型与 291 参数状态已构造并导出；未完成全状态键/元数据跨实现审计。 |
| L1 | partial | strict shim 已真实执行 CUDA 前向并保存前五步输入/logits；本次未完成这些张量的原生逐项比较。 |
| L2 | failed | native 完成 8 步并通过固定窗口 gate；strict shim 在第 6 步 backward 触发上述 Jittor 内部断言，未完成 8 步窗口及完整比较。 |
| L3 | blocked | L2 未通过；未运行恢复验收。 |
| L4 | 功能未通过 | 两侧都启动了公开 CLI，但 strict shim 训练中断，没有完成公开训练合同。 |
| L5 | blocked | 前置层未通过，未运行稳态性能验收。 |

该异常与既有 `grad_optional`/`NanoVector` slice overflow 诊断签名相同；该根因调查已经达到 Skill 规定的五轮上限，本次只保存这个新数据顺序和双卡配置的失败边界，不启动进一步追踪、不规避 backward 或修改实现。原始日志、状态、梯度和部分 checkpoint 未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-seqcls-ddp-balanced-convergence-training-v1/`。该结果不推翻上方旧协议的三步对拍，也不外推至所有序列分类或 ms-swift 双卡训练。

报告门禁运行键 `20261011-qwen2-seqcls-ddp-balanced-convergence-report-docgate-v1` / Slurm 16253 在 cscg-qh04 完成：`tools/check_repo_layout.sh` 通过；Torch 模式 `tests/structure` 为 `1384 passed, 8 skipped, 1019 subtests passed`（180.80 秒）。8 项 skip 分为 2 accelerator、2 declared、4 other；other 为解释器内未安装 Jittor 的 2 项检查及未安装 pytest-xdist 的 2 项检查。worker 设置 `use_mkl=0`，未触发 oneDNN 构建失败。

### 当前基线八行固定数据 profile：训练收敛协议复核

运行键 `20261011-qwen2-seqcls-ddp-convergence-training-v1` / native Slurm 16243 与 strict-shim `20261011-qwen2-seqcls-ddp-convergence-training-v2` / Slurm 16244 使用当前代码基线 Jittor `d66fece475851205cd6413e2aa341a649042b4bb`、ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。两者都是公开双卡 `swift sft --task_type seq_cls`、Qwen2-0.5B、FP32/eager、全参数 SGD、相同八行固定数据（SHA256 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`）、每 rank batch 4、8 个 optimizer step，保存 step 4/8 checkpoint。native 独立 oracle 先运行；strict shim 使用已预热的双卡 CUDA/NCCL 缓存。native 与候选输入、291 项参数初态逐 rank 对齐。两张 RTX 4090 UUID 与逻辑 ordinal 为 `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`/0、`GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`/1。

只读 GPU-worker 审计键 `20261011-qwen2-seqcls-ddp-convergence-audit-v1` / Slurm 16279 复核两侧全部 8 步原始证据。每个 rank/step 均有 291 个有限、非空 CUDA 梯度，跨 rank 的逐参数梯度哈希一致；两侧均有 8 个真实 optimizer step，step 4/8 checkpoint 存在，末态 291 个参数键中有 245 个相对同一初态发生变化。strict shim 的两个 rank 分别匹配 CUDA pointer、Torch 与 Jittor device ordinal 0/1，NCCL world size 为 2，训练开始/结束 `use_cuda=1`、`fallback_count=0`。固定收敛门槛按两个等大 rank 的每步 loss 求均值，再比较每侧前两步与末两步窗口：native 从 `3.943394729` 降到 `3.921428612`，strict shim 从 `3.943395615` 降到 `3.921425343`，两侧均下降。rank-local 曲线保留为诊断；rank 1 单独的 loss 略升，不是 Skill 规定的整批收敛门槛。

因此，此固定 profile 的 L0、L1、L2 与公开入口 L4 通过；L3 恢复和 L5 性能为 not-run。原采集保存逐参数梯度有限性、范数、最大绝对值和哈希摘要，没有保存梯度数组，故跨实现逐值梯度误差不能从现有证据回算；这项数值差异为诊断信息，不影响新协议下由有限非空梯度、真实更新及两侧收敛共同构成的 L2 门槛。原比较器曾额外要求每个 rank-local loss 窗口都下降，导致其结论无法用于本协议；Slurm 16279 使用不改写原始证据的只读审计重新按已确认协议核验。双卡均衡标签 profile 的真实 backward slice overflow 结果仍保留，不由本 profile 覆盖。历史旧协议严格对拍及其他 profile 的等级均不变；本记录不推广至其他模型、数据、精度、优化器或整个 ms-swift。

只读数值诊断运行键 `20261011-qwen2-seqcls-ddp-convergence-diagnostics-v2` / Slurm 16281 对齐了两端每 rank/step 的 logits shape/dtype/finite 与预测：全部为 FP32 `[4,2]`，argmax 均一致；rank 0 最大绝对差 `2.003e-5`、最大 relative L2 `1.126e-6`，rank 1 分别为 `1.073e-5`、`7.125e-7`。291 个末态权重键全部对齐，172 个张量哈希不同，最大绝对差 `1.490e-8`、整体 relative L2 `9.644e-11`；逐参数梯度摘要的最大 L2 范数差 `2.429e-6`、最大 max-abs 摘要差 `7.377e-7`，每 rank/step 的跨实现梯度哈希均不同。梯度逐值误差仍不能计算，因为未保存原始梯度数组。诊断 v1/job16280 因审计输入根目录多拼一层而在读取前退出；v2/job16281 成功，输出 JSON 中继承模板的 `run_key` 文字仍误写为 v1，实际 job 与输出目录为 v2，该元数据偏差已在 state manifest 明记，不影响其数组计算。训练 v2 的 shell 末尾还记录共享缓存集合变化（新增 fused SGD kernel），所以不宣称该训练阶段缓存稳定；它不影响本次 L0–L2/L4 结论。

本次文档门禁 `20261011-qwen2-seqcls-ddp-convergence-report-docgate-v1` / Slurm 16282 在 cscg-qh04 GPU worker 通过布局门禁（238 active Markdown）及 Torch-mode `tests/structure`（`1384 passed, 8 skipped, 1019 subtests passed`，179.97 秒）。8 项 skip 为 2 accelerator、2 declared、4 other；other 是解释器未安装 Jittor 的 2 项检查及未安装 pytest-xdist 的 2 项检查。worker 的结构测试解释器未找到 nvcc 并继续 CPU 路径；这只是文档/结构门禁证据，不是 CUDA 计算证据。

补入数值诊断后，最终文档门禁 `20261011-qwen2-seqcls-ddp-convergence-report-docgate-v2` / Slurm 16283 再次通过：238 active Markdown，Torch-mode `tests/structure` 为 `1384 passed, 8 skipped, 1019 subtests passed`（180.60 秒），skip 原因与前一轮相同。运行位于 cscg-qh04 GPU worker，结构测试内部仍因解释器未找到 nvcc 走 CPU 路径；本门禁只验证布局、文档和结构测试。

### 当前固定数据 profile：checkpoint L3 同进程与新进程恢复

恢复验收使用运行键 `20261011-qwen2-seqcls-ddp-resume-l3-v4` / Slurm 16376，Jittor HEAD `4a9e62f1949b868de71e4e70b40a06a35c6ef53a`（相对前述训练基线只增加报告/索引提交），ms-swift SHA `88d727951203256baa564c643c651b6f8d90fd7e`。该作业仅运行 strict shim；native 同进程和新进程恢复轨迹取自 v2 原生 oracle 只读证据，没有重跑。配置仍是公开 `swift sft --task_type seq_cls`、Qwen2-0.5B、双 RTX 4090/NCCL、FP32/eager、全参数 SGD、每 rank batch 4、固定八行且不 shuffle 的数据、step 4 checkpoint 后继续至 step 8。v4 shim 使用 rank 0 UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`、rank 1 UUID `GPU-f54b1242-12ae-0c83-66b0-1c08461046ec`；旧 native 轨迹 rank 1 是另一张 RTX 4090（`GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`），因此这里只比较各 runtime 自己的恢复与连续 oracle，不据此给出跨设备性能比较。两端各自的两个 rank 均核验到 CUDA pointer/Torch/Jittor 逻辑 ordinal 一致、NCCL world size 2；shim `use_cuda=1` 且全程 `fallback_count=0`。

两个恢复场景均通过。每 rank 在 step 4 checkpoint 中保存的 291 项模型 state 恢复后逐值相同且位于 CUDA FP32；optimizer 与 scheduler 恢复状态分别匹配该 checkpoint 保存前的完整状态摘要。Python、NumPy、Torch CPU 与 CUDA RNG 的 setter 输入摘要均与 step 4 保存值相同；本轮修正了 v3 观测器漏钩 `set_rng_state_all` 的问题。连续 oracle 与恢复运行的 step 5–8 输入张量逐项相同，证明此固定非 shuffle 数据配置的后续 batch 游标正确。每步均采集 291 项有限梯度和真实 optimizer 更新，global step 从 4 续到 8；按两 rank 等批次均值计算，native 的 step 5–6 到 7–8 loss 窗口均值为 `3.928749380 → 3.921428612`，shim 同进程为 `3.928752124 → 3.921426773`，shim 新进程为 `3.928752601 → 3.921425819`，恢复后各自继续下降。Skill 新协议不要求 native 与 shim 的状态或逐步数值跨运行时相等；这里的通过判据是各自 checkpoint 状态正确恢复、后续批次正确、梯度更新有效、loss 窗口下降及 strict CUDA/零 fallback。

L3 完成后，此固定 profile 的 L0–L5 均通过；其他数据顺序、模型、tuner、精度和训练策略仍为 `not-run` 或沿用各自独立失败记录。此前 L3 harness 运行键 v1/job16284、v2/job16318、v3/job16347 的错误和证据均保留在各自 state；其失败均不作为兼容性失败结论。v1 是 argv hook 安装错误，v2 是 checkpoint 文件散列时序竞争，v3 是 CUDA RNG setter 观测不完整；v4 才完成两种恢复场景。原始日志、checkpoint、事件与比较 JSON 未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-seqcls-ddp-resume-l3-v4/`。


### 当前固定数据 profile：双卡 L5 稳态性能

L5 使用新运行键 `20261011-qwen2-seqcls-ddp-training-l5-v2` / Slurm 16432；基线为 Jittor `635cc578ae5dfd3b6bf911368c30f7a1fc39ca5f`、ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`，模型和数据 SHA256 分别为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b` 与 `6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c`。在同一 cscg-qh04 allocation、同一双 RTX 4090 UUID（rank 0 `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，rank 1 `GPU-d813000b-1bfa-885e-cd61-0de97b365732`）上先运行独立 native oracle，再运行 strict shim。每侧各完成 12 个 optimizer step，前 2 步预热，后 10 步同步计时；全局步延迟取两 rank 较慢值，吞吐汇总两 rank 的 8 samples 与 74 non-padding tokens/step。审计键 `20261011-qwen2-seqcls-ddp-training-l5-artifact-audit-v2` / Slurm 16450 在 GPU worker 对既有产物只读复核：同 rank 的 native/shim 首批及 10 步 sample/token 数相同；两端每 rank 均有 291 个 trainable 参数，分类头 `score.weight` 有真实更新；CUDA pointer ordinal、设备 UUID、NCCL world size/rank 相符，strict shim 两 rank `use_cuda=1`、`fallback_count=0`。shim 预热后的 197 项 JIT `.so` 清单在测量结束时完全相同。

| Runtime | 稳态延迟 mean / median / P95 | 全局吞吐 samples/s | 全局吞吐 tokens/s | 峰值 allocated / reserved（rank 0；rank 1） |
| --- | --- | ---: | ---: | --- |
| PyTorch CUDA | 0.2519 / 0.2518 / 0.2534 s | 31.76 | 293.75 | 7.39 / 7.63 GiB；7.39 / 7.68 GiB |
| JTorch strict CUDA | 0.2979 / 0.2980 / 0.2995 s | 26.85 | 248.38 | 9.20 / 11.48 GiB；9.20 / 11.54 GiB |

JTorch/PyTorch samples/s 比值为 `0.8456`。显存数据是各自 `torch.cuda` allocator 在第 2 步后 reset 的 peak allocated/reserved；两种 allocator 口径不同，不将字节数解释为可直接互换的物理显存占用。该结果只给出此固定双卡 profile 的性能比值，不外推到其他序列分类配置或整个 ms-swift。

L5 v1/job16424 在 native 训练后被错误的“任意最小 bias 必须变化”采集断言中止，未运行 shim；不作兼容结论。v2/job16432 两侧训练与测量完整，最终原比较器因不支持的 walrus 语法退出；只读审计 v1/job16444 又误要求两个 rank 的数据分片相同。修正后的只读审计 v2/job16450 通过并生成上表；三次无效审计/采集路径均保留各自 state，不重跑训练。原始日志、逐 rank JSON、缓存清单和审计 JSON 未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-seqcls-ddp-training-l5-v2/` 与 `20261011-qwen2-seqcls-ddp-training-l5-artifact-audit-v2/`。

最终文档门禁使用运行键 `20261011-qwen2-seqcls-ddp-resume-l3-docgate-v1` / Slurm 16402，在 cscg-qh04 完成：`tools/check_repo_layout.sh` 通过，Torch-mode `tests/structure` 为 `1384 passed, 8 skipped, 1019 subtests passed`（203.69 秒）。8 项 skip 中 2 项为 accelerator、2 项为 declared、4 项为 other（解释器未安装 Jittor 的 2 项检查及未安装 pytest-xdist 的 2 项检查）；结构测试解释器未找到 nvcc 并继续 CPU 路径。本门禁仅验证布局、文档和结构测试，不构成 CUDA 训练或恢复证据。

加入双卡 L5 证据后的文档门禁使用运行键 `20261011-qwen2-seqcls-ddp-l5-report-docgate-v1` / Slurm 16453，在 cscg-qh04 完成：布局门禁通过（238 active Markdown）；Torch-mode `tests/structure` 为 `1384 passed, 8 skipped, 1019 subtests passed`（585.88 秒）。8 项 skip 为 2 accelerator、2 declared、4 other；后 4 项分别是两个子进程解释器未安装 Jittor 的解析检查，以及两个环境未安装 pytest-xdist 的检查。该运行使用独立空缓存，Jittor core 首次 CPU 编译耗时约 303.89 秒；解释器未找到 nvcc，结构测试继续 CPU 路径。本门禁仅验证布局、文档和结构契约，不构成 CUDA 训练或性能证据。
