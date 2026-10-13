# 2026-09-22：ms-swift LoRA 的 Ascend torch shim 验证

**状态（2026-10-13 复核）：完整逐格矩阵见下文。tiny 单卡 LLaMA LoRA 公开 SFT/恢复 L0–L4 通过，L5 未运行；另有 tiny LLaMA 浮点 `inputs_embeds` direct API 梯度与训练更新格分别通过 L0/L1、L0–L2，不能外推到公开 CLI、checkpoint 恢复或其他输入。两卡 tiny IA3 fused AdamW、ORPO 与目标 2.0-refactor DPO 的历史锁定配置公开训练/恢复及逐 rank 设备审计通过 L0–L4，历史严格比较结论原样保留。另有 IA3 与 ORPO 新收敛协议双卡配置分别通过各自 loss 窗口、checkpoint-16 新进程恢复、逐 rank 设备审计与候选零回退门禁，范围仅限各自锁定配置。tiny PEFT DoRA 新协议八步公开 SFT 与 checkpoint 新进程恢复通过 L0–L4；旧严格数值失败保留。tiny LLaMAPro 单 NPU公开 SFT 通过 L0–L2/L4，其原生 checkpoint 新进程恢复因 optimizer 参数组不匹配失败，故 L3 未通过。tiny Swift Adapter 双 NPU训练与恢复通过 L0–L4；单 NPU推理完成 2,461 项严格比较并通过 L0/L1/L4。tiny Mistral Swift Adapter 的已有公开 SFT/推理对拍通过，新增 cache-on、batch=2、top-k=32/top-p=0.9 与 top-p=0.8，以及 top-k=64/top-p=0.8 的四步公开 sampling 锁定格均通过 L0/L1/L4；新 `swift export --merge_lora` 工件两侧 29 键逐 tensor 精确一致，但从导出物公开重载的门禁失败：源为 Swift `ADAPTER` tuner，29 键中 8 个 `adapter_default` 张量未被标准 Mistral loader 消费，因此 native/ACL 输出一致仅覆盖共同加载的 21 个参数，不能算完整 merged-Adapter 通过；cache-off 配置格中公开 CLI 未提供 `--use_cache`，本地 config/generation config 均为 false 但实际 GenerationConfig 为 true，故 cache-off 的 L0/L4 失败且未运行 ACL 候选；相关 Swift 配置修复方案待用户确认。fresh-process checkpoint-2 恢复仍在原生 Swift/Transformers 路径失败，未运行 ACL 恢复候选。DPO 目标 2.0-refactor 锁定配置完成四步训练、checkpoint-3 恢复及 170 项严格对拍。真实尺寸 109.48M BERT-base 公开 SFT 锁定配置通过 L0–L5；10 次同步计时 candidate `172.812 ms/step`、原生 `70.354 ms/step`，吞吐比 `0.407x`。BERT-base 公开推理的 GELU 层内差异复现，L1/L4 失败；历史严格 GELU/更新对拍失败保留。Qwen3 tiny 旧三步严格公开 SFT 的 L2 失败结论原样保留；独立的 32 步新收敛/恢复合同锁定配置通过 L0–L4，不能追溯改判旧矩阵行。Qwen3 public greedy batch=2（4/16 token，另有 cache-on 16 token 与 cache-on near-context prefill=124/128 的 4-token decode；cache-off 新格在原生公开入口确认有效 cache 仍开启，ACL 未运行）、top-k=1、top-k=4/top-k=16 与 top-k=4+top-p=0.8 单 token sampling、top-p=0.8 单步/四步 sampling（含独立 cache-enabled 四步格）、cache-enabled top-k=2/4/8/16 四步 sampling（top-k=2/4/8/16 含 temperature=0.7 独立格；另有 top-k4+top-p=.8、temperature=.7 组合）、near-context 长度 124/128、以及 2+1 不满尾批公开推理锁定配置通过 L0/L1/L4；除上述锁定配置外的一般 sampling 参数与独立 cache-on/off 对照尚未验证；beam=2 strict ACL 在 stopping-criteria 的 `Mod` 失败，简单 `.tolist()` 探针未复现且 core 修复方案待确认。约 1.1B 推理严格比较失败；Swift Prompt direct API 新收敛协议 tiny 单 NPU L0–L3 锁定配置通过；公开 CLI 路由缺失，direct generation 的 native oracle 失败。真实多机资源阻塞，L5 仍未完成的配置保持未运行。所有通过只适用于各自锁定配置；全范围功能面仍未完成。**

各历史运行均绑定其 manifest 中的源基线与 dirty source，不因后续同步而改判。IA3 r3 工作负载的集成代码基线为 `a8dbba3984931a699aa814c31bcec7edf09fcb73`，upstream `2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；此次追补的是同一原始运行产物的 comparator 协议 v2/v3，不重跑模型。2026-10-12 本轮整合前重新 fetch `origin/2.0-refactor` 得 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，该目标 SHA 是当前 HEAD `a79e5706916bbb7da4ddb42e21624514c60bf5f9` 的祖先，无需合并；已保留原有未提交改动。此前报告记录的较早 SHA 仍是对应历史运行的真实基线。维护者：Torch compatibility / ACL backend maintainers。核心初始化、梯度状态语义、依赖版本、后端/驱动或协议变化时重新验证。本结果仍在持续验收；不创建 PR 或合并 PR。

## 验收范围与当前状态

本报告采用[统一验收合同](../../agent/skills/ms-swift-torch-compat/references/verification.md)的飞书唯一刻度：L0 导入/构造，L1 前向/推理，L2 训练/梯度，L3 完整 checkpoint 保存、全新进程恢复并续训，L4 真实公开 ms-swift CLI/launcher 端到端，L5 正确性通过后的稳态性能。后端身份、NPU 驻留、分布式 HCCL、严格零回退、有限值和精确键/形状是每层门禁，不独立占一个等级。

| 轨道 | 当前证据 | 未完成项或阻塞 |
| --- | --- | --- |
| 单 NPU 训练 | 锁定 tiny case 的公开三步对拍、完整 checkpoint fresh-process 精确续训通过，覆盖 L0–L4 适用门禁；另有 tiny LLaMA float `inputs_embeds` direct API 梯度与训练更新格通过 L0–L2 | L5 真实尺寸稳态性能未完成；direct API 结果不推广到公开 Swift CLI、checkpoint 恢复或其它模型 |
| 单机多 NPU 训练 | IA3/fused AdamW、ORPO 与目标 2.0-refactor DPO 锁定配置的公开双 rank 训练、严格 ACL、逐 rank 设备驻留及完整恢复对拍通过 L0–L4 | 各配置 L5 真实尺寸稳态性能未运行；其它模型/tuner 未覆盖；真实多机仍资源阻塞 |
| 真实多机多 NPU 训练 | `resource-blocked`：2026-10-08 21:24 CST 的 Slurm `sinfo -N -p npu` 仅列出实际 NPU 节点 `cscg-hw01`（`gpu:8`）；没有第二个 `cscg-hw00` 调度节点 | 需要至少两个真实主机且每主机至少两张 NPU；不能用单机多进程或主机别名代替。解除条件是调度器提供第二台实际 NPU 主机并能分配到两台各至少两卡 |
| 单 NPU 推理 | tiny Qwen3 greedy、top-k=1 与 top-k=4 单 token sampling、near-context（实际 prefill 124/128）与 partial-tail，tiny Swift Adapter/IA3，以及 tiny Mistral cache-on top-k32/top-p=.9、top-k32/top-p=.8 与 top-k64/top-p=.8 多步 sampling 锁定配置 L0/L1/L4 通过；Qwen3 与 Mistral 的 cache-off public infer 配置合同均失败（有效 GenerationConfig.use_cache=true，未运行 ACL 候选）；约 1.1B 与真实尺寸 BERT-base 公开入口严格比较失败；Qwen3 `swift export --merge_lora` 新格的原生/ACL safetensors 产物精确一致且公开重载推理通过 L0/L1/L4，greedy/logits 最大差为 0；Mistral export 工件 29 键虽逐 tensor 精确一致，公开重载门禁却发现 8 个 Adapter 张量被普通 Mistral loader 忽略，故不通过完整 merged-Adapter 验收；两者有效 GenerationConfig 均为 cache-on，性能未测 | L5 稳态性能未运行；训练 L2/L3 不适用；Mistral 参数设备分布见该行证据 |

### 按锁定工作负载逐格记录的 L0–L5 矩阵（更新至 2026-10-13）

格子状态只使用验收合同规定的 `passed`、`failed`、`not-run`、`resource-blocked`、`not-applicable`。只有同一行锁定配置及其原始运行证据支持的格子才标为通过；其它模型、拓扑或依赖不从该行外推。

| 轨道 × 锁定工作负载 | L0 | L1 | L2 | L3 | L4 | L5 | 证据与首个未闭合点 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 单 NPU 训练：tiny LLaMA、Swift q/v LoRA、FP32、固定整数 token，公开 SFT | passed | passed | passed | passed | passed | not-run | 第 292、349 节的三步轨迹与全新进程恢复；本行输入梯度不适用，真实尺寸稳态性能未验证；浮点 `inputs_embeds` 见紧随的独立 direct API 行。 |
| 单 NPU API：tiny LLaMA、PEFT q/v LoRA rank 2、FP32、浮点 `inputs_embeds.requires_grad` | passed | passed | not-run | not-run | not-run | not-run | 新键 `llama-lora-float-embeds-grad-20261012-r3`：native oracle job 2582（原生结果从同输入/初态的 r1 复用）→ ACL candidate job 2585 → worker 比较 job 2586；前向、可反传 loss、输入梯度及 8 组 LoRA 梯度均有限且非零，输入 `[1,8,64]` FP32；计算与梯度均在 `npu:0`，候选 ACL `backend_fallback=error`、起止 fallback=0。logits 完全相同；输入梯度最大绝对差 `1.1921e-7`、loss `4.7684e-7`，仅作诊断，不设跨实现浮点阈值。此 direct API 不代表 Swift 公开 CLI 或训练更新。r1 job 2583 在冷编译 Jittor 核心期间主动停止（未加载模型）；r2 job 2584 因探针误调 shim 不支持的 `torch.npu.get_device_name` 在模型加载前失败。两项均为环境准备/探针 harness，不是适配失败。 |
| 单 NPU API：tiny LLaMA、PEFT q/v LoRA rank 2、FP32、浮点 `inputs_embeds.requires_grad`，16 步 AdamW 更新 | passed | passed | passed | not-run | not-run | not-run | 新键 `llama-lora-float-embeds-train-20261012-r2`：bootstrap job 2588 → worker 预检 2589 → native oracle 2590 → strict ACL candidate 2591 → worker 对照 2592。实际本地模型目录名 `base`，模型/输入/初态清单在 worker 预检可读且未下载；ACL 侧 core cache SHA256 `7c2d5edd…3294bc`，`backend_fallback=error` 且起止回退均为 0。Native/candidate 的 16 步参数、输入、梯度、optimizer 状态均在 `npu:0`；两侧每步 loss 窗口均下降：native `4.887218356→4.865373850`，candidate `4.887218236→4.865373611`。worker 只读对照核验 48 个初态键精确、1008 项结构/键检查通过；其余 1000 个跨实现浮点量仅诊断，不设容差门槛。仅覆盖此 tiny LLaMA direct API、FP32输入梯度与该 AdamW 锁定配置，不覆盖公开 Swift CLI、checkpoint 恢复、生成或 L5。r1/job 2587 在模型加载前因旧 Jittor cache build stamp 被 `bootstrap --check` 拒绝，归类 harness/cache 预检失败；本次显式 bootstrap 后 no-build worker 预检通过。原始证据及脚本未版本化，位于 `$TASK_STATE/runs/llama-lora-float-embeds-train-20261012-r2/`。 |
| 单 NPU API：tiny LLaMA、PEFT q/v LoRA rank 2、FP32、浮点 `inputs_embeds.requires_grad`，8 步 AdamW 并从 step 4 新进程恢复至 step 8 | passed | passed | passed | passed | not-run | not-run | 新键 `llama-lora-float-embeds-resume-20261012-r1` 及只读新协议审计 `llama-lora-float-embeds-resume-audit-20261012-r1`：worker 预检 2593 → native continuous/resume 2594/2595 → strict ACL candidate continuous/resume 2596/2597 → 新协议审计 2599。旧 comparator 2598 的同运行时逐位 float trajectory 断言错误，退出 1；保留原始失败，不作兼容性失败判定，也不复用旧键重跑。审计 2599 在 worker 上确认两侧 checkpoint step=4、global step=4、模型 29 个 state keys、8 个 optimizer parameter states/24 个 slots、共 101,704 个 checkpoint 值；continuous 与恢复的 step 5 输入和恢复前状态相同，optimizer 从 step 5 续接。初始模型/Adapter 40 个键、100,704 个值精确相同，恢复状态各自对应保存 checkpoint；同运行时浮点轨迹及跨实现浮点差异按新协议只记诊断，不施加位级或数值阈值。Candidate ACL fallback policy 为 `error`，连续与恢复窗口起止 fallback 均为 0。实际模型目录名 `base`，模型/输入/初态清单在 worker 预检可读且关闭下载。自定义 NPZ/JSON checkpoint 只证明该 direct API 恢复，不代表 ms-swift checkpoint 兼容；L4/L5 未运行。原始证据位于 `$TASK_STATE/runs/llama-lora-float-embeds-resume-20261012-r1/` 与 `.../llama-lora-float-embeds-resume-audit-20261012-r1/`。 |
| 单 NPU 训练：BERT sequence classification、Swift Adapter、FP32、AdamW `eps=1e-6`，锁定 direct API 三步及 checkpoint-2 fresh-process 恢复 | passed | passed | passed | passed | not-run | not-run | 第 591 节：native job 2167 → strict ACL candidate job 2168；95 状态键/105,206 值的内部恢复精确校验通过，完整参数/buffer、输入、forward、gradient、optimizer 及更新对拍通过；最大更新差 `1.313e-7`，候选 ACL error policy、fallback=0，设备驻留通过。不是公开 Swift CLI，故 L4 未运行；L5 未运行。`eps=1e-8` 旧配置的更新失败保留，不能据此推广其它 epsilon、encoder 或公开入口。2026-10-12 的公开 `swift sft` native 边界预检 jobs 2577/2578 在训练前失败（详见后文）；没有 Jittor candidate，不提升 L4。 |
| 单 NPU 训练：tiny Mistral、Swift Adapter、FP32、AdamW `eps=1e-6`，公开 SFT 三步、checkpoint-2 fresh-process resume | passed | passed | passed | failed | failed | not-run | 既有键 `mistral-adapter-public-eps1e6-20261006` 的 native/strict ACL 公开三步训练及推理对拍通过；2026-10-12 新键 `mistral-adapter-public-resume-20261012-r1/r2/r3` 首次验证恢复。r1/r2 分别是 callback 环境变量缺失、launcher 预建 evidence 目录冲突的 harness 失败；r3 worker 前置通过并读取实际 `checkpoint-2`（global_step=2）、完整 optimizer/scheduler/RNG/Adapter 状态后，公开 CLI native resume 在 Transformers `load_sharded_checkpoint` 因 Adapter checkpoint 中无 `model.safetensors.index.json` 失败。候选恢复未运行；详见本节末尾。 |
| 单 NPU 训练：tiny BERT sequence classification、公开 `swift sft` full finetuning、FP32、AdamW `eps=1e-6`、3 步与 checkpoint-2 新进程恢复 | passed | passed | failed | failed | failed | not-run | 运行键 `bert-seqcls-cli-world1-20261010-r9/r10`，native job 2215 → strict ACL job 2216 → 修正后的比较器 2217；5,917 项检查中 8 项失败。初始权重/buffer、batch 精确，forward 最大差 `4.47e-8`、gradient `1.87e-7`、optimizer `1.46e-8` 均通过原阈值；首步 AdamW 更新最大差 `1.556e-6`、最终参数最大差 `2.210e-6` 超预锁容差，差异集中在 `bert.encoder.layer.0.intermediate.dense.weight`。恢复初始权重与 checkpoint-2 精确、运行内恢复轨迹精确。候选 ACL strict error、零 fallback、计算张量驻留 NPU；native fallback counter unavailable/unknown。公开 CLI 实际完成，但严格数值对拍失败，不提升 L2–L4。L5 未运行；不调宽阈值。 |
| 单 NPU 训练：tiny BERT sequence classification、公开 `swift sft` full finetuning、FP32 fused AdamW `eps=1e-6`、六条固定样本同批训练 12 步（新收敛协议） | passed | passed | passed | not-run | passed | not-run | 新键 `bert-seqcls-convergence-world1-20261010-r2`，native oracle job 2345 → strict ACL candidate/comparator job 2346；每侧 step 1–3 至 10–12 均值 loss 分别从 `0.69315342` 降至 `0.69309076`，两侧初始参数/buffers与每步 batch 精确相同。单步 forward、gradient、optimizer/参数更新跨实现最大差分别为 `4.13e-7`、`4.47e-8`、`1.70e-7`，仅作诊断。每侧 3,571 条张量设备 metadata 均为 `npu:0`；candidate ACL `backend_fallback=error` 且起止计数为 0，native fallback counter unavailable/unknown。实际入口为公开 `swift sft`；仅支持此新锁定数据/批次/学习率配置。r1 的 batch=1 收敛窗口失败（`0.69495217` → `0.73706774`）及旧 r9/r10 严格数值失败保持历史结论；本行未做 checkpoint 恢复。L5 未运行。 |
| 单 NPU 训练：tiny BERT sequence classification、公开 `swift sft` full finetuning、FP32 fused AdamW `eps=1e-6`、六条样本以 batch=2 分批训练 12 步并从 checkpoint-4 新进程续训（新协议） | passed | passed | passed | passed | passed | not-run | 新键 r4/r5（state manifest 留存）；native oracle job 2349 的训练/恢复产物经 worker 只读门禁 2351 复核通过，strict ACL candidate job 2352、跨实现初始权重/批次只读比较 job 2353 通过。两侧 loss 均值 step 1–3 至 10–12 从 `0.69315958` 降至 `0.69307745`；初始全参数/buffer 与 12 个实际 batch 精确相同。checkpoint-4 恢复前后 166 组模型/buffer/optimizer 张量逐项相同，optimizer groups、scheduler、epoch/global step、checkpoint 文件及 RNG 续接轨迹正确；恢复下一批不同于 step 4 且匹配连续 step 5，step 5–12 同运行时 forward/gradient/optimizer/update 轨迹精确续接。最大跨实现诊断差：logits `3.055e-7`、gradient `7.078e-8`、optimizer state `1.357e-8`、更新后参数 `1.304e-7`，不设跨实现训练误差阈值。候选所有审计张量位于 NPU，ACL strict `error` 且 fallback=0；native fallback counter unavailable/unknown。L5 未运行。r1/r2/r3 的回调路径、过严批次断言和只读重检 symlink harness 失败保留；不是框架兼容性失败，详见 state manifests。 |
| 单 NPU 训练：真实尺寸 BERT-base sequence classification（109.48M 全参）、公开 `swift sft`、FP32 fused AdamW `eps=1e-6`、batch=2、max_length=128、固定六样本训练 12 步并从 checkpoint-4 新进程续训（新收敛协议） | passed | passed | passed | passed | passed | passed | 运行键 `bert-base-real-size-seqcls-world1-20261011-r2`：native torch_npu oracle job 2356 → strict Jittor ACL candidate job 2360 → 只读完整比较 job 2363（`...-r2-paircheck-v3`）。两侧公开 CLI 12 步连续训练及 step 4 保存/新进程续训至 step 12 均完成；native 初段/末段窗口均值 loss `0.84811014 → 0.68229463`，candidate `0.72807539 → 0.60170611`。两侧初始全参数/buffer 与每步 batch 精确一致；各自恢复轨迹、下一批数据、global step、模型/optimizer/scheduler/RNG/trainer checkpoint 及 806 个恢复状态数组通过。全部参数、计算 batch、forward、梯度、optimizer state 在 NPU；candidate runtime/provider 为 Jittor shim/ACL，`backend_fallback=error`，连续与恢复观察窗口 fallback 均为 0；native 通用 fallback counter unavailable/unknown。跨运行时诊断差完整保留（最大 logits 差 `0.7260`、梯度差 `0.13237`、optimizer `exp_avg` 差 `0.02189`），按新协议不以跨实现浮点差作为训练失败阈值；不改变 tiny BERT GELU 严格对拍失败。此 L5 仅使用新运行键 `bert-base-real-size-l5-world1-20261011-r1`（native job 2364 → ACL candidate job 2365），两侧均在真实 NPU worker 对真实 109.48M 参数公开训练执行 3 步 warmup 后测 10 个 optimizer steps；每步开始与 optimizer 更新完成后显式同步设备。平均/中位延迟：native `70.354/70.093 ms`，candidate `172.812/173.011 ms`；nearest-rank p95 均为 `73.118/174.968 ms`，范围分别 `67.764–73.118/170.602–174.968 ms`。计时样本为 20 个有效样本、440 个有效 token；吞吐 native `28.428 sample/s, 625.410 token/s`，candidate `11.573 sample/s, 254.612 token/s`；candidate/native 延迟比 `2.456x`、吞吐比 `0.407x`。Swift Trainer 汇报的 memory(GiB) 均为 `1.77`；精确 allocator 同步边界 live/reserved 峰值不可观测。candidate 参数驻留 `npu:0`、Jittor shim/ACL、`backend_fallback=error` 且全程 fallback `0`；native 参数驻留 `npu:0`，fallback 计数 unavailable/unknown。所有历史严格对拍失败、重复提交的 jobs 2357/2358、无效依赖 2359 与比较器 harness 失败 2361/2362 均保留原记录；不作追溯重判。本行只适用该锁定公开 CLI/model/config。
| 单 NPU 训练：tiny BERT sequence classification、公开 `swift sft` full finetuning、FP32、AdamW `eps=1e-5`、3 步与 checkpoint-2 新进程恢复 | passed | passed | passed | passed | passed | not-run | native oracle job 2221 → strict ACL candidate job 2222 → worker comparator job 2223；5,917 项检查、0 失败。数据与初始模型 SHA256 两侧一致；初始参数/buffers、batch、恢复初始权重与 checkpoint-2 精确。最大差：forward `5.96e-8`、gradient `1.76e-7`、optimizer state `1.46e-8`、updates `4.69e-7`、updated parameters `8.55e-7`，均低于原锁定容差。两侧公开 CLI 连续训练与 fresh-process resume 完成；候选 ACL strict error、全阶段 fallback=0，参数/实际 batch/forward/gradient/optimizer state 实测驻留 NPU；native fallback counter unavailable/unknown。L5 未运行；仅限此锁定模型、入口和超参数。 |
| 单 NPU 训练：tiny BERT sequence classification、公开 `swift sft` 单步 GELU 定位、FP32、dropout=0、eager attention、AdamW `eps=1e-5` | passed | failed | failed | not-run | failed | not-run | 运行键 `bert-base-eager-layertrace-public-cli-20261010-r4`，native 2247 → strict ACL 2248 → worker comparator 2249。初始 203 项精确；前置 attention 输出及 `layer.0.intermediate.dense` 输出通过，最早失败在 `layer.0.intermediate` 激活输出，max abs `4.737e-4`，阈值 `atol=1e-6, rtol=1e-5` 未调整；logits max abs `1.094e-3`。随后 GELU 固定输入算子探针 `bert-gelu-isolation-20261010-r2`：native 2251 先于 strict ACL candidate 2252，输入 SHA256 两侧一致且输入/输出均驻留 `npu:0`；候选与 torch_npu GELU 的 max abs `4.735e-4`、RMSE `2.582e-4`。此算子探针未采集候选 fallback 计数，仅保留为定位证据，不能单独升级等级或归因实现根因；r1 启动标记错误且在计算前退出，原样记录于 run 目录。无完整恢复，L3 未运行；L5 未运行。 |
| 单 NPU 训练：Qwen3 tiny、Swift LoRA、FP32、bool-mask SDPA、公开 SFT | passed | passed | failed | not-run | failed | not-run | 第 404 节：第二次 AdamW 更新超出预置容差；L3 未进入，L4 端到端严格验收失败。 |
| 单 NPU 推理：Qwen3 tiny、原生 checkpoint adapter、greedy 4 token、公开 infer | passed | passed | not-applicable | not-applicable | passed | not-run | 第 404 节：加载状态、logits 与 token ID 精确相等；tiny 计时不满足 L5。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、greedy 4 token、实际 batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-greedy-world1-20261012-r2`：native 2627 → native gate 2628 → strict ACL candidate 2629 → worker comparator 2630；worker 实测两个 batch、每批 2 行，8 次前向/19 个 NPZ。4 条提示、输入、加载参数、greedy token/解码结果精确一致，logits 最大差 0；ACL `fallback_policy=error`、结束计数 0。L5 未运行。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、greedy 16 token、实际 batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-greedy16-world1-20261012-r1`：worker 预检 2631 → native 2632 → native gate 2633 → strict ACL candidate 2634 → worker comparator 2635；所有阶段退出码 0。4 条提示、两个 batch、每批 2 行、64 个生成 token；67 个快照含 32 次前向。输入、41 个加载参数、greedy token/解码结果精确，logits 最大差 0；ACL `backend_fallback=error`、结束计数 0。L5 未运行。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、本地 cache-enabled 配置、greedy 16 token、实际 batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-cacheon-greedy16-world1-20261013-r2`：worker 预检 2984 → native oracle 2985 → worker native gate 2986 → strict ACL candidate 2987 → worker comparator 2988，均退出 0。模型/Adapter 实际目录 `cache-on-base` / `checkpoint-3`，两个本地权重/tokenizer 与配置清单可读，无下载；两条固定提示形成一个真实 batch=2 请求、生成 32 token。Native gate 观察到 cache=true，16 次 NPU forward 输入长度为 prefill 15 后 15 个单 token；候选与原生的 16 组输入/logits 比较通过，max abs logits 差 0、greedy token 与响应精确。Candidate 为 ACL、`backend_fallback=error`、结束 fallback=0；L5 未运行。cache 由本地 config 副本启用，不代表 CLI 有 `--use_cache` 或 cache on/off 性能对照。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、cache-enabled near-context prefill 124/128、greedy 4 token、实际 batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-cacheon-nearctx-greedy4-world1-20261013-r2`：预检 job 2991 → native oracle 2992 → worker native gate 2993 → strict ACL candidate 2994 → worker comparator 2995，全部退出 0。实际模型目录 `cache-on-base`、Adapter `checkpoint-3`，raw prompt 长度 `[115,115]`，模型和 generation config 的 `use_cache=true`，未下载权重。原生 gate 观测一个 batch=2 请求的四次输入长度 `[124,1,1,1]`，在模型 128 位置上限附近进入 cache decode。candidate ACL 严格 `backend_fallback=error`、fallback=0；输入、加载参数、greedy token/响应和四组 logits 精确，最大 logits 差 0。L0/L1/L4 通过，L5 未运行；仅证明此锁定组合，不代表其它近限长度或 cache on/off 性能对照。r1/job 2990 的预检虽核对了 cache-on fixture，但遗漏检查公开命令的 `--model` 路径（实际仍指向旧 cache-off fixture）；因此没有启动模型阶段，作为 harness-only 失败保留。r2 在 worker 预检中锁定并逐字验证 native/candidate 两条命令的 cache-on model 路径后再运行。原始运行证据位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-nearctx-greedy4-world1-20261013-r1/` 与 `...-r2/`。本格基线 HEAD `e318880aa9ad981bf579ef287ad9bbd26d0da7a1`；目标 SHA `4896af611650d1436302c06b0653ac61726fd10c` 为其祖先；dirty source diff SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、本地模型/generation config cache-off、near-context 115/128、greedy 4 token、实际 batch=2 | passed | failed | not-applicable | not-applicable | failed | not-run | 新键 `qwen3-public-infer-batch2-cacheoff-nearctx-greedy4-world1-20261013-r5`：worker 预检 job 3000 通过，确认模型目录 `base`、Adapter `checkpoint-3`、输入长度 `[115,115]`、清单可读且无下载；原生公开 infer job 3001 在 observer 中止于实际 `GenerationConfig.use_cache=true`，与本地 config/generation config 的 false 冲突。未启动 ACL candidate，故这是公开配置合同失败而非适配失败；L1/L4 失败，L2/L3 不适用，L5 未运行。早期 r1（设置阶段 TypeError）、r2/job 2996（预检 run-root 错）、r3/job 2997（CLI 仍选 cache-on 模型）、r4/job 2998/2999（预建 observation 目录冲突，未进入推理）作为 harness/setup 尝试保留。r5 证据：`$TASK_STATE/runs/qwen3-public-infer-batch2-cacheoff-nearctx-greedy4-world1-20261013-r5/`。修复需要 Swift generation 配置路由继承本地模型配置或提供显式公开开关；属于上游/Swift 源码改动，待用户确认后再实施。 |
| 单 NPU 导出：Qwen3 tiny、公开 `swift export --merge_lora`、FP32/safetensors、单文件 64 MB 分片上限 | passed | not-applicable | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-export-merge-lora-world1-20261013-r5`：worker 预检 3008 → native export 3009 → native artifact gate 3010 → strict ACL export 3011 → candidate artifact gate 3012 → worker comparator 3013，全部通过且检查器均在 `srun` worker。复用本地已验证 `base` 与 `checkpoint-3`，离线无下载；两侧 runtime 分别为 torch_npu/Ascend910B3 与严格 ACL/`backend_fallback=error`（候选结束 fallback=0）。输出均为 `model.safetensors` 单文件、25 个 tensor keys，配置一致，键/形状/dtype/有限值门禁通过，逐 tensor 数值比较 `atol=1e-5, rtol=1e-4` 且最大绝对差 0；Adapter 工件已合并/移除。该结果仅证明锁定公开导出工件路径的可用性与序列化 parity，不证明导出物重新加载、推理、参数设备驻留或其他模型/adapter。前置 r1/job 3002（符号链接路径断言）、r2/job 3003（嵌套 `export --help` 返回 87）、r3/jobs 3004/3005（shell 输出引号错误）、r4/jobs 3006/3007（Qwen3 template 参数歧义）均未生成导出工件，保留为 harness/config 失败。原始证据位于 `$TASK_STATE/runs/qwen3-public-export-merge-lora-world1-20261013-r5/`。 |
| 单 NPU 推理：Qwen3 tiny、从 `swift export --merge_lora` merged output 重载、公开 `swift infer`、near-context 115/128、greedy 4 token、batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-export-reload-infer-world1-20261013-r5`：预检 job 3018 → native public infer 3019 → worker native gate 3020 → strict ACL candidate 3021 → worker comparator 3022，均通过。实际模型路径分别为已验证导出键的 `native/output` 与 `candidate/output`，目录名均 `output`，没有加载 Adapter/下载权重；输入和初态清单哈希由 worker 核对。两条 prompt、batch=2、有效配置的 cache=true；两侧四次输入长度 `[124,1,1,1]`，输出/greedy token、加载参数与输入精确，logits 最大差 0；ACL `backend_fallback=error`、fallback=0，计算/参数在 NPU。原导出目录文件的 generation config 为 false，但 Swift 公开 infer 有效 GenerationConfig 又变为 true；此格验收的是 merged 导出物重载推理，不是 cache-off。性能 L5 未运行。r1/job 3014 清单哈希过期、r2/job 3015 未复制导出 SHA 清单、r3 仅准备阶段失败未提交、r4/job 3017 observer 拒绝预建目录，均在模型加载/推理前停止，作为 harness 失败保留；r5 使用新键完整通过。原始证据位于 `$TASK_STATE/runs/qwen3-public-export-reload-infer-world1-20261013-r5/`。 |
| 单 NPU 导出：Mistral tiny、公开 `swift export --merge_lora`、FP32/safetensors、单文件 64 MB 分片上限 | passed | not-applicable | not-applicable | not-applicable | passed | not-run | 新键 `mistral-public-export-merge-lora-world1-20261013-r1`：srun worker 预检 3023 → native export 3024 → native artifact gate 3025 → strict ACL export 3026 → candidate artifact gate 3027 → worker comparator 3028，全部通过。复用本地已校验 `base` 与 `checkpoint-3`，输入/初态清单可读，离线无下载；两侧输出均为单一 `model.safetensors`、29 个 tensor keys，完整配置相同、逐 tensor `array_equal` 精确、最大绝对差 0，Adapter 工件已移除。candidate 使用 ACL、`backend_fallback=error`，进程结束 fallback=0；native 为 torch_npu/Ascend910B3。只支持此 Mistral fixture 的公开导出工件路径与序列化 parity，不证明 merged 导出物重新加载/推理、通用设备驻留或其它模型/Adapter；L5 未运行。原始日志、脚本、清单和产物位于 `$TASK_STATE/runs/mistral-public-export-merge-lora-world1-20261013-r1/`；基线 HEAD `fb4e0ebd06d512e1a57f98de97a6d125c32bc074`，source dirty diff SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。 |
| 单 NPU 推理：Mistral tiny、从 `swift export --merge_lora` 目录公开重载、greedy 4 token、batch=2 | failed | failed（两侧仅共同加载的 21 个参数上数值一致；完整模型语义未成立） | not-applicable | not-applicable | failed | not-run | 父键 `mistral-public-export-merge-lora-world1-20261013-r1` 的 `native/output` 与 `candidate/output` 均为目录名 `output`，各含 29 个相同 safetensors keys；只读 worker 审计键 `mistral-export-reload-weight-coverage-audit-world1-20261013-r3` job 3044 确认源 `adapter_config.json` 的 `swift_type=ADAPTER`，导出 `args.json` 的 `tuner_type=adapter`，父 manifest 记录命令使用 `merge_lora=true`。标准 `MistralForCausalLM` loader 报告并忽略 8 个 `model.layers.{0,1}.mlp.adapter_default.linear{1,2}.{weight,bias}` 键；源 adapter state 同样包含这 8 键，故导出未满足完整合并语义，L0/L1/L4 均不通过。公开 native 与 strict ACL infer 虽成功，但其已加载参数仅 21 个且精确相同；输入快照相同，4 次匹配 logits `atol=1e-5, rtol=1e-4` 通过且最大差 0，greedy tokens/responses 精确相同，candidate `backend_fallback=error`、fallback=0。有效 `GenerationConfig.use_cache=true`，不是 cache-off 测试。此运行的比较器 job3040 原退出 0，但 stdout 被追加到自身写入的 `comparison.json`，导致文件整体不是合法 JSON；job3044 从快照和首行摘要重算了对照，未把格式损坏误作模型差异。r1/job3033 因 native preflight harness 漏写 JSON 停止；r2/job3035 的 observer 将普通 HF model 错认为 Adapter wrapper，在首个 forward 前停止；r3 的预检 job3036、native oracle 3037、native gate 3038、strict ACL candidate 3039 与 comparator 3040 均退出 0。只读审计 r1/job3041 在解析报告前失败；r2/job3042 已通过权重/推理断言但报告字段取错；r3/job3044 完成修正审计。未重下已验证权重。原始证据位于 `$TASK_STATE/runs/mistral-public-export-reload-infer-world1-20261013-r3/` 与 `$TASK_STATE/runs/mistral-export-reload-weight-coverage-audit-world1-20261013-r3/`；基线 HEAD `fb4e0ebd06d512e1a57f98de97a6d125c32bc074`。修复/复测 Swift ADAPTER 与 `--merge_lora` 的 tuner 路由需另开新键；此前历史严格对拍结论不追溯改判。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、seeded sampling top-k=1、16 token、实际 batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-topk1-sample-world1-20261012-r1`：worker 预检 2636 → native 2637 → native gate 2638 → strict ACL candidate 2639 → worker comparator 2640；所有阶段退出码 0。四条提示、两个 batch、每批 2 行、64 个生成 token；67 快照含 32 次前向。输入/加载参数精确，top-k=1 sample token 与响应精确，logits 最大差 0；ACL `backend_fallback=error`、计数 0。top-k=1 只证明 sampling 调用分支且每步只有一个可能 token，不代表通用 RNG 对拍。L5 未运行。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、seeded top-k=4 sampling、每 prompt 1 token、实际 batch=2 | passed | passed（prefill logits 精确；两侧 sample token 均在各自 top-4） | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-topk4-sample-step1-world1-20261012-r1`：worker 预检 2652 → native 2653 → native gate 2654 → strict ACL candidate 2655 → worker comparator 2656；全部退出码 0。四条提示、两个 batch 各 2 行；model=`base`、adapter=`checkpoint-3`，权重未下载。初始参数与 prefill 输入精确，logits 最大差 0；两侧 token 恰好均为 `[53,84,79,46]`，但验收只要求各自落在对应 top-4，不据单次抽样宣称跨 runtime RNG 一致。Candidate ACL `fallback_policy=error`、fallback=0；所有检查器均在 srun worker。该锁定的一步 sampling 入口 L0/L1/L4 通过；多步采样分布、top-p、cache 变体与 L5 未运行。原始数据位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topk4-sample-step1-world1-20261012-r1/`。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、本地模型配置 cache-enabled、top-p=0.8 多步 sampling、4 token、实际 batch=2 | passed | passed（prefill 精确，匹配 decode logits 在容差内；各侧 token 经本侧 nucleus 验证） | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-cacheon-topp08-sample-multistep4-world1-20261012-r2`：worker 预检 2700 → native 2701 → worker native gate 2702 → strict ACL candidate 2703 → worker comparator 2704，均退出 0。4 条 prompt 分两个 batch=2 请求，每条最多生成 4 token；`GenerationConfig.use_cache=true`，观测 decode 输入 `[prefill,1,1,1]`。两侧初始参数与 prefill IDs 精确，prefill logits 最大差 0，匹配 decode logits 的 `atol=1e-5, rtol=1e-4` 比较通过；两侧 sampled tokens 恰好相同，但不以此宣称 RNG 等价；各侧每个 token 均通过本侧 top-p nucleus 与下一步消费检查。候选 ACL `backend_fallback=error`、fallback=0；权重未下载。cache 通过只修改本地 config 副本，不代表公开 CLI 有 `--use_cache`，也不是 cache on/off 配对比较；L5 未运行。原始证据未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topp08-sample-multistep4-world1-20261012-r2/`。
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、cache-enabled seeded top-k=2 四步 sampling、实际 batch=2 | passed | passed（prefill 精确；匹配 decode logits 在容差内；各侧 token 属于本侧 top-2） | not-applicable | not-applicable | passed | not-run | 模型运行键 `qwen3-public-infer-batch2-cacheon-topk2-sample-multistep4-world1-20261012-r2`：worker 预检 2717 → native 2718 → worker 原生门禁 2719 → strict ACL candidate 2720；比较器初版 2721 因复制的 top-k=4/top-p=0.8 配置断言失败（harness-only），只读修正版键 `qwen3-topk2-sample-multistep4-comparator-fix1-20261012-r1` job 2722 通过。四条 prompt 分两个 batch=2 请求、每条最多 4 token，缓存 decode 输入为 `[prefill,1,1,1]`；实际模型目录 `cache-on-base-topk4`、adapter `checkpoint-3`，输入/初态清单可读且未下载权重。两侧初始参数与 prefill 输入精确，prefill logits 最大差 0，六个匹配 decode 步 logits 在 `atol=1e-5, rtol=1e-4` 内；每侧 sampled token 均属于自身 top-2 并被下一步消费。Candidate ACL `backend_fallback=error`、fallback=0。一次观测到两侧 token 序列相同不构成 RNG 等价结论；L5 未运行。r1/job 2716 的复制 sbatch 错留旧 `R` 路径，只执行了旧 top-k=4 预检、没有模型推理；该尝试和 2721 harness 失败均保留。原始证据位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk2-sample-multistep4-world1-20261012-r2/`，只读修正比较位于 `$TASK_STATE/runs/qwen3-topk2-sample-multistep4-comparator-fix1-20261012-r1/`。基线 HEAD `f46b3bd6813648495486b23252b52837bfb0f930`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`，同步目标 `origin/2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`。|
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、cache-enabled seeded top-k=16 四步 sampling、实际 batch=2 | passed | passed（prefill 精确；匹配 decode logits 在容差内；各侧 token 属于本侧 top-16） | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-cacheon-topk16-sample-multistep4-world1-20261012-r1`：worker 预检 2724 → native 2725 → gate-only 修正 2727 → strict ACL candidate 2728 → worker comparator 2729。初版只读 native gate job 2726 在数据检查前因 manifest 缺少 job 映射触发 `KeyError`，作为 harness-only 保留；2727 只读复用既有 native 产物并通过。四条 prompt、两个 batch=2 请求、每条最多 4 token；实际模型/adapter 目录 `cache-on-base-topk4` / `checkpoint-3`，模型与初态清单可读、未下载，cache=true。两侧参数与 prefill 输入精确，prefill logits 最大差 0，六个匹配 decode 步满足 `atol=1e-5, rtol=1e-4`，每侧 sample token 均在各自 top-16 并由下一步消费；ACL strict `backend_fallback=error`、fallback=0。本次 token 序列相同不代表 RNG 等价。L5 未运行。原始证据位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk16-sample-multistep4-world1-20261012-r1/`，只读比较位于 `$TASK_STATE/runs/qwen3-topk16-sample-multistep4-comparator-fix1-20261012-r1/`。基线 HEAD `38131e2e24af130fcc596ffacc00343982b1b947`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；本格开始前从 mirror 成功同步的 `2.0-refactor` SHA 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是基线祖先。|
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、cache-enabled top-k=32 + top-p=0.9、temperature=0.7 四步 sampling、实际 batch=2 | passed | passed（prefill 精确；六组匹配 decode logits 在容差内；每侧采样 token 符合本侧 top-k/top-p 集合） | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261012-r1`：worker 预检 2882 → native 2883 → native gate 2884 → strict ACL candidate 2885 → worker comparator 2886。4 条 prompt、两个真实 batch=2 请求、共 16 token；8 次 NPU 前向的 decode 长度为 `[15,1,1,1]` 和 `[17,1,1,1]`。加载参数/prefill 精确、prefill 最大差 0，六组 decode logits 在 `atol=1e-5, rtol=1e-4` 内；ACL strict `backend_fallback=error`、fallback=0。两侧本次 token 序列相同不证明 RNG 等价。L5 未运行。 |
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、cache-enabled seeded top-k=4 四步 sampling、实际 batch=2 | passed | passed（prefill 精确；六个匹配 decode logits 在容差内；各侧 token 属于本侧 top-4） | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-cacheon-topk4-sample-multistep4-world1-20261012-r1`：worker 预检 2738 → native 2739 → worker native gate 2740 → strict ACL candidate 2741 → worker comparator 2742，全部退出 0。实际模型/Adapter 为 `cache-on-base-topk4` / `checkpoint-3`；缓存有效配置为 true，离线复用已校验权重与 adapter、无下载。四条 prompt 分两个真实 batch=2 请求，各生成 4 token；输入/加载参数精确，prefill logits 最大差 0，六个匹配 decode 步满足 `atol=1e-5, rtol=1e-4`，各侧采样 token 属于本侧 top-4 并被下一步消费。ACL `backend_fallback=error`、fallback=0。观测到 token 序列相同不作为跨 runtime RNG 等价判据；L5 未运行。原始证据位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk4-sample-multistep4-world1-20261012-r1/`。
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、near-context prefill 124/128、greedy 4 token、实际 batch=2 | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-nearctx-greedy4-world1-20261012-r1`：预检 2658 → native 2659 → native gate 2660 → strict ACL candidate 2661 → worker comparator 2662；两条提示组成一个 batch=2，首次实际 `input_ids` 长度 124，模型 `max_position_embeddings=128`，后续四次前向输入 batch 均为 2。两侧输入、加载参数、greedy token/响应精确，logits 最大差 0；候选 ACL `backend_fallback=error`、fallback=0。离线复用 `base` / `checkpoint-3` 和已核验 core，无下载。该锁定格 L0/L1/L4 通过，L5 未运行。
| 单 NPU 推理：Qwen3 tiny、Swift LoRA、FP32/SDPA、公开 `swift infer`、greedy 4 token、3 条输入形成不满尾批（2+1） | passed | passed | not-applicable | not-applicable | passed | not-run | 新键 `qwen3-public-infer-batch2-partial-final-greedy-world1-20261012-r1`：预检 2641 → native 2642 → native gate 2643 → strict ACL candidate 2644 → worker comparator 2645；worker 实测两批，8 次前向的输入批次为 2、2、2、2、1、1、1、1。3 条结果、输入、加载参数、token/文本精确，logits 最大差 0；ACL strict `error`，fallback=0。Candidate 2644 仅在完整推理输出之后的复制版 stdin shell 检查以 `NameError: error` 退出；比较器 2645 验证模型产物及零回退后通过。未重跑 candidate。L5 未运行。 |
| 单 NPU 推理：tiny LLaMA、Swift Adapter length=16/GELU、greedy 4 token、公开 `swift.cli.main infer` | passed | passed | not-applicable | not-applicable | passed | not-run | 原生 job 1940（run r4）→ strict ACL job 1975（run r6）→ worker comparator job 1976；2461 项、0 失败，输入/token 精确一致，参数最大差 `4.47035e-8`、logits `1.19209e-7`。初始 Adapter 线性层在 CPU，Swift Adapter 首次前向按输入设备延迟迁移；之后 16 次前向的 29 个参数均为 `npu:0`。候选 strict `backend_fallback=error`，完整阶段 fallback 起止均为 0。L5 未运行。 |
| 单 NPU 推理：tiny LLaMA、Swift IA3 checkpoint-1、greedy 4 token、公开 `swift infer` | passed | passed | not-applicable | not-applicable | passed | not-run | 新运行键 `ia3-public-infer-current-world1-20261011-r3`：native oracle job 2386 → strict ACL candidate 2387 → worker comparator 2389。37 个快照、2,048 个 logits 值、0 失败；公开 JSON 结果、全部已加载参数、每次 forward 输入与 token ID 精确一致，logits 最大差 0。native/candidate 两个进程记录分别确认 torch_npu/ACL、rank0/world-size=1；所有实际 forward 输入、同步后全部模型/adapter 参数、logits 与 token tensor 均为 `npu:0` 且候选 location=device；ACL `backend_fallback=error`、fallback start/end=0。首次 Slurm harness job 2383 在模型加载前失败；r2 候选 job 2385 暴露探针过早读取 lazy location，原样保留；比较器 job 2388 的文件筛选失败已在同一只读产物上修正，不重跑模型。L5 未运行；首轮编译时间不作为稳态性能。 |
| 单 NPU 推理：tiny Mistral、Swift Adapter checkpoint-3、公开 `swift infer`、cache-on、batch=2、top-k=32 + top-p=.9、temperature=.7、四步 sampling、prefill 超过 sliding window 16 | passed | passed（prefill 与匹配 decode logits 在容差内；各侧活动 token 属于本侧 top-k/top-p 集合；adapter 叶子输入/输出均在 NPU） | not-applicable | not-applicable | passed | not-run | 模型运行键 `mistral-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261012-r3`：worker 预检 2850 → native torch_npu infer 2851 → native gate 2852 → strict ACL candidate 2853；比较器专属键 `mistral-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261012-r4` 的 marked-srun job 2855 只读复用 r3 文件且通过，没有重跑推理。四条固定原始 20-token prompt 分两个真实 batch=2 请求，模板后 prefill 长度均为 29（大于滑窗 16），每个请求观察到 `[29,1,1,1]`。两侧有效 `GenerationConfig` cache=true、sample=true、top-k32/top-p=.9/温度.7、max-new-token=4；两侧加载参数数值、prefill 输入精确，prefill logits max abs `8.94e-8`，匹配的全部 decode row 在 `atol=1e-5, rtol=1e-4` 内；EOS=2 后 pad=0 不计入采样检查，各侧活动 token 符合本侧截断候选且后续 cache 消费前一采样 token。两侧本次输出/token 序列恰好相同，不作为跨 runtime RNG 判据。base 21 个参数两侧均为 `npu:0`；32 个 adapter 叶子调用的输入/输出均为 `npu:0`；但 native 观察到 8 个 adapter 参数留在 CPU，而 candidate 参数在 NPU（权重值相同）。该设备分布差异如实保留，未测性能；candidate ACL `backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。cache 由离线本地模型副本的 config/generation config 启用，不是 Swift 公开 cache on/off 对照；L5 未运行。r1 job 2846 使用错误 Jittor 1.3 cache 造成 candidate `bootstrap --check` 失败；r2 job 2847/2848 分别为候选 cache 预检失败/原生输出，gate job 2849 的“全部参数必须 NPU”断言过严，均未形成 ACL 候选结论。r3 比较器 job 2854 将 EOS 后 pad 误算作 sample token，在比较器断言前失败；r4 修正 EOS-aware 活动行检查后通过。上述均为缓存/检查器 harness 失败，不追溯改判或覆盖。原始产物未版本化，模型运行证据位于 `$TASK_STATE/runs/mistral-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261012-r3/`，只读比较与父产物摘要位于 `...-r4/`；本格基线 HEAD `7872017dedde72b1d96642c7c1138c64c7e0bff2`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`，本次集成前目标 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 为 HEAD 祖先。 |
| 单 NPU 推理：tiny Mistral、Swift Adapter checkpoint-3、公开 `swift infer`、cache-on、batch=2、top-k=32 + top-p=.8、temperature=.7、四步 sampling、prefill 超过 sliding window 16 | passed | passed（prefill 与全部匹配 decode 行 logits 在容差内；活动行样本满足本侧 top-k/top-p 集合；adapter 叶子输入/输出均在 NPU） | not-applicable | not-applicable | passed | not-run | 新键 `mistral-public-infer-batch2-cacheon-topk32-topp08-temp07-sample-multistep4-world1-20261012-r2`：worker 预检 2864 → native torch_npu infer 2865 → native gate 2866 → strict ACL candidate 2867；初始 comparator job 2868 将 EOS 后 pad id 0 误判为 sample，保留为 harness 失败；新 comparator-only 键 `mistral-public-infer-batch2-cacheon-topk32-topp08-temp07-sample-multistep4-comparator-fix1-20261012-r1` 的 marked-srun job 2869 修正活动行检查并只读复用已有产物通过，未重跑推理。四条固定原始 20-token prompt 分成两次真实 batch=2，prefill 长度均为 29（大于滑窗 16），每次观察到 `[29,1,1,1]`。两侧有效 GenerationConfig 均为 cache=true/sample=true/top-k32/top-p=.8/temperature=.7/max-new-tokens=4；加载参数及 prefill 输入精确，prefill 最大 logit 差 `8.94e-8`，全部匹配 decode row 在 `atol=1e-5, rtol=1e-4` 内。每侧 top-k/top-p 活动 token 都在自己的截断候选集，EOS=2 后 pad=0 不计入采样；严格 ACL `backend_fallback=error`、fallback=0。两侧此 run 输出碰巧相同，不作为跨 runtime RNG 判据。native 21 个 base 参数位于 NPU、8 个 adapter 参数留在 CPU；candidate adapter 参数在 NPU；两侧 32 个 adapter 叶子调用的输入/输出均为 NPU，设备差异保留。无性能验收；L5 未运行。基线 HEAD `5afec492fe3383528c8f18ff81ac4f9e5c74ada3`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；模型/adapter 均为本地已验证文件，无下载。 |
| 单 NPU 推理：tiny Mistral、Swift Adapter checkpoint-3、公开 `swift infer`、cache-on、batch=2、top-k=64 + top-p=.8、temperature=.7、四步 sampling、prefill 超过 sliding window 16 | passed | passed（prefill 与全部匹配 decode logits 在容差内；各侧活动 token 属于本侧 top-k/top-p 集合；adapter 叶子输入/输出均在 NPU） | not-applicable | not-applicable | passed | not-run | 新键 `mistral-public-infer-batch2-cacheon-topk64-topp08-temp07-sample-multistep4-world1-20261012-r1`：worker 预检 2872 → native torch_npu infer 2873 → native gate 2874 → strict ACL candidate 2875 → 只读比较器 2876，均通过。四条固定 20-token 原始 prompt 分成两个真实 batch=2 请求；native gate 观察到两组 8 次 NPU 前向，序列长度均为 `[29,1,1,1]`，有效 GenerationConfig 为 cache=true/sample=true/top-k64/top-p=.8/temperature=.7/max-new-tokens=4。两侧加载参数与 prefill IDs 精确，prefill 最大 logit 差 `8.94e-8`，8 组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；各侧采样 token 满足本侧 top-k64 后 top-p=.8 候选集并由下一步 cache 输入消费。两侧本次 token 序列相同不代表跨运行时 RNG 等价。Native 21 个 base 参数在 `npu:0`、8 个 adapter 参数在 CPU；candidate ACL `backend_fallback=error`、fallback=0，adapter 叶子调用输入/输出在 `npu:0`。本格无性能验收，L5 未运行。模型/adapter/输入/初态均为已验证本地材料，未下载权重；原始证据未版本化，位于 `$TASK_STATE/runs/mistral-public-infer-batch2-cacheon-topk64-topp08-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `2b607f5ac31e079de8af465f4164c6a9a681c5a0`，同步目标 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 为该基线祖先。 |
| 单 NPU 推理：tiny Mistral、Swift Adapter checkpoint-3、公开 `swift infer`、本地模型与 generation config cache-off、batch=2、top-k=32 + top-p=.9、temperature=.7、四步 sampling | failed | blocked（实际有效配置是 cache-on，不能验收请求的 cache-off 语义） | not-applicable | not-applicable | failed | not-run | 新诊断键 `mistral-public-infer-batch2-cacheoff-topk32-topp09-temp07-sample-multistep4-world1-20261012-r3`：marked-srun preflight 2861 确认两份本地配置均为 `use_cache=false`、公开 `swift infer --help` 无 `--use_cache`、torch_npu 与 strict ACL/core/fallback 预检通过；native 公开 infer 2862 本身完成并产出四行结果，但 observer 捕获的有效 GenerationConfig.use_cache 为 true。marked-srun 原生 gate 2863 据此拒绝 cache-off 格，ACL candidate 未运行。根因在 Swift generation config 准备/合并语义，不是 ACL 兼容失败；没有修改源码。另保留 r1 job 2859（bootstrap observer 父目录缺失，未进入 CLI 检查）和 r2 job 2860（确认 CLI 没有显式 flag、未加载模型）的早期尝试。cache-off 修复方案待用户确认；不追溯改判本行。 |
| 单 NPU 推理：约 1.1B Qwen3、公开 infer | passed | failed | not-applicable | not-applicable | failed | not-run | 第 349 节：公开响应/tokens 精确，但 logits/KV 数值有 34/1618 项超出固定阈值；L5 不得启动。 |
| 单 NPU 推理：真实尺寸 BERT-base sequence classification、checkpoint-12、公开 `swift infer --infer_backend transformers --task_type seq_cls` | passed | failed | not-applicable | not-applicable | failed | not-run | 新键 `bert-base-seqcls-public-infer-world1-20261011-r1`：native torch_npu job 2366 → strict ACL candidate job 2367 → worker comparator 2368。六条输入、tokenization 和公开标签一致；102 项中 20 项 logits/log-probability 数值超过固定 `atol=1e-5, rtol=1e-4`。两侧参数、输入与 logits 均在 `npu:0`；candidate strict `backend_fallback=error` 且起止 fallback=0。层级 trace（2371/2372/2376）首个差异为 `layer00.intermediate_activation`，此前 embedding 精确且 dense 误差在阈值内。相同 GELU 输入 SHA 的隔离运行（native 2377、candidate 2378、比较 2380）复现 `max_abs=4.73302e-4, RMSE=2.58275e-4`；当前 ACL 注册路径用 `approximate=0` 的 `aclnnGeluV2`，既有 NPU GELU route/contract 探针显示 torch_npu 的 `none`/`tanh` 输出相同并符合 tanh 公式。差异归于两运行时 GELU 数值语义不同；保留严格推理失败，不改写语义或放宽阈值。比较器路径错误 job 2379 保留为 harness 失败，不用于数值结论。L5 未运行。 |
| 单机 2 NPU 训练：IA3、旧 fused AdamW 设备/单步审计、公开 SFT | passed | failed | failed | failed | failed | not-run | 第 420 节：forward 数值未保存，故 L1/L2 严格比较失败；候选 `optimizer.pt` 非有效 ZIP，L3 失败。局部参数、梯度、更新与设备观察保留为部分证据，不提升等级。 |
| 单机 2 NPU 训练：IA3、FP32、`adamw_torch_fused`、公开 `swift.cli.sft` 四步及 checkpoint-3 fresh-process 恢复 | passed | passed | passed | passed | passed | not-run | 新运行键 `ia3-world2-fullstate-fused-l3-r3-20261008`；Slurm 1713/1714 执行候选，1724 在 worker 上完成 protocol v3 比较。checkpoint 224 项与轨迹/设备 298 项零失败；optimizer 浮点状态按 `1e-7` 比较、adapter/参数轨迹及 forward/gradient/update 按预锁容差比较；每 rank NPU/HCCL、候选 strict error、fallback=0。补充单步数值/驻留审计运行键 `ia3-world2-fused-value-audit-20261010-r2`，oracle 2189 → candidate 2190 → read-only comparator 2193，双 rank 严格比较 0 失败：初始/冻结参数和 batch 精确一致，forward 最大差 `4.76837e-7`、gradient `3.72529e-9`、optimizer 42 个值最大差 `3.49246e-10`，预置容差分别为 `atol=1e-6, rtol=1e-5`；参数、计算输入、输出、梯度和 optimizer 张量逐 rank 驻留对应 NPU，候选 ACL strict error 且 fallback=0。该补证不改变四步及恢复的 L0–L4 结论，也不覆盖真实尺寸性能。L5 未运行。该 L0–L4 是原锁定数值比较协议下的历史结论；新收敛协议复核见末尾追补。首次比较器 `1715` 与 protocol v2 的失败原样保留，原因及修正见下文。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift IA3、FP32 fused AdamW、新收敛协议 8 步与 checkpoint-4 fresh-process 恢复（native oracle only） | not-run | not-run | failed | not-run | not-run | not-run | 新键 `ia3-world2-convergence-resume-20261011-r2`，native job 2412；旧 strict 对拍键不复用。Native 连续 step 1–3/6–8 loss 均值 `4.84470685 → 4.85686668`，checkpoint-4 恢复 step 5–6/7–8 均值 `4.84205961 → 4.84973407`，均未下降，故未启动 candidate。Native 实际完成公开 CLI 的 8 步连续训练及新进程恢复，但 Jittor L2–L4 未运行。脚本 gate 另有两个修正点：on-log 聚合事件覆盖了 step-8 loss、rank 0 主进程记录使进程文件计数为 3；原始事件含 step-8 loss，rank 设备事件无驻留错误。r1/job 2411 因逐作业句柄未命中在模型启动前退出；jobs 2413/2414 依赖取消。L5 未运行。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift IA3、FP32 fused AdamW、新收敛协议连续 32 步与 checkpoint-16 fresh-process 恢复至 step 32 | passed | passed | passed | passed | passed | not-run | 新键 `ia3-world2-convergence-resume-20261011-r3/r4`，native oracle 2420，ACL candidate continuous 2421，rank cache preflight 2423，ACL fresh resume 2424；原严格比较 job 2425 仍按旧协议报告 1,313 个 exact float mismatch failure，报告保持不变。新收敛/恢复审计 job 2436 对 r3/r4 只读检查 0 失败；native continuous `4.844706853→4.791386445`、native resume `4.838513374→4.791386445`，ACL continuous `4.844707171→4.791386922`、ACL resume `4.838513374→4.791386922`。双方每 rank 初态与实际 batch 序列精确；checkpoint-16 至 step32 的 optimizer/scheduler/trainer/RNG 结构、step 和下一批连续性通过，native 与候选同运行 RNG/scheduler 状态差为 0。各 rank 参数、计算输入/输出、梯度、optimizer state 实测驻留对应 NPU；HCCL world-size=2；候选 ACL `backend_fallback=error`、完整观察窗 fallback=0。1313 个历史 strict float mismatch 仍保留为旧协议结论，仅在已确认新协议中作为诊断，不使用数值阈值。仅此锁定配置 L0–L4；L5 未运行。原始证据未版本化，位于 `$TASK_STATE/runs/ia3-world2-convergence-resume-20261011-r3/`、`...-r4/` 与 `...-r15-acceptance-audit-v5/`。
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32、公开 `swift.cli.rlhf --rlhf_type dpo` 三步与 checkpoint-3 fresh-process 恢复（历史 Jittor 1.3.11） | not-run | not-run | not-run | not-run | not-run | not-run | 运行键 `dpo-world2-20261003` 的候选日志实际为 Jittor 1.3.11，目标为 Jittor 2.0-refactor，故不计入本矩阵。旧 case 有公开双卡训练/恢复与 adapter/optimizer 数值比较，但没有全梯度、每 rank batch/forward 输出和计算张量物理设备审计；不得据此授予目标 L0–L4。需在当前基线下 oracle-first 完整审计。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32、公开 `swift.cli.rlhf --rlhf_type orpo` 四步及 checkpoint-3 fresh-process 恢复（历史 Jittor 1.3.11） | not-run | not-run | not-run | not-run | not-run | not-run | 运行键 `orpo-world2-rngfix-20261007` 的候选日志实际为 Jittor 1.3.11，不是本任务目标 Jittor 2.0-refactor；故不计入本矩阵。该历史运行的 native/candidate 数值轨迹与恢复 comparator 均为零失败，但 observer 未保存 forward 值或每 rank batch/参数/gradient/optimizer 物理 device。需在当前 2.0 基线下 oracle-first 重跑完整审计。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32 fused AdamW、公开 `swift.cli.rlhf --rlhf_type dpo` 新收敛合同连续 32 步与 checkpoint-16 fresh-process 恢复至 step 32 | passed | passed | passed | passed | passed | not-run | 新键 `dpo-world2-convergence-resume-20261011-r2/r6`：native torch_npu oracle job 2506 → strict ACL candidate job 2517 → 只读新合同复核 job 2520（`...-r7`）及 native/candidate 初态与 batch 交叉核验 job 2521（`...-r8`）。r8 对每 rank 的 28 个可训练参数、21 个冻结参数、1 个 buffer 做精确比较，并逐 rank 精确比较连续 32 步与恢复 16 步共 96 组 batch；0 失败。公开 `swift.cli.rlhf` 参数入口分别完成连续 32 步及新进程从 checkpoint-16 恢复至 step 32。Native continuous loss 窗口均值 `0.680024366 → 0.588044643`、resume `0.626893322 → 0.588044643`；candidate 复核对应 `0.680024366 → 0.588044643`、`0.626893322 → 0.588044643`。两 rank 的 step 1–32/17–32 loss、实际 batch、初态、checkpoint 参数及 integer optimizer/scheduler/RNG 状态、数据游标与恢复继续轨迹门禁通过。两种轨迹每 rank 的全部参数、实际计算输入/输出、前向输出、梯度、buffers 和 optimizer state 审计均驻留对应 NPU；world-size=2/HCCL，candidate runtime/provider 为 Jittor shim/ACL、`backend_fallback=error`、每个连续/恢复 rank 的 fallback 计数均为 0；native 通用 fallback counter unavailable/unknown。历史 strict candidate gate job 2517 的 `candidate-gate.json` 保留 1,792 个 optimizer 浮点数组精确不等失败；协议确认后，job 2520 仅对 r6 原始产物重跑审计逻辑，键/shape/dtype、integer state、有限浮点值、状态结构、恢复轨迹、batch、设备与回退门禁均为 0 失败，36 组浮点数组差异保留为诊断（最大绝对差 `1.1175871e-8`、最大相对差 `8.0705e-3`，不再单独判失败）。这不是追溯修改 r6 的严格失败报告。L5 未运行；结论只适用于该 tiny DPO/Swift LoRA/锁定数据配置，不代表其它 tuner、模型族或完整 ms-swift 功能面。原始证据未版本化，位于 `$TASK_STATE/runs/dpo-world2-convergence-resume-20261011-r2/`、`...-r6/` 与只读复核目录 `...-r7/`。|
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32、公开 DPO 四步与 checkpoint-3 fresh-process 恢复（目标 2.0-refactor） | passed | passed | passed | passed | passed | not-run | 新运行键 `dpo-world2-2x-full-audit-20261009-int64roll-fix01`：复用先行 native oracle job 1783；strict ACL candidate job 1887 连续训练四步，job 1888 在新进程从 checkpoint-3 恢复并续至 step 4；比较 job 1905 完成 170 项比较、0 失败，最大绝对差 `1.1920928955078125e-07`。两 rank 均为 HCCL/world-size=2；candidate `backend_fallback=error`，连续与恢复阶段 fallback 均为 0。每 rank 参数、实际 batch/计算输入、前向输出、梯度和 optimizer state 均驻留对应 NPU；原生 torch_npu 通用 fallback counter unavailable/unknown。L0–L4 仅适用于此锁定配置，L5 未运行。旧候选/比较器失败原样保留，根因与完整证据见下文。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32 fused AdamW、公开 `swift.cli.rlhf --rlhf_type orpo` 新收敛合同 32 步与 checkpoint-16 新进程恢复至 step 32 | passed | passed | passed | passed | passed | not-run | 新键 `orpo-world2-convergence-resume-20261011-r1`：native oracle job 2456、native gate 2458 先于 strict ACL candidate job 2477；acceptance audit job 2483 通过。Native continuous `4.881912→4.390640`、resume `4.612972→4.390640`；candidate continuous `4.881913→4.390640`、resume `4.612972→4.390640`。两 rank 实际 batch、初态、HCCL/world-size=2 与逐 rank 设备驻留检查通过；候选 ACL `backend_fallback=error` 且完整窗口 fallback=0。checkpoint optimizer state audit job 2490 确认两侧 checkpoint-16 各 28 个 state slot、84 个字段、34,844 个值逐 runtime 精确对应其 continuous step-16 保存边界，optimizer step 与 trainer global_step 均为 16。只读诊断 job 2484 对 480 组跨实现快照记录浮点差异、无阈值，恢复连续性 388 组无结构/有限值失败，最大绝对差约 `1.49e-7`。L5 未运行；历史严格对拍行保持不变。前置和审计器路径/reader 失败运行均保留在 state；没有重跑训练。证据目录：`$TASK_STATE/runs/orpo-world2-convergence-resume-20261011-r1/`、`...candidate-convergence-resume-20261011-r3/`、`...candidate-convergence-audit-20261011-r2/` 至 `...-r12/`。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32 fused AdamW、公开 ORPO 四步与 checkpoint-3 fresh-process 恢复（目标 2.0-refactor） | passed | passed | passed | passed | passed | not-run | 新键 `orpo-world2-2x-full-audit-20261008-jobguardfix01`；native oracle job 1800 → strict ACL candidate job 1801 → fresh-process resume job 1802 → comparator job 1803。公开 `swift.cli.rlhf --rlhf_type orpo`，world-size 2、HCCL、四步固定轨迹和 checkpoint-3 续训；146 项比较、0 失败。两侧 rank0/rank1 的参数、实际 batch/计算输入、前向输出、梯度及 84 个 optimizer-state 张量均在对应 NPU；candidate runtime=Jittor shim/provider=ACL、`backend_fallback=error`、起始/结束 fallback=0。native runtime/provider=torch_npu，通用 fallback counter `unavailable/unknown`。L5 真实尺寸稳态性能未运行；结论只适用于该锁定配置。未版本化原始证据位于 `$TASK_STATE/runs/orpo-world2-2x-full-audit-20261008-jobguardfix01/`。 |
| 单 NPU 训练：tiny LLaMA、Swift LLaMAPro、FP32 fused AdamW、公开 `swift.cli.sft` 三步 | passed | passed | passed | failed | passed | not-run | 运行键 `llamapro-world1-full-audit-20261009-jittor2-fused-r1`。native oracle job 1917 先运行；strict ACL candidate job 1922 与 worker comparator job 1926 对 30 项比较、0 失败，最大绝对差：参数 `5.97909e-7`、optimizer `2.7940e-9`、logits `8.9407e-8`、梯度 `1.4901e-8`、更新 `5.9605e-7`，均在预锁容差内。公开三步训练的 L0–L2/L4 通过；candidate `backend_fallback=error`、HCCL world-size=1、fallback=0，参数/batch/forward/gradient/optimizer 均在 NPU。L3 native 新进程恢复 job 1928 首次因 SwiftModel checkpoint 缺根目录 shard index 失败；新运行键 `llamapro-world1-full-audit-20261009-jittor2-fused-resume-index-r1` 的 job 2141 仅为原 checkpoint 复制并添加索引元数据后重试 native oracle，随后通过模型加载并在 optimizer load 处失败：SwiftModel 统计 0 个可训练参数，保存的 optimizer 参数组含训练参数，报 `loaded state dict contains a parameter group that doesn't match the size of optimizer's group`。安装版 Swift 4.5.2 `TunerMixin.prepare_model` 传入 `is_trainable=True`，但 `SwiftModel.from_pretrained` 的公开签名默认 `inference_mode=True` 且未消费 `is_trainable`；这与零可训练参数现象一致，是当前证据指向的上游恢复语义根因。两次 native 失败均未运行 candidate resume；须先有 native 公共恢复通过，不能用 adapter-only 加载替代完整状态。此配置 L3 failed、L5 not-run。原始证据：`$TASK_STATE/runs/llamapro-world1-full-audit-20261009-jittor2-fused-r1/`、`...-resume-r2/`、`...-resume-index-r1/`。 |
| 单 NPU 训练：tiny LLaMA、PEFT DoRA q/v、FP32 fused AdamW、公开 `swift.cli.sft` 八步及 checkpoint-2 新进程恢复到 step 8 | passed | passed | passed | passed | passed | not-run | 新协议连续运行键 `dora-world1-convergence-audit-20261010-r3`（oracle 2321、比较 2322）与恢复运行键 `dora-world1-resume-audit-20261010-r5`（oracle+candidate 2336、只读比较 `dora-world1-resume-compare-20261010-r2` job 2338）。各 runtime 连续与恢复窗口分别下降；optimizer/scheduler/RNG 精确恢复、step 2 到 8 六个实际 batch 及初始可训练/冻结状态逐 runtime 精确衔接，两条轨迹共 14 次参数更新检查通过。两侧逐张量输入、forward、梯度、optimizer 与训练参数设备均在 `npu:0`；candidate strict ACL `backend_fallback=error` 且观察窗 fallback=0，oracle fallback counter unavailable/unknown。跨实现浮点差异仅诊断。L4 限于此公开 CLI 锁定 case；L5 未运行。历史严格数值失败保留，未追溯改判。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift Adapter（length=16/GELU）、FP32 fused AdamW、公开 `swift.cli.sft` 三步及完整恢复到 step 4 | passed | passed | passed | passed | passed | not-run | 初始三步公开 SFT 运行键 `adapter-world2-full-audit-20261009-jittor2`：native oracle 1930 → strict ACL candidate 1931 → worker comparator 1935，2092 项、0 失败。L3 复核运行键 `adapter-world2-resume-audit-20261009-jittor2-r2/r4` 与 `adapter-world2-uninterrupted-audit-20261009-jittor2-r2`：native resume 2082、strict ACL resume 2091、native uninterrupted 2135、strict ACL uninterrupted 2136；worker compare-r6 job 2139 检查 11,637 项、0 失败。含两侧逐 rank 新进程 checkpoint 恢复/续训与不中断轨迹、batch 精确一致、checkpoint adapter/optimizer/scheduler/trainer/RNG，参数、forward、gradient、update 与 optimizer 值在预锁类别容差内；最大绝对差 logits `1.1920928955078125e-7`、梯度 `5.960464477539063e-8`、参数/更新 `4.470348358154297e-8`、optimizer `1.1175870895385742e-8`。两 rank HCCL 与实际 NPU 驻留逐项通过；候选 ACL `backend_fallback=error` 且起止 fallback=0。L5 未运行。先前 native checkpoint 分支和 comparator r4/r5 协议/元数据失败仍保留，不用于本 PASS；详见后续 L3 复核记录。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift Adapter（length=16/GELU）、FP32 fused AdamW，新收敛协议 32 步及 checkpoint-16 新进程恢复至 step 32 | passed | passed | passed | passed | passed | not-run | Native oracle r3 与 candidate continuous/resume 是独立轨迹；恢复审计新键 `adapter-world2-convergence-resume-20261011-r10`，Slurm job 2453，gate `failure_count=0`。两 rank HCCL/ACL、fallback `error` 且计数 0；step 16–31 实际 batch 与各自不中断轨迹精确。每 rank 8 个可训练参数、24 个 optimizer state tensor、scheduler/global step 16 与 checkpoint reference 精确恢复，rank-local RNG 精确恢复；candidate step 17–19 至 30–32 平均 loss `4.696953138 → 4.666621208`，native 对应窗口 `4.696953297 → 4.666621208`。每 rank 参数、计算输入/输出、前向输出、梯度及 optimizer state 物理驻留对应 NPU。R8 的 1,298 项旧严格浮点 mismatch 与 R9 的检查点提取前置失败均保留，不追溯重判；L5 未运行。原始证据未版本化，位于 `$TASK_STATE/runs/adapter-world2-convergence-resume-20261011-r8/`、`...-r9/`、`...-r10/`。 |
| 单机多 NPU 训练：Swift LoRA 候选公开分布式路径 | not-run | not-run | failed | not-run | failed | not-run | 第 349 节：已有真实 HCCL 微检，但候选公开 Trainer 路径先后在 DDP hook 导入、token-count gather 处失败；未产生可验收的分布式训练轨迹。原生 2/4/8 NPU 仅为 oracle。 |
| 真实多机训练：每台至少 2 NPU、至少 2 台主机 | resource-blocked | resource-blocked | resource-blocked | resource-blocked | resource-blocked | resource-blocked | 第 316、349 节：现有真实分配只有单主机；解除需实际多机 Slurm 资源。单机模拟不计。 |
| 单 NPU 训练：tiny LLaMA、Swift Prompt direct API、FP32、新收敛协议 16 步与 checkpoint-8 新进程恢复到 step 16 | passed | passed | passed | passed | not-applicable | not-run | 新锁定配置：native key `swift-prompt-direct-api-checkpoint-world1-20261012-r2`（model job 2538、gate 2549）；candidate continuous key `...-acl-r1` job 2553 的 16 步连续段通过，后半段只因 probe 对 shim `_OptState` 调用 `.clear()` 失败。Candidate resume-only key `...-acl-r2` job 2558 从该 candidate step-8 checkpoint 在 fresh process 恢复并续训至 step 16；配对只读审计 job 2560 通过。Native/candidate continuous loss 首末三步均值分别 `4.88290278→4.85498110`、`4.88290278→4.85498047`；resume 分别 `4.86900727→4.85498095`、`4.86900727→4.85498047`。两侧各自 checkpoint 的 step-8 model/optimizer 与 continuous snapshot 精确，恢复 step/cursor、固定 batch、scheduler 与 Python/NumPy/Torch CPU/NPU RNG 状态检查通过；candidate ACL `backend_fallback=error` 且 fallback 起止均为 0。初始模型逐项一致；same-runtime 恢复轨迹及跨 runtime 浮点差仅作诊断、不设阈值。candidate job 2551 的 shim import harness failure、gate jobs 2540/2544 的旧断言缺陷、candidate job 2553 的 `.clear()` harness failure、配对审计 job 2559 缺 Slurm step marker 均保留，不计作模型失败或新 PASS；新审计 job 2560 在正确 step launcher 下通过，旧严格结论未改判。公开 Prompt CLI route 不存在，L4 not-applicable；L5 未运行。Raw evidence: `$TASK_STATE/runs/swift-prompt-direct-api-checkpoint-world1-20261012-r2/`, `...-acl-r1/`, `...-acl-r2/`. | 新键 `swift-prompt-direct-api-current-world1-20261011-r5`：native job 2403 → strict ACL candidate job 2404 → worker 只读诊断 job 2405。两侧初始模型 state 与固定 NPU batch 精确一致；每侧首/末三步均值 loss 分别为 native `4.8643740→4.8478611`、candidate `4.8643740→4.8478613`。每侧全参数/buffer、输入 batch、每步 logits、Prompt 梯度、更新参数和 SGD momentum state 均记录为 NPU；Prompt 参数更新，冻结参数保持不变，candidate ACL strict `error` 且完整运行窗 fallback=0。跨运行时保存的 loss/梯度/参数/momentum 差异只作诊断；probe 未保存 forward logits 数组，因此 logits 数值诊断缺失。L3 未运行。公开 SFT Prompt route 在源码盘点中不存在，故该 direct API 不构成 L4，公开 CLI 项 not-applicable；L5 未运行。历史 2026-10-05 SHA `8ce6996` 探针不继承。 |

IA3 r3 已完成：native oracle 连续训练与 fresh-process resume 先运行；候选随后通过真实双 NPU `swift.cli.sft` 连续四步与 checkpoint-3 新进程续训。Slurm 1715 原始比较失败是因 comparator 把内部 optimizer 张量设为精确零容差；cmpv2 仅修正该项后 checkpoint 224 项零失败，但轨迹 comparator 又把跨 runtime 恢复初始可训练参数错误设为零容差。cmpv3 将这项恢复初始参数比较改为预置轨迹阈值 `1e-6`，其余键/shape/dtype/有限值、冻结参数精确、optimizer、scheduler、trainer、RNG、逐 rank HCCL/device/fallback 检查保持不变；Slurm 1724 完成，checkpoint 224 项及完整轨迹/设备 298 项均零失败。候选两 rank 使用 ACL、`backend_fallback=error`、结束回退计数均为 0；batch、forward output、参数、梯度和 optimizer state 的 rank 本地物理设备记录均为对应 NPU。原始运行及 comparator 协议/结果未版本化，位于 `$TASK_STATE/runs/ia3-world2-fullstate-fused-l3-r3-20261008/` 与 `...-cmpv2-20261008/`、`...-cmpv3-20261008/`。这是这一锁定配置的 L0–L4 证据，不覆盖 L5 或其它模型/tuner。

### 按适配功能面列出的覆盖边界

| 功能面 | 全范围状态 | 已验证的锁定配置与缺口 |
| --- | --- | --- |
| 入口与配置 | not-run | tiny 单卡 `swift` SFT 与 Qwen3 tiny `swift infer` 已在上表逐配置判定；完整 CLI/launcher 与配置组合矩阵未跑完。 |
| 模型族 | not-run | causal LM 的 tiny LLaMA、Qwen3 有配置级证据；BERT sequence-classification encoder / Swift Adapter direct API 有 L0–L3 证据；tiny BERT full finetuning 的公开 CLI 在 `eps=1e-5` 锁定配置通过 L0–L4，另有新协议 `eps=1e-6` batch=6 固定样本配置通过 L0–L2/L4，L3 未运行；batch=1 的旧 `eps=1e-6` 严格更新对拍与 r1 收敛窗口失败均保留。Qwen3 约 1.1B 推理严格数值失败。当前 ms-swift 4.5.2 worker registry 盘点有 222 个 model type，其中 118 个标记多模态；无 T5 type，另含 ModernBERT encoder/reranker 与视觉/多模态类型。除上述 BERT 锁定用例外，其它 encoder 与视觉/多模态 L0–L5 未运行。 |
| tuner | not-run | Swift LoRA tiny 单卡 SFT、IA3 两卡 FP32 fused AdamW、ORPO 锁定配置均有 L0–L4 证据；LLaMAPro 单 NPU与 Adapter 双 NPU训练各有配置级 L0–L2/L4 证据，但 native L3 均未通过；PEFT DoRA 有一个八步公开 SFT 锁定 case 的 L0–L4 证据。ms-swift 4.5.2 在 `swift.tuners.mapping` 注册 Prompt，但 `tuner_plugin.mapping` 和 SFT `prepare_adapter()` 没有 Prompt 路由，CLI 训练准备最终会报 `Unknown tuner_type`；因此没有可执行的公开 Prompt CLI 轨道。当前 Prompt direct API 的 tiny 单 NPU 12 步配置通过 L0–L2；另一个 16 步新收敛/checkpoint-8 fresh-process resume 配置通过 L0–L3，L4 因公开 CLI 路由缺失而不适用，L5 未运行。12 步训练 runner 未保留 raw logits，故该配置跨运行时 logits 数值诊断缺失；16 步配置通过只适用于其锁定数据、优化器和恢复协议。旧 SHA 8ce6996 探针不继承。其它 Swift tuner 与 PEFT 互操作未运行。 |
| 训练流程 | not-run | 上表锁定的单卡 SFT 有通过/失败项；IA3 fused AdamW、ORPO 双卡公开流程按历史与新收敛合同各有锁定配置通过 L0–L4；Adapter 双卡公开训练 L0–L2/L4 通过但 L3 native oracle 失败；DPO 在公开训练首步失败，其它 RL/奖励训练组合尚未完成 oracle-first 对拍。 |
| optimizer、scheduler 与 AMP | not-run | tiny FP32 AdamW、ORPO 与 IA3 fused AdamW 的 optimizer checkpoint/恢复状态及逐 rank 物理驻留按锁定运行验收；ORPO 新收敛键另检查 checkpoint optimizer 28 个 state slot 与 34,844 个值逐 runtime 精确衔接；AMP及其它 optimizer 组合未运行。 |
| checkpoint 与恢复 | not-run | tiny 单卡 LoRA、IA3 fused AdamW、ORPO（含新收敛合同 checkpoint-16/step-32）、Swift Adapter 双卡及 PEFT DoRA 锁定配置有全新进程恢复证据；LLaMAPro native 新进程恢复先后遇到 SwiftModel shard-index 路径及零可训练参数/optimizer 组不匹配；其它模型、tuner 与状态组合未运行。 |
| 推理与生成 | not-run | Qwen3 tiny（batch=2，greedy 4/16 token、top-k=1 16-token 与 top-k=4 单 token sampling、near-context prefill 124/128、不满尾批 2+1）及 Swift Adapter/IA3 的锁定公开推理各自通过 L0/L1/L4；约 1.1B 与真实尺寸 BERT-base logits 严格比较失败；多步一般 sampling、top-p 与额外 cache 变体未运行。 |
| datasets、tokenizers、collator | not-run | 固定本地 JSONL 与锁定 tokenizer/collator 路径随已列训练 case 被执行；通用数据源、流式数据、划分策略及更广 collator 组合未运行。 |
| safetensors、导出与设备迁移 | not-run | 已列 tiny 与 IA3 锁定配置 checkpoint/adapter 路径通过；其它 export 格式和跨设备迁移组合未运行。 |
| Transformers、PEFT、Accelerate 等依赖调用 | not-run | 只验证 ms-swift 在上述公开 case 实际经过的调用路径；不代表任一依赖库独立全量支持，未覆盖 API 未运行。 |
| 分布式 | not-run | IA3 与 ORPO 锁定配置候选公开训练/恢复在 HCCL 双卡逐 rank 驻留审计通过；历史 ORPO/IA3 配置另有严格对拍通过记录，新收敛配置使用各自收敛与完整状态验收；其它公开流程未覆盖，真实多机资源阻塞。原生多卡 oracle 不等于候选兼容。 |
| 稳态性能 | not-run | 所有轨道尚未同时满足相应正确性与 L4 前置，现存 tiny/编译计时不作 L5。 |

功能面表中的多个状态表示不同锁定配置的分项状态，不是“部分通过”的合成等级；具体模型、流程、设备和拓扑以紧邻的工作负载矩阵及后续章节为准。

四轨分别验收，不能用单卡 case 或通信微检替代分布式公开 launcher。CPU 模型、对拍、训练与性能不在本任务范围。没有新增 adapter，也没有修改 ms-swift 掩盖缺口。已有记录证明的是下列确实执行的部分，不再沿用“单卡 L0–L5 完成”的总体声明。

## 已有运行环境

验证 ms-swift 自身 `swift.tuners.LoRAConfig` 经 Jittor torch shim / ACL 执行，与独立原生 PyTorch/torch_npu NPU oracle 比较；已有数值和计时运行均使用单 Ascend 910B3。

- Python 3.11.15；oracle/shim 使用独立 package site。
- ms-swift 4.5.2、PEFT 0.17.1、Transformers 4.57.6，两侧一致。
- 原生 torch 2.10.0+cpu wheel 配 torch_npu 2.10.0，实际在 NPU 执行；wheel 标签不作为设备证据。
- CANN 9.0.0、driver 25.5.1；官方镜像 digest：`sha256:8742779115ff73944113f5826836c7aa538757c7401e1a636696d184dba8bc4e`。
- 同完整初始 parameters/buffers、同离线固定输入；已有运行键包含源码、版本、设备、协议、权重、输入、环境及包装器哈希。并行进程使用独立 JITTOR_HOME/可写缓存；串行可独占复用已完成的编译缓存，来源记入 manifest，训练及恢复仍使用新进程。相同键复用保存的结果。

原始文件均在 `$TASK_STATE`，不提交日志、NPZ、模型、容器文件或缓存。精确环境见 `session/host-environment-facts.json`；各运行 `manifest.json`、`source.patch`、`source-untracked/` 界定源码范围，不能仅凭共同 HEAD 把不同运行当作同一源码结果。新增四轨运行按环境合同补齐拓扑、每 rank 固定数据划分、全局 batch、优化器和 checkpoint 等运行键输入；不改写旧 manifest。

## 按统一刻度归属已有证据

| 证据归属 | 结果与实际范围 |
| --- | --- |
| L0/L1/L2 定位用例：`ms_swift_lora_llama` | Swift 构造、完整 NPU 前向/反向及独立 oracle 对拍通过。输出误差 0；8 个必需 LoRA 梯度最大归一化误差 `3.7604574904210894e-7`。21 个主干参数保持冻结；integer input_ids 输入梯度不适用。40 项 NPU inventory、线程数 2 和版本/输入/权重条件通过。 |
| L2 多步训练证据：`ms_swift_lora_llama_adamw3` | 固定输入、3 次真实 AdamW 更新通过；83 项初始参数、逐步 loss/梯度/参数/update-delta 轨迹齐全。梯度和 delta 独立按对应 oracle 幅度归一化，避免大初始参数掩盖错误更新。逐步 backward/update 驻留及冻结集合通过；允许首步 LoRA A 数学零梯度，要求整体确实更新。实际 harness 容差：loss/参数 0.002，梯度/delta 0.01。该证据不包含完整 checkpoint，不能归为 L3。 |
| L5 所需的历史测量材料：约 1.1B FP32 Swift LoRA | 3 次预热与 10 次完整训练步、177 项产物及真实 NPU 驻留比较通过。未完成 L3/L4 前置，尚不能通过 L5；实测数字保留下表。 |
| 所有已执行 case 的候选后端/回退证据 | 实际 ACL 身份、单 NPU 驻留、`fallback_policy=error` 和核心计数 0。它们是已执行阶段的硬门禁证据，不构成独立 L5，也不能推出未运行轨道通过。 |

原生端没有已验证的通用 fallback counter：较新的原始报告使用 `None` 表示未提供，绝不是零计数通过。早期单步 oracle 报告中的 0 是旧报告占位，不能视为原生实测，原始产物保留不改。原生 NPU tensor residency 与已知 CPU-fallback 告警检查是独立证据，不替代通用计数。候选的零计数来自真实核心接口和严格策略。

候选尚无证据证明每个分布式 rank 的 hostname、global/local rank、world size、NPU ID、HCCL、梯度、优化器与 checkpoint 状态符合新协议，也尚无完整训练状态的全新进程恢复与续训结果；原生参考的新增证据见下文。上述旧证据只按其已覆盖内容复用，不能补推这些缺项。

## 新协议公开入口与实际断点（01:20 更新）

以下为同步基线 `61294cd14673ba60f4072542a2e73ea2c8f8c509` 上的新增证据。
原生通过只建立 oracle，不意味着候选相应等级通过。两侧锁定 tiny Llama（hidden 64、
2 层、2 heads、vocab 128）完整初始模型和 Swift q/v LoRA；FP32、HF32 关闭，
全局 batch 8、固定顺序、AdamW 三步。CPU 模型执行不在范围。

| 项目 | 实际结果与运行键 |
| --- | --- |
| 原生公开训练与完整恢复 | control `5863b9829ebcd8f8864f4af32e47c185384b48213a6dc132400d68b3c2685182`，fresh restore `d5f276cce0420015129475dd8bdaf5f988f9741f999dca3a01c2b4dc9be3ba9e`；比较 `2c07a6603be0ea89806fb7725d9ed4c0a29c53ddfc0984cf1763c448154f45dc` 全项严格一致。 |
| 原生公开推理 | `f99cfd9668f4ab1e36c76e51fb636976218ff9d161656f33fd4f7b0a33b01234`：实际模型及 adapter 加载，4 个请求/16 个 greedy tokens，记录 logits 与完整 KV cache；候选对拍未执行。 |
| 原生 RNG 恢复 | 汇总 `bda8ca45308e6f72f4629b113fc115fac69e0551d855d5b44acee1d811ff3047`：rand/randn/非零 dropout 的进程内、保存、独立进程恢复共 9 个阶段通过；不要求两运行时 RNG 字节格式相同。 |
| 候选公开训练首断点 | `7efb6e6d4fc373337faa4384cbf11aeb854ca97f37cd01eaa7b2eefb02742c4f`：核心编译完成，Transformers from_pretrained 的真实 meta Tensor 缺失；同时观察到 NPU 被发现为 cuda 的设备接口问题。 |
| 候选 RNG / launcher / 确定性断点 | `8e7848e5471eeadd752fa2165603c841dda6c5b5d4aa92fffc65057119da9fad` 缺 NPU RNG checkpoint API；`917b4148d36a288bb6ef894c592a30e638aae350696cd2883939e577b6d5e547` 缺公开 torch.distributed.run；`e289dddff5e921c6ef5cb1f533a44259d27f1bb30fd0a6b0013b30be6eb67bfd` 确定性 setter 是空操作，CANN 独立查询仍为 0。 |
| 两卡 HCCL 微检 | `13538b5180ab2cc6a9a4924a2f56859ddbaca88eae1a73187f58ad23646b9608`：原生 4 项通信通过；候选真实 HCCL 初始化/all-reduce/broadcast 通过，all-gather 因 compat 未分派至已有 HCCL 算子失败。候选已执行阶段 fallback=0，整体微检失败。 |

原生完整 checkpoint 包含 Swift adapter、验证未变的完整 base、optimizer/scheduler、RNG、
trainer step 与数据位置证据。为让 HF Trainer 正确识别 Swift adapter checkpoint，证据工具
生成标准 safetensors shard index，映射到实际已有 adapter 文件；没有修改下游库。
恢复由公开 `resume_from_checkpoint` 完成，插件只观察并核验初始恢复状态及续训轨迹。
未开启 `full_determinism` 的原生对照出现约 `9.54e-7` 的 loss-only 差异；严格比较如实失败，
没有放宽容差。双方公开训练协议已锁定 `--full_determinism true`，之后原生严格比较通过。

上述断点对应的核心 meta/RNG、ACL 确定性、compat NPU facade/launcher 与 HCCL dispatch
修改正在整合验证，不能据源码存在宣称支持。原始证据分别保存在 `$TASK_STATE/delivery/`
的 `runs/`、`inference-runs/`、`api-probes/`、`hccl-runs/` 和 RNG 汇总目录。
真实多机仍缺第二台 Ascend 主机；没有模拟第二主机或把元数据合同检查写成多机通过。

## 公开入口基础能力与当前断点（2026-09-22 核验）

同步基线 `61294cd14673ba60f4072542a2e73ea2c8f8c509` 加各 manifest 记录的 dirty source。
`delivery/foundation-runs/6ed8ba92ad53ecaa5dba198ecbfbfcb6bc85ce60ec3d3e2d545e1711fba767d7`
通过五项真实 ACL/meta 检查（5 executed / 5 passed），并通过真实 NPU 默认 SFRL pool 峰值生命周期、reset-live 及原样 Swift 内存查询。fallback_policy=error、fallback_count=0。
meta 覆盖无存储构造、默认梯度状态、Parameter/clone/detach、assign 与 to_empty；不包含 CPU 模型计算。
峰值由 NativeRuntime 的真实分配/释放事件维护，范围仅直接 raw backend→默认 SFRL pool，排除 workspace、驱动及 HCCL 内存；单可见设备执行，不声明多设备隔离已验证。

已保留的公开 Swift candidate 前置失败：
- `0ad826750685f89973b11e3b85b706641ce42d9fff18cd19d8ca71c2ba1667e0`：Transformers eager 导入仍要求可导入的 `torch_npu.npu_fusion_attention`。compat 仅发布明确 UNIMPLEMENTED 的失败入口，未增加融合注意力能力。
- `67e1933e026844841bf6ccc826fae9ce0d6f4ddb91cb5ada03ce629dd5e738e7`：base 模型加载成功，device_map 为 npu:0；数据预处理 Manager 因 shim 覆盖短 TMPDIR 而报 AF_UNIX path too long。使用已有 JITTOR_TORCH_KEEP_TMPDIR=1 的启动环境选项解决，不修改下游。
- `ffea416466d0f1411b8942cde8999e3847552d488017f7b490932e070cf3d609`：进一步完成数据预处理及模型/adapter 准备；打印模型时 native extra_repr 假定 compat LayerInitializer 有 __code__ 失败。此运行未进入有效训练步，不算 L4。

运行键独立保留；串行复用已完成的 67e193 JITTOR_HOME 仅复用编译产物，训练进程全新，原失败不会升级为通过。内容依赖校验未关闭。

维护回归 `06f9c8c5fb489b5995381d7e692ec6935c6f7f239909e7e2fb959ec1066bdd65` 实际执行 8 项，8 通过、0 skip，覆盖 meta、设备身份、确定性与默认 SFRL 峰值生命周期；严格 ACL 回退策略为 error、计数为 0。修复了仅构造、尚未使用的 SFRL 描述符导致峰值账本永久拒绝的问题。此前 `a13164f914a39b8bfa24aca09e508afb9ec268317b60e4aab9b917bcfd0105c5` 的 7 通过/1 失败记录原样保留，不重标为通过。

compat 默认 `extra_repr` 修复的元数据定向检查 `4feb1222a3f5d30cc4d56cf61a2802c6c0684105bc03d4a3d57d5d240710997b`：3执行/3通过/0 skip，不构造张量或执行模型。

| 新增候选基础能力证据 | 已验证范围 |
| --- | --- |
| RNG 汇总 `0169ea0215bcfc4a8a5e0546b4956af195df9df70d8ceec392b092c851e34978` | rand/randn/非零 dropout 各自的进程内恢复、保存、全新进程恢复，共 9 阶段全部通过；另一个坏 payload 检查通过，拒绝后状态及后续随机数逐位不变。仅单可见 NPU，不推出多设备原子恢复或完整训练 checkpoint 通过。 |
| truth reduce `91c31989aa3adefd8a1b3ba97a4368efd5fa0ae45d498d3853e93d8fadb42546` | 真实 NPU truth-reduction dispatch、keepdims 与 isin 定向检查 1 执行/1 通过。 |
| bool setitem `52a48eabd0fe9239dcfc2310f4b3adef9815f5e6412385e07aef633ab86b58f8` | 真实 ACL 布尔切片赋值及精确设备数据检查 1 执行/1 通过。 |

这些候选检查均保持真实 ACL 驻留、error 策略和 fallback_count=0。RNG 各阶段仍是独立新进程；串行独占借用已完成运行的原编译缓存不改变运行键、产物校验或失败结论的口径。

候选公开训练仍未通过 L3/L4。`eafd149a7d2b1d87c48ee283c132a69d07972808986ece68b9a06f807fe5c0b3` 的 optimizer.param_groups 泄漏内部 native Var、`119b8a297e14660cc11f25ced08d84362afd13a67ebc608107be09671d96938d` 的恢复后 step dtype/device 失败均保留。修复后的真实 NPU optimizer 检查 `0266c5b109b13a5430ab6ae11f22147211862936a344785b93645dda5515d76a` 通过；与原生 `8b135db2331a8087687de74f961f09575a95569ec99d1754d9eb907afa9069c3` 的比较 `4472300f7709b3c7f67f3777fcbdd1f38cd3c8f58f8ec360a395f68f8ead87f8` 通过，30 个浮点值最大绝对差 `1.49e-8`，跨运行时比较按原定容差，双方同进程新 optimizer 恢复后的继续更新各自精确一致。维护回归 `1dedc18ab4b82958aa923d665e6ec24441e6a6abcf47ae697cb8b0afb4cc30aa` 为 1 通过/0 skip；这些不替代全新进程训练恢复。

公开训练 `04ff0f3c96bae2adee8bddd7ede4560fdc131b7d1af6f0fe5e3c6d65fa20ed6c` 随后因 scheduler.state_dict 含函数对象失败。LambdaLR/MultiplicativeLR callable 状态保存恢复、initial_lr 与真实初始化/get_lr 上下文修复的 7 项纯元数据检查，原生 `ece21d60b3ac1600c62287ce05179a792d00f7c5a5990467fa154edefc6afc95` 与候选 `c758d788387e501fad0a600b9fb09cbb255ded9e14b73f800e6ec33ab9064754` 均 7 通过/0 skip。未构造模型或张量；公开训练仍需新运行验收，不能将该失败重标成功。

独立单 NPU 公开推理已通过 L0/L1/L4：原生 `01b7fedc778aea38328103c079416bbd1bcafdfb0d30041dcb7cb499f3ce7179`、候选 `7af2b00b150da0d59f07d0ce2ccf62d8f2b12011f568114c9b2a8072206ea532`，审计键 `862472f6603a4ec4e8df0e970701464d49525f2a3d5967034bb1772a7c6d8d8b`。v3 协议按实际运行时分类持久参数与派生非持久状态；锁定模型及 adapter 加载完整检查、4 个请求的输出结构与 41 项张量快照通过，logits/cache 最大绝对差 `1.1920928955078125e-7`，包含 EOS 的真实 greedy token ID 精确一致。候选真实 ACL、error 策略、fallback_count=0；原生无统一计数器，保持 unknown/null。L2/L3 训练项不适用于该推理轨道，L5 未运行。此前 `77481e34fdd42326b13a82307a672e0da77010693ca29f7e203192b0251b9f64` 的 BOOL 切片赋值失败及旧比较失败保留，不以新协议改写旧运行结论。

真实多机启动合同元数据检查 `d8bcd38e2bc5800ca8aec9a58516fa468aee0b3504af288b27a67d65596efb66`：6项通过，只使用标准库；实际单主机资源检查退出78且未启动worker。真实多机仍资源阻塞。

## 最新增量证据（2026-09-22 核验）

以下各项保留自己的 manifest、dirty source 和失败产物；同步基线仍为
`61294cd14673ba60f4072542a2e73ea2c8f8c509`。运行完成、定向回归通过和完整轨道验收
分别记录，不以局部修复结果推定公开训练或恢复已通过。

| 项目 | 已验证结果及边界 |
| --- | --- |
| 两卡真实 HCCL | `fbb22a3a15f2e2aebe38f8b2da34c4f7b05291a92dbed4e9c0cf1fc369b0722c` 的 `attempt/comparison.json` 通过：两 rank 实际执行 all_reduce、broadcast、all_gather、barrier；候选严格 error 策略、fallback_count=0，原生 counter unknown/null。它只证明单主机两卡通信，不证明分布式 Swift 训练、checkpoint 或公开 launcher 通过。此前 `13538b…` 的 all_gather 失败保留。 |
| 合并维护回归 | `3de8eff27405164f16960255b784fdf19de0665925a344e90bbf1346325406cc`：11 collected / 11 executed / 11 passed / 0 skip，error 策略、fallback_count=0；postflight 确认源码及包装器未变。覆盖 meta、设备身份、确定性、默认 SFRL 峰值、truth reduction、bool 赋值及 AdamW 公开状态/恢复。 |
| 布尔索引及梯度 | `9f47403bd65f5872e15f8549f4388bbfa63ca3c8dcc0820018a8c1737e32c553`：真实 ACL `test_npu_single_bool_index_with_full_slices` 1 执行/1 通过/0 skip，error 策略、fallback_count=0。该定向结果不代替完整训练回归。 |
| 公开训练的当前失败 | `4a789c193c61b32689e362f4a232b1ae01c9eaf3e749d8fa68db444c0df099c2` 已执行两次实际 optimizer 更新，step 2 保存后 `on_save` 观察器拒绝 `parameters/value2` 非 NPU 驻留，整体退出 1。不能把已经生成 checkpoint 文件视为完整保存/恢复通过。 |
| 前两步离线诊断 | `93cc0163388ddd787a57a29f89e429168e9f39d1074fca6cdd80363695b9b4dc` 只核对该失败运行已存在的两步轨迹：29 初始参数、8 可训练参数，键/形状及实际输入等精确结构检查未发现差异；loss、logits、梯度、参数、更新和 optimizer 有逐类误差统计。报告明确 `numeric_pass_fail=not evaluated; no predeclared cross-runtime tolerance`，没有数值通过判定，也不授予 L3/L4。 |
| 序列化驻留修复回归 | `d98d50adf6b11ebe82aa6cc1c758d91b2bfa8062ac8b8791b7525bea20cbf3c1`：真实 NPU `test_npu_serialization_preserves_live_state_dict_residency` 1 执行/1 通过/0 skip。在 error 策略和 forbid_backend_fallbacks 下执行，保存的计数证据为 `fallback_delta=0`；不把增量字段误写成未记录的绝对计数。仍须重新执行公开训练保存、全新进程恢复与续训。 |
| 真实尺寸原生公开推理 | `5f47d3f5d51baca172f57c1a9c44777fa85d1b321cbf4f771f617c7cfe8fcd26`：约 1.1B FP32 模型、Swift q/v LoRA，经真实公开 CLI 完成 4 请求/16 forwards/16 greedy tokens，各 prompt 521 tokens，退出 0，完成文件及快照哈希齐全、postflight 通过。只建立原生正确性执行证据；候选对拍和独立稳态性能尚未通过。观察器含同步/D2H，运行耗时不是 L5 结果。 |

原始证据分别位于 `$TASK_STATE/delivery/` 的 `hccl-runs/`、`maintenance-runs/`、
`api-probes/`、`runs/`、`partial-training-comparisons/`、`real-size-inference-runs/`。
真实尺寸 fixture `fc8a8deae25e7fcc7e97ae0cb6a390926a95c9c2a3c3f36a4064f2622ddedc30`
复用历史锁定初始参数，原生 NPU 构造并导出；独立计算的非持久 buffer 未注入。
它的 tokenizer 词表为 32000，不能复用 tiny case 的 128 词表解释输入。
这些新增结果不改变 tiny 公开推理已通过、公开训练 L3/L4 未完成、四轨 L5 未通过及
真实多机资源阻塞的总体边界；CPU 模型测试仍未运行，也不声明 CPU 支持。

## 原生单机多卡公开参考（02:10 更新）

2/4/8 NPU 的原生参考均经过真实 `swift.cli.main → torch.distributed.run`，每 rank
记录实际 HCCL、设备绑定、完整模型/梯度/优化器/RNG 和 checkpoint。全局 batch 8，
每 rank microbatch 1，分别累积 4/2/1 次；三步控制运行及 step 2 保存后的全新进程续训
均执行。严格恢复比较包括 raw microbatch loss/logits、全部可训练梯度、参数/update、
优化器/调度器、RNG 及实际下一批输入，未放宽容差。

| 原生规模 | 控制运行 | fresh restore | 严格恢复比较 |
| --- | --- | --- | --- |
| 2 NPU | `05f8013d9a5b0396940e15e8da1cf56b58eb77e5d63adf2fcdc5cc60d0d51058` | `6b799fa1a1357e2886309127f217d19245f4496b2276e08524676c4a0742184b` | `7c5d52f528a321c033515f6fd9d124e1fb86b5c368f78f1288c0cd5f0d703d22`：通过 |
| 4 NPU | `e30f43651fe2511c12d0c9f47f559edf777f9eee8b16e46d56a34885f86b3daa` | `daedc1022a4d40ece9c7f490aa98d8ed4ea19eb39a805e11fc47df6ab4ffa30b` | `0f18b4ff9fa535d9dec9b5e799eb54c653ad3a17deaedbdb56b36ddeb0a74117`：通过 |

八卡控制运行 `f384f5d90bf6815aacb2ebaaf7581c12d46eb6c6d5664cc3fcfb5dfbfae2e42f`、
恢复运行 `9befd9d3dee6ec45af64caa5d38d871a73e9935da2a48d466f7b114c18579f85` 均通过；
严格恢复比较 `86cebe5de67d423cf9edcb4ac2fc42e0be211465f4573909459d6f882bdf291c` 通过。
八卡前置 HCCL 微检 `6978c37ee93ea34ee38e21855d215a0876d851b5bb3daac39cca0b1650962d3d`
在所有 rank 完成四类通信。八卡窗口独占设备，候选计算在 worker 全退出后才恢复。

实际分区另行对照单卡的固定 token 输入：rank r 在第 s 个全局步骤、第 k 个本地
microbatch 消费记录 `8*s + world*k + r`（从 0 计），控制运行共同覆盖 0..23，
续训覆盖 16..23。各 rank 同步后的梯度和参数一致。这是输入分区与同规模恢复证据，
不是跨 world 的数值等价结论，也尚未覆盖同步前本地梯度。

启动时须使用 CANN 9 接受的 `HCCL_DETERMINISTIC=true`，并采用短的作业本地
`TMPDIR` 避免 AF_UNIX socket 路径上限。公开数据集 barrier 先初始化 communicator，
之后 Transformers 构造 Trainer 时将环境变量写为 `1`；起止记录保留实际变化，未修改
下游或截获写入。成功结论不推广到后写值之后新建的其他 process group。此前错误值
与过长临时路径的失败分别保留在 `012da5ed…`、`0cdb7f45…`，不记为通过。

详细证据见 `$TASK_STATE/delivery/native{2,4,8}-public-result.md` 与各 run 的归档。
候选单机多卡对拍仍未完成；真实多机继续受资源阻塞。

## 保留的真实尺寸测量

已有测量完成 1,100,611,584 参数的 FP32 Llama / Swift LoRA 训练，563,200 个可训练参数，batch 1/seq 512、eager attention、HF32 关闭。3 次预热后测量 10 次完整 zero_grad/forward/loss/backward/AdamW/device-sync；没有把 D2H 快照放进计时。177 项 loss/最终参数/梯度产物通过精确 key/shape/dtype 与数值比较，性能产物归一化误差容差为 **0.02**。时间统计已由原始样本重算。

| 指标 | 原生 torch_npu | Jittor ACL |
| --- | ---: | ---: |
| 中位训练步时间 (s) | 0.1469488514 | 0.6504897987 |
| 最小时间 (s) | 0.1463247468 | 0.6423129011 |
| P10 / P90 (s，线性分位) | 0.1467983386 / 0.1471413851 | 0.6438713219 / 0.6545085962 |
| 输入 token/s（512/中位时间） | 3484.206 | 787.099 |
| loss token/s（511/中位时间） | 3477.400 | 785.562 |
| 结束时 allocated/live (B) | 4,475,036,160 | 4,477,397,504 |
| 结束时 reserved (B) | 8,023,703,552 | 11,530,141,696 |
| 预热后峰值 allocated (B) | 7,316,457,472 | 未提供 (None) |
| 预热后峰值 reserved (B) | 8,023,703,552 | 未提供 (None) |

候选中位耗时为原生的 **4.42664 倍**。这组数字只证明所列 case 和测量协议的历史结果，不表示性能领先、达到额外速度阈值或四轨 L5 已验收。候选首个预热步约 **123.7077 秒**，冷启动成本单独保留，不混入稳态中位数。全部 10 个样本及 3 个预热时间保留在原始运行报告中。

原生内存来自 `torch.npu` allocator，峰值在预热后重置；候选来自 `jt.core.device_memory_used/reserved(0)` 同步边界样本，峰值未知，不能据此比较峰值节省。候选真实 ACL 身份、warmup/final 及每步 loss 驻留、error policy 和实际 fallback 计数 0 均通过。被比较的最终运行已在同步前创建并物化返回的 detached loss，避免 lazy 值尚未驻留造成错误证据。

## 已验证修复与维护边界

1. 核心两处 atomic 计数直接流输出在 GCC 10 下产生歧义，改显式 `.load()`；相同 translation-unit 检查及后续完整核心构建通过。
2. accelerator 静态初始化触发 sync_all，访问未构造的全局 fetch 队列。两条队列改由 NativeRuntime 首次访问构造，保留 pending/deferred 生命周期和 cleanup 顺序；结构检查及真实 NPU 回调次序、重复 sync 不重放、第三次 fetch 回归通过。不据此宣称任意回调重入/退出竞态安全。
3. 维护的 shim deploy 补齐缺失 Torch distribution 元数据；未私造兼容包。metadata API 2.11.0 与默认模块 Jittor 版本不同，属于现有发布策略。
4. compat `_ip` 改读取真实 requires_grad，修复 host-to-NPU copy_ 在 no_grad 中解冻参数。8 组合 Parameter/Linear、host/NPU、冻结/可训练及 connected inplace 梯度回归通过。没有泛化到所有 `.data` 或启用梯度时的冻结目标写入。
5. runner 显式将 from_numpy 输入迁至 NPU，保留核心对混合设备的拒绝。真实 Embedding 回归通过。
6. `torch.get_num_threads()` 改委托核心 OpenMP 查询，修复 CPU 数量 192 误报为线程数的问题。真实 OpenMP 2→3→5 及委托检查通过；setter/interop 未因此获得实现。

5 项真实 NPU 回归、17 项证据 metadata 及 2 项线程检查已有通过记录。已有门禁按白名单执行：结构目录包含 CPU 前向/梯度/对拍，不能整个运行并称纯结构。选中的 C++ ownership/OpenMP 检查是元数据/结构验证，不是 CPU 模型测试。

**已有定向门禁：** `checks/36933c7f2428b9a58b5a6a2dbf00fef7c9688b54d728b6de4b80a24e1a1c3ccb`：81 collected / 81 executed / 81 passed / 0 skipped，17.80 s（不含首次编译）；真实 NPU 回归期间 `fallback_policy=error`、核心计数 0。覆盖受影响的 fetch 生命周期、复制/原地梯度、设备迁移、线程查询、多步/性能证据合同、当时的文档及后端结构检查；当时布局检查和 `git diff --check` 通过。这是原基线记录，不代表当前同步提交或本次新验收合同已经重跑门禁。

## 可追溯证据

下列键分别位于 `$TASK_STATE/runs/` 或 `$TASK_STATE/comparisons/`。原始路径、脚本名及产物内历史等级标签不改写；下表按工作负载识别，验收等级以上文统一刻度为准。

| 角色 | 完整运行/比较键 |
| --- | --- |
| 单步 oracle | `169c603c50d5e964bd72ecc42b93add81d6f85219a51baad58025727c6d61050` |
| 单步 candidate | `7fe94c94b5c193995015dd68d4b090047378cf50dbefdd9d39a68ab73bcde52b` |
| 单步 comparison | `7cd7db6a32a6bbe9d64e3bcd004cc00a2c6df85e619a0f6dcbd4e0327dc2829e` |
| 三步训练 oracle | `26cd043896ff6ec1444c5326ebb3ca695a7e65798e06e77916bc4763e45e1c4e` |
| 三步训练 candidate | `6edc0ea84b3dd7a0002adbb6553817ec3ea8794ed3dcc6f88417157e75e5a024` |
| 三步训练 comparison | `9f611125ddda3f8cff0e8bbd73b4d97b727d6d6f710c7312d8a8f2520cb6f85a` |
| 真实尺寸测量 oracle | `b6b279d397337080bb94a6321a68d57ab0f5a1df4d348d0f369725c7f0c154db` |
| 真实尺寸测量 candidate | `8adf387f8708d531c0e2514c2fc216032551e083f327669a355a585bf8028e44` |
| 真实尺寸 measurement comparison（修正容差字段后最终结果） | `b8e312b7b1d7b4dc38342ee7fb689f7658bb2ba71e62c7ca8c361c28e09ecf9d` |

单步 oracle/source 指纹为 `e0ec299638c5c4126473c618dfa4cbc4934cf291632c03b7a0200c88f67a4d54`，候选为 `7d419fee6bce4c4665e601e8eaf73d76430c4dbaaf2d9eb116be45b40d108d78`；保留原生参考与修复后候选各自 manifest，不声称源码相同。三步训练双方 source 指纹均为 `f77b84977411919a22af8dcbebd7264a24df6a858c96aeed91da2ad29c688f1b`。真实尺寸测量双方 source 指纹均为 `e9eb4f23ba9728c7dbb8d32c87fba60f50b73c0e965bd844c29a029fe1acf215`。比较 manifest 还哈希实际比较器和全部输入产物。

## 历史用例重放接口

先检查运行键和产物完整性：没有相关变化就复用，不能为了重述等级重复模型计算。`TASK_STATE` 指向本次未版本化证据目录；`ORACLE_RUN`、`CANDIDATE_RUN` 取上表对应完整运行目录。所有 Python、编译和测试均通过 `srun --jobid=720 --overlap` 执行；登录节点仅做 Git、文本与编排。更改源码前停止同一工作树已有模型/编译。

```bash
# CASE 取本报告三个显式 case 之一；新键确需运行时先 oracle，再加载完整 weights 运行 candidate。
srun --jobid=720 --overlap bash -c \
  'source "$1/host-env.sh"; bash "$1/run-case.sh" torch npu "" "$2"' \
  bash "$TASK_STATE" "$CASE"
srun --jobid=720 --overlap bash -c \
  'source "$1/host-env.sh"; bash "$1/run-case.sh" jittor npu "$2/result.weights.npz" "$3"' \
  bash "$TASK_STATE" "$ORACLE_RUN" "$CASE"

# 历史脚本名不代表当前等级；compare-l3.py 比较的是三步训练，现归 L2。
srun --jobid=720 --overlap "$TASK_STATE/host-oracle-python" \
  "$TASK_STATE/compare-l3.py" "$ORACLE_RUN" "$CANDIDATE_RUN"
srun --jobid=720 --overlap bash "$TASK_STATE/run-final-gate.sh"
```

单步、三步、真实尺寸对应的历史比较器分别为 `compare-l2.py`、`compare-l3.py`、`compare-l4.py`。包装器保存精确命令及 manifest；失败退出码、缺产物、非有限值、缺梯度、dtype/shape 差异及驻留不符都拒绝。相关源码或协议改变时生成新键并保留原证据。这些内部 case 的重放不代替四轨真实公开入口验收。

## 公开训练对拍及恢复断点（04:25 更新）

新 `public-training-parity-v2` 协议在运行前绑定分类容差、比较器及说明文件；
初始序列化参数、optimizer step 和组控制状态精确比较，Adam moments 按预声明容差比较。
候选 `f976d62b228464b1bfb5db14d048875eba67d2b43e64e68140a83fc3e392a332`
与原生 `9a6afcfa43e316f04b2bb1b8f505e123a4a25fe78adf52925d3f67610ab40395`
均完成真实公开 Swift 三步训练及 checkpoint-2/3 保存。离线运行键
`16f3b0b36ad4e1756300462c4866cc205d06e7cb3f74c9585a90236342d5fa5a`
对拍通过：24 个固定 microbatch、全部可训练参数梯度、更新及优化器状态；候选记录
ACL、error 策略、fallback_count=0。输入为整数 token，输入梯度不适用。
这补齐公开训练的 L1/L2 证据；因 L3 尚未通过，不宣布训练 L4 完整通过。

原生 fresh-process 恢复 `a3f7f0bfc7c5396dd6e6ceb83d123ed99f5cc6d3e42fb39956b48c6d3b907c36`
经离线键 `24f86a62135fc2cc0598a3f028d64f672a477b3a5e0bcf089844640263922969`
证明第 3 步与不中断参考精确一致。候选恢复
`474f9c6f6709ab4096738503bbf4db0d13ab2de772a7ca1924b27a382848b2d9`
在恢复 host RNG 控制状态时报 `RNG host engine state mismatch`，整体失败。
独立宿主控制解析复现 `ccfdf21578f79a10171c75b3919c4d10fd6ba71ef7399fed1685edf4b1b5e300`
确认原解析拒绝有效状态、显式消费分隔空白后接受；它不是 CPU 模型验证，也不代替修后 NPU 恢复。
此前 `0169ea…` 九项 RNG 结果覆盖 NPU counter replay，并未覆盖这条 host engine 恢复分支。

单卡 hook 微检查 `8f182ec2b3e2aafa1e2ddb31506af90fbfb5800b4d57efca2533d96564f48b95`
中无 hook 的两步训练通过，但注册 hook 使同一 Parameter 的真实 leaf 标志由 true 变为 false，
观察组失败；后续两卡 hook 阶段未运行。不得将 plain 阶段或 HCCL 通信通过改写为分布式训练通过。

## RNG、真实尺寸数值及 leaf hook 增量（05:00 核验）

以下结论限定各运行 manifest 所绑定的实际源码，不将已通过的单项检查推广为完整恢复
或分布式成功。CPU 模型计算仍不在范围；RNG host 控制状态解析不属于 CPU 模型验证。

| 项目 | 实际证据与结论 |
| --- | --- |
| host RNG 恢复修复 | `25d0c76911f74f826502852e94bf65cb4a80b1054a44eaabed3b00aaacf0f32a` 通过：真实 checkpoint 的 torch.load / torch.random.set_rng_state 控制路径、自身状态往返和三类坏 payload 拒绝原子性均检查；ACL RNG 状态不变，error 策略、fallback_count=0。维护回归 `7fdf95ddd29a44cc56bb238194ebbcd29bf1af480cdb8534e95001eb21bc74f2` 的 `test_cpu_state_restore_does_not_reseed_acl` 为 1 执行/1 通过/0 skip，初始计数 0 且 fallback_delta=0。完整 fresh-process 训练恢复仍须重跑，不能据此声明 L3。 |
| 真实尺寸公开推理执行与失败 | 原生 `5f47d3f5d51baca172f57c1a9c44777fa85d1b321cbf4f771f617c7cfe8fcd26` 与候选 `d47623e80206f235e482634be57b0937a845f9f855874acdc1a9aa074f5f22b7` 均完成约 1.1B 公开推理、4 请求/16 tokens。但严格比较 `d69c8471117551bf55972e2eee631354f3f23c67d45cda161de5ffaa7ed4b4a3` 退出 1：首个 logits 快照在固定 rtol=1e-4/atol=1e-5 下 65/32000 元素超差，最大绝对差 `2.849102e-5`。执行成功不等于 L1/L4 数值通过，L5 不得启动。 |
| 全量离线数值诊断 | `0cf881e2d2e1dfb73f87b56eb24448e8a3ac3f9a42437c07e72c44175af1f775` 只分析现存快照：1618 个张量中 1220 个未通过原严格阈值；序列化初始状态精确、public responses/tokens 精确，元数据未发现差异。非持久 inv_freq 有 18/32 个数不同、最大差 `5.960464477539063e-8`，自身仍在预定计算容差内；最早已观察到的 KV 差异为 layer0 keys，最大差 `9.083747863769531e-5`、218 元素超差。没有逐层 activation，因此不能把“最早可见 KV 层”写成根因层；未放宽容差。 |
| Rotary 原语定位 | `328cd35fc81e78cdd4a1f08f1e3564bd1c52aff3c2ac9c1ad00e8cada76996df` 的双方真实 NPU 角色均执行完成；离线比较 `7b88d9dcaba71c25d6ce0e89393e55e8c442d9b2482f5261e12dc4b1b87e2161` 对 arange/cast、指数、pow、倒数、position matmul、频率拼接、cos/sin 等全部 13 个阶段逐元素精确一致。候选 count0、原生 counter null。该结果仅排查锁定原语路径；不能据此宣称完整模型数值失败已解决，也不能在没有复现时修改 pow/rotary 算子。 |
| 原生 leaf hook 与优化器提交 | `11f793b313d9be9aa75b3146d24f7fc324182dc768e59343c9d693c40782cc00` 已完成核心构建，注册身份及前两次 backward 的 hook/local-gradient 检查通过，但 SGD step 使 Parameter 成为 `grad_fn=fused_sgd` 的非 leaf，整体失败。共享优化器提交边界修复后，`453a803e6b68df16f1b3d62f7e9f4cf4c6c7440a6bacc2343210462491211475` 的原定维护方法 1 执行/1 通过/0 skip，error 策略，初始 count0 且结束 delta0；覆盖 leaf 身份、顺序/替换/移除、局部与累积梯度、更新后继续 hook，以及错误 shape/dtype/device 替换拒绝。非 leaf hook 沿用旧机制，实际两卡 hook、完整公开训练及恢复未由此通过。 |

优化器矩阵的前置失败也原样保留：`6e9911e55790fbcf2e2d541a0613ebd917d88c13ca32c43c53c6d2f66a5b2296`
因对无 momentum 的无状态 SGD 索引不存在的 state 失败；测试改为允许该明确无状态类别。
`0044d0890b76e35635af134bacea67b9e2152208d3a02d08b55eeda27d60fb56`
在 SGD 构造参数 `fused=False` 处失败。现有 shim 构造器不接受该关键字，属于明确 API 缺口；
覆盖便携提交路径可使用现有 parameter-group `fused=False` 开关，不因此宣称构造器兼容。
修正后的真实 ACL 矩阵 `7793dd80adc83474a50f90a097f4ad55ad5edabb50d4632f22e473bc5ac1f96c`
已通过：1 个维护方法、7 个 factory 各 2 次实际更新、0 skip；error 策略、初始 count0 及
结束 delta0。明确断言每类完整 state 键集合，覆盖 SGD 自动/便携 momentum、Adam、
AdamW 便携/fused、RMSprop、Adan 的 leaf/hook/冻结参数/状态策略及移除后第三次 backward。
该矩阵不掩盖 SGD 构造关键字缺口，不代替真实分布式训练或完整恢复。

新增原始文件分别位于 `$TASK_STATE/delivery/api-probes/`、`real-size-inference-runs/`、
`real-size-drift/runs/`、`rotary-primitive-probe/` 以及 `$TASK_STATE/evidence-audit/`。
总体保持：tiny 推理已通过其适用正确性/公开入口；真实尺寸推理数值失败；公开单卡训练
已存在三步 L1/L2 对拍，但候选完整 L3 恢复未通过，故训练 L4 未完整验收；分布式训练未完成，
真实多机资源阻塞，所有轨道 L5 均未通过。

## 单卡公开训练完整恢复（05:12 核验）

共享优化器提交与 RNG 解析修复后的新控制运行，原生
`37ea72fd6cf1bbb2b9911a1d473f666967845935308f9f4262ffe8252e4b0477`
与候选 `7dee183aee0fb4117c0ec3365036e952c5e5ecb000ce5de7ad3048703ec459d1`
均完成公开 `swift sft` 三步训练。严格跨运行时比较
`393d90698d94c489d15a344e5989bd330d54ca469321e1a38e8ec35caf1de784`
通过，未改变既定容差。

候选以全新进程从 checkpoint-2 恢复的运行
`5c9a9d5b7073b04c6005ab0560669a380f21feeb7ad3473280ebbc2d2cb12e4b`
完成 step3；精确续训比较
`3f33178fb0239e408463b362858926a9a198efa49ff19d2916d2b03e0bfc939e`
通过，比较覆盖 8 个 loss、8 个 logits、16 个梯度快照、96 个优化器数组、120 个参数数组、
62 个状态数组及 29 个参数更新，并核验完整 checkpoint 文件哈希、恢复状态与 RNG。
候选实际 ACL，fallback_policy=error、fallback_count=0，实际处理 8 个续训 microbatch。
原生 fresh-process 恢复 `c81dc2e430d5b4c61222d492088ed90789a015603663089e62db61e100a5ae66`
也由 `981578a35a71f30d868547c2aab970e881e2241b78c479eb47b1a1b231a34ffd`
通过同运行时精确续训比较。

这补齐锁定 tiny、整数 token 输入、普通单卡训练 case 的完整 checkpoint 恢复与公开入口证据。
输入 token/label 为整数，不把其不可微梯度虚构成通过；浮点 `inputs_embeds` 的独立 direct API 证据见 2026-10-12 的矩阵补行；不推广到公开 Swift SFT 或其他模型配置。
单卡训练 L5 仍未运行，分布式公共训练及真实尺寸推理的剩余门禁分别验收。

## 维护回归、结构范围与多机准备（05:26 核验）

维护的真实 ACL 组合回归
`abffd41cb89d077885ce7ae2f15ead5845c93e299da66b86488ac8acf27d9219`
为 15 执行、15 通过、45 个 pytest 阶段均通过、0 skip，error 策略且初始/结束 fallback_count=0。
覆盖 meta、实际 NPU 身份、确定性、显存计数、布尔归约/写入/索引、优化器公开状态与恢复、
序列化不改变现场驻留、leaf hook 及七优化器提交路径。原生 full structure 命令不是纯元数据：
`3fa2c39b4b6f9d271d9e75c3a4e70cd66c61d0a97fd1eafadc145d2d017a1220`
布局检查通过，但 structure pytest 在约 15% 时已有七项未分类失败，之后进入 native collect-only
编译；审查发现后续含范围外 CPU 数值用例，故仅中断该 pytest。未取消作业 720，没有将中断
或 skip 记为通过。其原因尚未逐项定位。

审查后的结构子集与布局检查
`ed81a9b938a7fa6f0afa7e37c0362325f94823a0fb819e81e65f873bce47d59d`
通过：38 执行/38 通过/0 skip，另有 1 个显式不选择的独立 C++ 编译项，不计为通过。
子集限定 holder 源码归属、fallback 策略、ACL 结构边界和文档结构；不声明全仓结构通过。

双卡原生公开 v3 证据已加入每 microbatch 的八个 LoRA 本地梯度以及完整输入可微性清单。
控制 `040da30a6c84e9a5f7adfcc5fa0b937dbd14ffc21224ef8c57a9dc67955fff98`
与新进程恢复 `e355fd1965719d7d51e04f864c506210aad3be82f53a59bfb75eb92b489c976c`
由 `c3304c115c84bec9b1490e0cf63b4599947c3c4ec824812e7aebfc71a8a3ebb9`
通过各 rank 精确续训比较；候选公共双卡尚未通过。v3 比较器的 20 项纯 NPZ/元数据正反例
`63b74f983ce11b558ad9276c0704cfcdfbf25b592f4a1ab502d4096e7615c787`
通过，不能替代真实分布式梯度次序验证。

真实多机独立实现已准备在 `$TASK_STATE/delivery/multihost-proposal/after-v2`，重基当前
v3 观察协议，包含真实 Slurm 主机核验、全局 rank 私有缓存、跨主机整池租约、HCCL 微检
及真实公共 launcher、完整 checkpoint 共享路径和逐节点退出核验。元数据检查
`de2c9f4b228cad66a8e7eceeeffbdbe0f9301c6e66eb8d04148ad13833d690a3`
通过，实际 allocation 仍只有一台主机，runtime_execution=false；没有模拟第二主机，
未声明多机 HCCL、训练或恢复通过。解除阻塞需要两个实际 Ascend 主机，每主机至少两张设备，
并实测驱动/CANN/网络和共享文件系统锁一致性。

## v3 完整训练证据与 v4 推理断点（05:50 核验）

以下是各 manifest 所绑定源码/协议的结果，更新前文对应轨道的当前状态；历史失败仍保留。

**单卡公开训练 v3。** 新协议逐 microbatch 记录八个 LoRA 参数 hook 的实际梯度贡献，
并列举全部位置/关键字 tensor 输入的 dtype、shape、requires_grad 和适用性。
原生 `e016f2f04b70328477872ccc53dee1fa4ac7884707a10a93b5850a0bb26b8798`
与候选 `5a590c596d9fb3dba46a573fd54757ddaeb805a079267e3dc7914467bf0a8c05`
由 `7c43d97d69cff0cb25aab81666d1b824a4e5f1f6cb818f70ce23f5f5e5fda8df` 严格对拍通过。
候选 fresh-process 续训 `11ce99d47af1386ac5c4cc9358d92adb6c4795782e9359f2bad292d740234622`
经 `417badbe996de264b994a43dcf47acdf0299f37b64f95699e86a2ad4dca56653` 精确通过；
原生续训 `6025abcb44db8ca606160c16ab8fa37ed3999a98f6125ede65875fda533a70fd`
经 `988b9465c2ab6a861329f58a0c98999f9183d011da9ee6035d80ed03136e8876` 精确通过。
输入是不可微整数/bool，明确 N/A；协议拒绝未覆盖的浮点或 requires_grad 输入，不能声明浮点输入梯度支持。
该结果补强锁定单卡 case 的 L0–L4，未运行 L5。

**原生多卡与候选双卡必须分开。** 原生四卡公开控制
`d147e1bbfccd8a26fff48de2d2a949f66d88bfedeca54e536970b829b269f71a`、恢复
`23a371f2f63ee8fd4c247e2b9cc575fe8dd9b6ebcb3bd2ae0fc436c1317f7d59`
由 `273675978e0423805b86f2df4c9b607cca99c40806b8603b43a915bf01df1d9f` 通过逐 rank 精确续训。
原生 2/4 卡控制及恢复的固定 global-batch 分片审计四项均通过，依次为：
`298dc9f900042898b05bf2be54333545bd1800964679a4d77d4c46e7a597d9d4`、
`34389fd62a1cba88f8bc35542e1b5f96278d4179f783ae8ebdf1a3d2c3171ff6`、
`ca4b33148c9c81c5e04e7e39efc92f4027c6888a7f379f8f19663f9ac81dd694`、
`9fa8efbb844a13ae9eaedc3c309ca3770034b0a84ffda6b44b6ff6faed79b520`。
这些是 oracle 证据，不代表候选多卡验收。

候选双卡 `938ecba806bf8d01c9a7e6de21692617786de9f0d2b4d4147adbc8e9a669aa8b`
完成各角色私有缓存预热及实际 HCCL 双 rank 通信检查，但真实公开 Swift/Accelerate 入口
在导入 `torch.distributed.algorithms.ddp_comm_hooks` 时失败。兼容命名空间修复已应用，
新键 `54683642cdd439e0b5341703117fbbc5900dcd7957dd83c87226663e35354e5e`
已越过导入断点，但在公共 Trainer 的 token 计数 gather 中因裸 `scalar[None]` 被 ACL 拒绝而失败。最小索引入口修复已应用，待新运行验证；尚无候选多卡训练/梯度/恢复通过结论。

**真实尺寸推理仍严格失败。** 显式共同 NPU 初始化 v4 的原生
`146b00bb6140fd8076c3bb16f651d97cf94cc51d40a00eb6b77c76d25c8c1f72`
与候选 `796e39e26587c85945fd4e0e493a59cba3a07173a07606c5b6547455b0cae1b1`
完成执行，但严格比较 `782de967bf7d8a903f9425b8ff8d225612732117f799ae299aa6c1460f4cd284` 失败。
全量离线诊断 `7f63b78b4f92bb96b1c795ea8f2eca0f5b1ba74430384fe9083453f7b37ba827`
记录 1618 个 tensor 中 34 个超出原定 rtol=1e-4/atol=1e-5；初始参数/buffer 精确，
公开响应与 tokens 精确，候选 fallback0，均不能抵消 KV/logits 数值门禁失败。
逐层 activation 诊断 `ccf303a5175cba78e1f4e6cddd1c906c67db6dedaf8022cd957197cefca66059`
确认双方观察前后各自 prefill logits 逐位一致；首个非零差异在 layer0 RMSNorm 输出，
最大绝对差 `9.5367431640625e-7`，该点本身尚无超差；首个已观察严格超差在 layer3 residual 输出。
这是定位证据，不是 Rsqrt 或其他算子根因定论，不放宽验收阈值，不启动 L5 性能。

后续独立 RoPE primitive 诊断使用运行键 `328cd35fc81e78cdd4a1f08f1e3564bd1c52aff3c2ac9c1ad00e8cada76996df`
和比较键 `7b88d9dcaba71c25d6ce0e89393e55e8c442d9b2482f5261e12dc4b1b87e2161`，在物理 NPU 2、
FP32/HF32 关闭下对 arange、浮点转换、指数、标量 pow、逆频率表达式、位置、位置矩阵乘法、
频率拼接、cos/sin 及三个显式算子探针共 13 阶段逐位比较，均为 exact、0 个差异、0 个容差失败。
该探针不执行模型，也不复现模型 buffer 初始化状态；因此只说明这组隔离输入上的 primitive 表达式一致，
不能解释 layer 3 residual 差异或推翻真实模型严格失败。原始文件位于
`$TASK_STATE/delivery/rotary-primitive-probe/`。

**默认设备 API。** `fe7a1203de38cbd308b3c60b7b207b384d262f0f13d293b5e0d9eef92a973c72`
真实复现 `set_default_device('npu:0')` 后 getter 错报 `cuda:0`。修复位于 compat 的默认设备
报告路径，复用 ACL 类型解析并遵守实际 factory 上下文，不修改模型或 observer 掩盖错误。
定向真实 ACL 维护回归 `3650bcf86c25136e038a93482eb96394b4168afbb017c5db902e0e02b4a55376`
为 1 执行/1 通过/0 skip、初始 fallback0 且 delta0；CPU/meta 分支只查元数据，不执行 CPU 模型。

截止本节：候选双卡新运行、候选四/八卡完整 public 路径均未完成；真实多机仍缺至少两台主机。
所有轨道 L5 未通过。原生 torch_npu 无通用 fallback 计数器，继续记录 null；
候选 error+0 与真实 NPU 驻留证据仅对各自已经实际验证的运行生效。

## 2026-10-10 增量：BERT 新协议完整恢复通过

BERT batch=2 新运行在每个 epoch 内形成三种不同输入批次。原生 job 2349 与候选 job 2352 均经公开 `swift sft` 从头训练 12 步并写 checkpoint-4；随后各自在独立新进程恢复并继续到全局 step 12。两侧分别证明自身末段 loss 窗口低于初段，逐项参数、输入 batch、logits、梯度、optimizer 状态及参数更新的设备观测均为 NPU；候选 ACL 使用 `backend_fallback=error` 且完整阶段起止计数均为 0，原生计数器保持 unavailable/unknown。checkpoint-4 含模型权重、optimizer、scheduler、RNG、trainer state、训练参数及 tokenizer/config。166 组 checkpoint 模型/buffer/optimizer 张量在新进程恢复时逐值相同；恢复的首批 input_ids 与 checkpoint 前一批不同且与连续 step 5 相同，step 5–12 的前向、梯度、optimizer 与更新参数在同一运行时内精确匹配连续控制轨迹。两侧 12 个实际 batch 和初始参数/buffer 跨运行时精确相同。浮点跨运行时差仅记录为诊断，不用于训练验收阈值。

原生模型产物的第一次 checker 把 `checkpoint-4` 错识为 `checkpoint-004`；第二次 checker 错要求平衡数据的 labels 在相邻 batch 中变化；一次只读重检又因 evidence symlink 布置错误失败。上述 harness 失败均保留在各自 run manifest；没有运行对应 ACL 候选。修正后 job 2351 的原生只读门禁、job 2352 候选恢复和 job 2353 跨运行时初始权重/batch 身份审计均通过。原有严格比较失败不变更判定。L5 真实尺寸性能未运行；此配置不能推广到其他模型或 tuner。

## 2026-10-07 增量：Qwen3 SDPA 训练失败与公开推理通过

本节同步到 upstream `2.0-refactor=9e37dc025bc2e3e94f59e04a8fa5004e934e6b83` 后记录；集成基线为 `a53220d37640458b8fb48e51d1877791990dd963`。原始产物未版本化，位于 `$TASK_STATE/runs/qwen3-sdpa-public-r3-20261007/` 与 `$TASK_STATE/runs/qwen3-public-infer-nativeckpt-r1-20261007/`。

| 单 NPU 轨道 | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3 tiny 公开 SFT，FP32、Swift LoRA、bool-mask SDPA、batch 2、3 步 | passed（实际公开构造） | passed（同权重 forward） | **failed**（第二次 AdamW 更新/参数轨迹超出预置阈值） | not-run | not-passed（L2 前置失败） | not-run |
| Qwen3 tiny 公开 SFT，新收敛合同、FP32、Swift LoRA、bool-mask SDPA、固定 8 行、batch 2、连续 32 步并从 checkpoint-16 新进程恢复 | passed | passed | passed | passed | passed | not-run | 新键 `qwen3-sdpa-convergence-resume-20261012-r2`；native job 2532 → strict ACL candidate job 2534，均通过各自连续与恢复 loss 窗口。初始全模型/缓冲区及训练权重分别精确锁定，batch identity 精确；同运行时 checkpoint 边界及续接通过。跨运行时浮点只作诊断，见 2026-10-12 增量。旧三步 strict comparator 的失败保持原判，不调整其容差或结论。 |
| Qwen3 tiny 公开推理，FP32、Swift LoRA、greedy 4 token | passed | passed（完整加载参数、logits、token ID） | not-applicable（推理） | not-applicable（推理） | passed（`swift.cli.main infer`） | not-run（tiny 且无稳态协议） |
| Qwen3 tiny 公开推理，FP32/SDPA、Swift LoRA、确定性 beam search（4 prompt、4 token、`num_beams=2`） | passed | failed | not-applicable（推理） | not-applicable（推理） | failed | not-run（tiny） | 新键 `qwen3-public-infer-beam2-world1-20261012-r3`：worker 预检 job 2605、native oracle job 2606、native gate job 2609 通过；实际模型目录 `base`、Adapter 目录 `checkpoint-3`，离线无下载。Native 在 Ascend910B3 完成 4 prompt/16 token；strict ACL candidate job 2610 完成模型加载及首个前向后，在 Transformers beam-search stopping criteria 调用 `Tensor.tolist() → numpy()` 同步时失败：融合图报告 ACL 未注册 `Mod` launcher，`backend_fallback=error` 依约阻止 fallback。L0 构造通过；beam 解码 L1 首个失败；L4 公开入口未完成。未完成 native/candidate 全量比较，未运行 comparator（job 2611 因 candidate 失败取消）。Gate jobs 2607/2608 对正常的两个 Torch 进程记录/argv 作了过严数量假设；job 2609 修正后只读门禁通过。诊断键 `qwen3-public-beam2-tolist-mod-diagnostic-20261012-r1/r2`（job 2612/2613）中原生与严格 ACL 对连续和简单切片 int32 tensor 的 `.tolist()` 均通过（fallback 0），未复现 beam 生成时的失败形态，因此不把问题归于通用 `tolist`。原始运行清单、快照、日志和门禁 harness 失败均留在 state；未修改源码或重判历史 greedy 通过。ACL `Mod` core 修复方案待用户确认。 |

**公开 SFT 失败仍按原结论保留。** 原生 job 1608 与严格候选 job 1609 都真实执行 3 步。候选使用 ACL、`fallback_policy=error`、`fallback_count=0`；两侧实际 bool attention mask SDPA、batch、参数、forward logits、梯度和 AdamW moments 均在 NPU。固定容差文件 `public-parity-tolerances-v1.json` 未改：forward logits 最大差 `2.16e-7`、梯度最大差 `6.61e-8`；优化器更新 step index 1 失败，`lora_A` 对应最大更新差 `4.0270388e-6`，最终参数差 `4.0754676e-6`。诊断表明一个近零梯度误差被 AdamW 归一化放大；未调宽阈值、未改源码。optimizer scalar step 状态按原生语义在 CPU，其余 moment tensor 驻留 NPU；若准入要求每个标量状态也必须在 NPU，此项另外不满足该更严门槛。详细逐项证据见运行 manifest 与 `comparison.json`。

**独立公开推理通过。** 新运行键 `qwen3-public-infer-nativeckpt-r1-20261007`，oracle job 1613 先于严格候选 job 1614；候选之后由 worker 比较 job 1617 对拍。两侧从相同 Qwen3 base 和同一份**原生** SFT checkpoint-3 adapter 加载，使用四条固定请求、FP32、SDPA、greedy 生成 4 token。公开 `swift.cli.main infer` 两侧均成功；全部加载参数、16 组实际前向输入、16 组 logits 和四组生成 token ID 精确一致，37 份 snapshot 完整，logits 最大差为 0。候选 launcher/child 都报告 ACL、严格 error 策略、fallback 计数 0；两侧模型参数、forward 输入与 logits 的审计设备均为 `npu:0`。原生 fallback 通用计数器 unavailable，记录为 unknown。候选首次构建使用隔离 JITTOR_HOME；推理 job 1614 的墙钟包含 JIT 编译，不能作为 L5。比较器 job 1615 因系统 Python 缺 NumPy、job 1616 因读取 Python 字面量时误用 JSON 而失败，均在比较前失败并保留；job 1617 使用 oracle Python 与 `ast.literal_eval` 后通过。这是 Qwen3 tiny 推理路径证据，不覆盖官方大权重、其他模型/tuner、训练 checkpoint 恢复或性能门槛；推理通过不改判同模型族训练 L2 失败。

完成推理对拍时，Slurm 作业 1613–1617 均已退出。个人 fork 远端仍在 `a4fa5b2801bef077b54d9bf669c08c1b3c6b4b37`；已获授权的普通推送再次因未配置 HTTPS username 凭证而失败，没有读取或保存 token，也没有 force push。该认证待办独立于验证结论。


## IA3 两卡公开 SFT optimizer/device 审计（2026-10-08）

基线为 integration HEAD `fbe0b1d1b29d3f35279c71c3591d7388c16a4b19`，upstream `2.0-refactor` 已同步到 `0b8850c3d1c48a284c81f0395bb05647f521a218`；工作树开始时 clean。远端个人 fork 的跟踪 SHA 上次确认为 `a4fa5b2801bef077b54d9bf669c08c1b3c6b4b37`；本轮针对 fork 的 live refspec fetch 挂起后中断，未改写跟踪引用。作业、插件与原始张量证据未版本化，位于 `$TASK_STATE/runs/ia3-world2-device-audit-20261007/`、`ia3-world2-device-audit-r3-20261007/` 和 `ia3-world2-fused-optimizer-audit-20261007/`。

| 运行键/作业 | 结论 |
| --- | --- |
| `ia3-world2-device-audit-20261007`，native fresh job 1626；candidate job 1627 | 原生 torch_npu 每 rank 35 参数、4 输入、2 forward outputs、14 梯度均在对应 NPU；42 个 optimizer tensors 中 28 moments 在 NPU、14 scalar step 在 CPU。候选 1627 在 Jittor 首次编译中于 20:22:28 达 1h walltime，未产生张量证据，保留 timeout。 |
| `ia3-world2-device-audit-r3-20261007`，native resume job 1628；条件 candidate job 1629 | job 1629 的两个 rank 均记录 ACL/HCCL world=2、`fallback_count=0`；参数、实际 batch、forward/compute outputs、梯度和 28 个 moment tensors 在对应 NPU，但 14 个 scalar step tensors 在 CPU。该设备审计没有保存 forward/optimizer state 数值，因此不宣称该运行完成跨运行时数值对拍。job 1628 原生恢复 step 3→4 的 optimizer tensors/rank 均在 NPU；缺候选恢复设备配对。 |
| `ia3-world2-fused-optimizer-audit-20261007`，native job 1630；candidate job 1631 | 公开 `swift.cli.sft`、IA3、FP32、真实两 rank/HCCL、单步。原生 torch_npu 与候选 ACL 的每 rank 35 参数、输入 4、forward outputs 2、compute output 1、14 可训练参数、14 梯度及 42 optimizer state tensors 均位于相应 NPU。候选 process end 与回调记录两 rank `fallback_count=0`；原生通用计数器 unavailable/unknown。 |

fused profile 在预先锁定的 FP32 容差下：两 rank 的 14 个初始 IA3 参数精确相等，4 组 batch 精确相等，22 个冻结参数/buffer 精确相等且训练前后未变；14 梯度最大绝对差 `3.725290298461914e-9`，pre/post/final 可训练参数快照最大绝对差均为 0。两侧每 rank 均为 14/14 非零梯度与 14/14 非零更新。worker 数值报告为 `$TASK_STATE/runs/ia3-world2-fused-optimizer-audit-20261007/numeric-audit-v3.json`（job 1665）。

此结果仍失败于严格序列化门禁：原生 `optimizer.pt` 为有效 ZIP（27141 bytes），候选文件为 11612 bytes 且 `zipfile.is_zipfile` 返回 false；独立 worker `torch.load` 报 `Invalid magic number`。候选 optimizer checkpoint 的 state 数值因此未能读取/对拍。插件同时只记录 forward output 的 device/shape/dtype 元数据，没有保存 logits/hidden 数值，所以 forward/output 数值比较未执行。optimizer checkpoint 与 forward 值缺失都不能从参数/梯度/更新轨迹间接推定通过。比较器 job 1660 因 final 路径拼错失败，1661 因无法读取候选 optimizer 文件失败；1665 将可比快照写入 JSON 后，其 stdout 摘要代码 KeyError 退出，报告中的单项实质失败是候选 checkpoint 归档格式。

因此本证据只补强单机两卡 IA3 的 L0–L2 设备观察与部分数值轨迹，不授予严格 L2、L3 或整体 L4；不覆盖恢复候选、推理、其他 tuner/model、L5 或多机。下一断点是用修正后的观测协议保存 forward 数值和 optimizer state 数值，并先修复/定位 candidate optimizer checkpoint 格式；按验收顺序 native oracle 在前。


## IA3 fused AdamW 双卡完整恢复增量（2026-10-08）

运行键 `ia3-world2-fullstate-fused-l3-r3-20261008` 绑定集成代码基线 `a8dbba3984931a699aa814c31bcec7edf09fcb73`、upstream `7a18abf295668d9b19da5fa1657f5606e84b65a0`。Slurm 原生 torch_npu continuous/resume 在候选之前完成；严格 ACL 候选 Slurm 1713 通过公开 `torch.distributed.run → swift.cli.sft` 完成 world-size=2 四步训练，1714 在全新进程从 checkpoint-3 恢复并完成第 4 步。

比较协议的逐次结果完整保留在 `$TASK_STATE/runs/ia3-world2-fullstate-fused-l3-r3-20261008/` 及两个 comparator 目录。job 1715 的首个 checkpoint comparator 因把 shim continuous-vs-resume optimizer tensor tolerance 错设为 0 而报 28 项差异（最大 `4.66e-10`）；协议 v2/job 1721 将该项改回预先锁定的 optimizer tensor tolerance `1e-7`，checkpoint 224 项零失败，但 trace 比较仍把跨 runtime resume 初始可训练参数错误要求精确相等，产生 8 项最大 `1.1921e-7`。协议 v3/job 1724 只将该项改为已预锁的参数轨迹 tolerance `1e-6`，其它精确与容差规则不变；checkpoint 224 项和 trace/device 298 项均零失败，Slurm exit `0:0`。两次比较器失败结果都未修改或删除。

trace comparator 同时检查连续训练与恢复轨迹中的每 rank batch、前向输出、梯度、更新前后参数、最终参数和 optimizer 数值/标量；checkpoint 比较检查完整键、shape、dtype、有限值、adapter/optimizer/scheduler/trainer 状态及两 rank RNG。候选每 rank runtime 为 ACL/HCCL、`backend_fallback=error` 且 fallback 计数 0；batch、forward、trainable 参数、梯度和 optimizer tensors 的物理设备记录均为对应 NPU。原生通用 fallback counter 不可用，记为 unknown。工作负载满足适用 L0–L4；不授予其他模型、tuner、数据/精度组合或 L5 性能通过。

2026-10-08 的 Slurm structure job 1718 在 NPU worker 完成：`tests/structure` 1389 passed、6 skipped、24 warnings，严格 execution 要求开启；`tools/check_repo_layout.sh` 通过。该门禁完成后报告矩阵按上述结果更新。


## ORPO 双卡公开训练与恢复增量（2026-10-07）

既有 ORPO 失败键和 rank 样本顺序错位结论继续保留。本轮只采用修正 RNG/CPU Generator 语义后的新运行 `orpo-world2-rngfix-20261007`：锁定 tiny LLaMA、Swift LoRA、FP32、两条偏好样本、seed/data_seed=42、HCCL world-size=2、四步公开 `swift.cli.rlhf --rlhf_type orpo`。原生公开轨迹 oracle job 1390 在先；候选 job 1406 完成 continuous steps 1–4 和 checkpoint-1..4，两个 rank 都报告严格成功及 fallback=0。

跨实现比较 job 1407 对每 rank 四步 batch、21 冻结参数、1 buffer、28 梯度/更新、adapter、optimizer、scheduler 和 trainer；`failure_count=0`，batch/frozen/buffer 精确，28/28 可训练张量每步都实际更新，最大梯度差 `2.61e-8`、更新差 `5.60e-8`、参数差 `1.24e-7`。恢复 job 1408 在全新双 rank 进程从 checkpoint-3 继续到 step 4；job 1409 将其与候选 continuous 轨迹比较，`failure_count=0`，数据/冻结状态精确，梯度最大差 `1.49e-8`、更新最大差 `1.11e-8`，adapter 差 `1.11e-8`、optimizer 差 `3.73e-9`，scheduler/trainer 与双 rank RNG 状态精确。

复核还发现 candidate logs 明确报告 Jittor `1.3.11.0`，与当前目标 Jittor 2.0-refactor 不同；故该运行在目标矩阵所有等级都记为 `not-run`。即便只看它的历史数值运行，它也不满足全部 L1–L4 设备门禁：`trajectory.py`/`recovery_trajectory.py` 仅将 batch、冻结参数、buffer、梯度及参数快照搬到 CPU 保存，没有记录原始 `.device`；也没有保存 model forward 输出值。日志中的 model `hf_device_map={'': npu:<rank>}` 与 `fallback=0` 不能代替对每 rank batch、forward、gradient、optimizer state 的物理驻留核验。故该行 L1 首先失败，L2–L4 保持未通过；需要新的带 device/forward observer 的双侧运行键才能关闭。数值比较与原失败历史仍按原样保留。详细原始结果和各作业日志位于 `$TASK_STATE/runs/orpo-world2-rngfix-20261007/`，CPU Generator 先行 oracle/回归及原修复说明见同目录 manifest 与 `cpu-generator-parity-20261007` manifest。


## 目标 2.0-refactor 的 DPO/ORPO 全审计调度与失败记录（2026-10-08）

运行时守卫原先依赖共享文件 `current-job-id`。DPO candidate job 1761 与 ORPO analysis job 1787 并发时，1787 覆盖了 1761 写入值；DPO 完成全部 289 个 `jittor_core` 编译单元和 ACL backend 编译后，shim 子进程按 fail-closed 规则拒绝 job identity。作业在训练前退出，没有候选 batch、forward、gradient、参数更新或 optimizer 状态证据，因此该键不授予目标 L0–L4。失败日志与 manifest 保存在 `$TASK_STATE/runs/dpo-world2-2x-full-audit-20261008/`。

已将外部运行时守卫改为读取 `current-job-id-$SLURM_JOB_ID`，并要求 Slurm job ID 合法、存在该 job 的独立 marker，且 marker 与当前 job 一致；所有子进程仍须处于 Slurm overlap step。DPO 新键复用已成功且先运行的原生 torch_npu oracle job 1783（连续训练与 fresh-process resume），候选使用独立 JITTOR_HOME、运行缓存、launcher 状态和运行 ID，并串行复用已构建的 ccache 对象；Slurm 1797 正运行，1798 依赖候选、1799 依赖恢复完成。原始失败运行键与日志保留。

ORPO identityfix01 job 1790 在启动 `launch.sh` 时因文件缺少可执行权限退出（exit 13），模型没有启动；该键保留为失败。全新 jobguardfix01 键修正权限和 per-job marker：原生 job 1800 在 2026-10-08 20:45:55 至 20:48:18 于 worker 成功完成，候选 job 1801 于 20:48:18 在 `cscg-hw01` 启动。2026-10-08 21:00 CST 复核时，候选仍处于 Jittor 首次编译阶段，rank 日志推进到 87/289；尚未产生候选模型训练、设备审计或回退计数证据。恢复 job 1802 依赖候选成功，比较 job 1803 依赖恢复完成。不能把原生 oracle 成功提升为候选通过。相关包、checksum、Slurm 日志和 manifest 均未版本化，位于 `$TASK_STATE/runs/`。

2026-10-08 21:22 CST 再次检查同一运行键时，job 1801 仍在 `cscg-hw01` 运行。共享 `jittor_core` 已完成 289/289；候选随后进入 Jittor Python 绑定扩展编译，worker 上可见活跃 `cc1plus`，rank 0 的独立编译缓存仍有新对象写入。rank 0/1 launcher 日志仍只有 `ORPO_DEVICE_AUDIT_START`，没有候选模型构造、batch、forward、gradient、optimizer 或 fallback 结果；因此候选验收仍未通过任何目标级别。job 1802/1803、DPO 定向复现 1821 和结构门禁 1823 仍依赖前序作业；不重跑或替换这些运行键。

2026-10-08 21:42 CST 对同一 job/运行键复核：job 1801 仍为 `RUNNING`（已运行 53 分钟，4 小时上限），worker 上 torchrun 父进程和两个 rank 子进程仍存活；rank 0 的隔离 ccache 在 21:41:49 继续生成新 manifest/cache 统计，说明首次扩展编译仍有进展。两 rank 观测日志和 shim 汇总日志仍停留在 launcher 开始标记，没有新增候选模型、batch、前向、梯度、优化器或 fallback 结果。Slurm accounting 当前不可用（`sacct` 报 storage disabled）；队列中 1802、1803、1821、1823 仍按既有依赖等待。故保持原运行继续，不因编译耗时重新提交；本次状态检查不构成任何候选 L0–L4 通过证据。

2026-10-08 21:54 CST，job 1801 已越过首次 JIT/ACL 扩展编译，进入真实公开 `swift.cli.rlhf --rlhf_type orpo` 两 rank 训练初始化：rank 0 配置显示 `Train: 0/4`，日志出现 HCCL all-gather；trace 目录已有 rank0/1 的 trainable initial、frozen initial、buffer initial 和 `batch-step0.npz`。这只证明入口已开始执行且 step-0 batch 已被采集；尚无前向数值、梯度、优化器更新/状态和 fallback 完整窗口。设备事件与最终回退计数由回调在 train-end 才落盘，因此目前仍不能通过 L0–L4。worker 同时输出 `torch.utils.checkpoint` 在 shim 下仅直接执行函数、不会重算激活的警告；该警告已保留，待结果中评估其对该工作负载内存/性能解释的影响。

2026-10-08 21:56 CST，同一运行键的 rank0/1 均新增 `forward-values-step0-call0.npz`（各 4388 bytes），候选的 step-0 模型前向已实际返回并保存数值。两 rank 日志继续有 HCCL all-gather；尚无 step-0 梯度、optimizer state、更新后参数或完整 fallback/device 清单文件。故暂仅记录候选前向已执行，尚不能判定其设备驻留、数值对拍或训练层通过；比较仍依赖后续原生 oracle/候选结果与 job 1803。

2026-10-08 21:59 CST，候选两 rank 均已产生 step-0 gradient、pre/post 参数和 optimizer-values 快照，且 rank0/1 均已记录 step-1 batch；公开训练日志显示保存 `checkpoint-1`。job 1801 仍在运行并继续编译/执行 ACL kernels，未完成四步轨迹。由于 per-rank 原始 device 事件和最终 fallback 汇总要等 callback train-end 写出、原生对照及 job 1803 比较尚未发生，这只是候选单步执行证据，不授予 L2/L3/L4 通过。

2026-10-08 22:00 CST，ORPO candidate job 1801 以 Slurm `COMPLETED 0:0` 结束。两 rank 的 step0–3 batch、forward-values、gradients、pre/post parameters 和 optimizer-values snapshots 均存在，且输出目录有 checkpoint-1..4。rank0/1 的 `events.json` 均记录 Jittor shim/ACL、HCCL world-size=2、strict `fallback_policy=error`、起始及最终 fallback count=0、4 步日志及 final state；`devices.json` 的参数、compute input/output、forward output、gradient、optimizer-state 等事件在 rank0 仅出现 `npu:0`、rank1 仅出现 `npu:1`。这是 candidate 单侧的四步公开 ORPO 与设备驻留证据；native torch_npu oracle job 1800 已先运行并有对应 rank trace，但数值跨实现比较尚待 job 1803，故不据此判 L2/L4 对拍通过。全新进程恢复 job 1802 已按 `afterok:1801` 启动，下一步从 checkpoint-3 继续；L3 仍待恢复轨迹和比较结果。

## DPO Bool-mask 探针的过期基线修正（2026-10-08）

只读检查发现待运行 job 1821 的脚本在 oracle 前要求工作树 HEAD 精确为 `4d1c8c617d1c3f5b44bebb38f16065bc59c61355`，而当前基线为 `79eadd9c5ab225474c3253bb2c306b755646212f`。两 SHA 间 `src/`、`python/`、`backends/` 无差异，只有结果报告有变更，但原 SHA 守卫仍会阻止该脚本启动 oracle。保留 1821 原提交和 r3 运行目录，不覆盖其记录。

最初新增 r4/job 1825 依赖 `afterany:1823`，并将精确 SHA 守卫更新到 `79eadd9`；提交本报告后，守卫又会因新的文档提交而失效。故在其启动前取消 1825，保留其作业历史和 r4 运行目录，再新增唯一运行键 `getitem-boolmask-fused-acl-repro-20261008-r5`，job 1826 同样依赖 `afterany:1823`。r5 固定代码基线 `79eadd9`，在 worker 上要求该基线为当前 checkout 的祖先且 `src/`、`python/`、`backends/` 无差异，同时记录 worker 实际 HEAD；这样允许后续仅文档的提交，不会掩盖源代码变化。r5 复用 r3 probe 与已完成 r2 编译缓存，先执行 torch_npu oracle，再顺序执行 `backend_fallback=error` 严格 ACL candidate，并断言设备驻留与零 fallback。job 1823 完成后才会启动 r5，不与现有 ORPO/恢复/比较 JIT 链并发。探针仅用于定位 `roll → bool mask → leading-row slice → masked select` 的 ACL 算子断点，不代表 DPO 公开训练通过。r3/r4/r5 脚本、manifest 和 SHA256SUMS 未版本化，分别位于 `$TASK_STATE/runs/getitem-boolmask-fused-acl-repro-20261008-r3/`、`r4/`、`r5/`。

## 本轮终态复核（2026-10-08）

ORPO `orpo-world2-2x-full-audit-20261008-jobguardfix01` 已由 native job 1800、candidate job 1801、恢复 job 1802 和 comparator job 1803 完整闭环。公开入口为 `swift.cli.rlhf --rlhf_type orpo`，tiny LLaMA、Swift LoRA、FP32 fused AdamW，world-size 2；连续四步并从 checkpoint-3 在新进程续到 step 4。比较器共 146 项、0 失败。native torch_npu 和 candidate ACL 两边 rank 0/1 的初始/最终参数、batch 对应计算输入、前向输出、梯度和 84 个 optimizer state 张量均分别在 `npu:0`/`npu:1`；candidate HCCL、`backend_fallback=error`，连续训练和恢复阶段 fallback 均为 0。native 通用 fallback counter 不可用，记录 unknown。此锁定配置的 L0–L4 通过，L5 未运行。

历史 DPO 公开候选 job 1797 曾在首步 ACL fused graph 的 `reindex/void` 处失败，不能用窄布尔选择探针替代该图；该根因由后续 int64 Roll 修复处理，旧失败不改判。r5/job 1826 的 native oracle 已运行，但 candidate 与结构门禁 job 1823 都在 Jittor 导入前因未设置 `CCACHE_DIR` 而退出；这是运行脚本环境错误，不是算子验收结果。后续全新 DPO 公开训练与恢复由本报告下方 2026-10-09 专节记录，仅对该锁定配置授予 L0–L4。Slurm 当前仍保留 1798 (`DependencyNeverSatisfied`) 与 1799（等待 1798）两项终态不可达的依赖作业；不把它们描述成仍会执行的验证。

结构门禁 job 1832 在 NPU worker 上运行至结束：1385 passed、8 skipped，另有 `test_process_mode_contract.py` 两项因子进程冷启动超过默认 600 秒而失败，child collector 明确提示提高 `JITTOR_TEST_CHILD_TIMEOUT`。仅复制编译产物到另一 `JITTOR_HOME` 的 job 1858 在测试前因路径相关缓存键变化而重建 `jit_utils` 并以退出码 3 结束，不作为测试结果。随后 job 1860 在原 Jittor/ccache 路径上把 child timeout 设为 1200 秒，仅重跑该失败文件，4 项全部通过（20.90 秒）；因此定向覆盖已通过，但 job 1832 的全量结构门禁仍记为未全绿，不宣称全量门禁通过。原始日志在 `$TASK_STATE/runs/ms-swift-report-structure-timeout-retry-20261008/pytest.log` 和 `$TASK_STATE/runs/ms-swift-process-mode-contract-retry-20261008/pytest.log`。

2026-10-09 的提交前全量结构门禁首次作业 1907 在 NPU worker 上完成 1387 passed、8 skipped，但严格会话审计将缺失的 `pytest-xdist` 依赖计作 2 项 `other` skip 并以退出码 1 结束。作业 1910 的补依赖脚本因在 batch shell 而非 Slurm overlap step 执行 pip，在测试前退出；两次结果均保留且不算门禁通过。新运行键 `ms-swift-dpo-report-structure-xdist-step-20261009` 的 job 1911 在 overlap step 将 `pytest-xdist==3.6.1` 及其依赖安装到独立目录，并在真实 NPU worker 重跑全量 `tests/structure`：1389 passed、6 skipped、0 other skips，24 warnings，用时 569.24 秒；`tools/check_repo_layout.sh` 通过。该全量结构门禁通过，原始作业日志、安装记录和隔离缓存均未版本化，位于 `$TASK_STATE/runs/ms-swift-dpo-report-structure-20261009/`、`ms-swift-dpo-report-structure-xdist-20261009/`、`ms-swift-dpo-report-structure-xdist-step-20261009/`。

本次仅更新结果文档后的门禁也保留：job 2142 在 Jittor 首次重建 `jit_utils` 后按设计退出 3，未进入测试；新运行键 job 2143 在真实 NPU worker 完成 `tests/structure`，1388 passed、6 skipped、1 failed，布局检查通过。唯一失败是 `test_process_mode_contract.py` 子进程收集 `_NATIVE_TARGET` 时，内部连续 3 次遇到 `jit_utils was rebuilt and cannot be reloaded`（exit 3），未触发其进程模式断言；worker log 位于 `$TASK_STATE/runs/report-l3-update-structure-r2-20261009/`。这不改变已通过的 1911 基线结果，也不把当前提交门禁说成全绿。

## DPO 掩码融合图的精确重现（2026-10-09）

运行键 `dpo-mask-select-exact-repro-20261008-r1`（worker job 1862）以两个 chosen/rejected label batch 重现公开路径 `cat → roll → labels != -100 → origin[:num_examples][loss_mask[:num_examples]].mean()`。原生 torch_npu oracle 先在 NPU 完成，labels、mask、origin、selected、loss 在拷贝回主机前均为 `npu:0`；选中值为 `[0,1,2,4,5,6]`，均值为 `3`。随后 candidate 使用 ACL 和 `backend_fallback=error`，在 `getitem_op.py:304` 计算 `slices.sum().item()` 时失败。严格门禁报告融合序列 `array → reindex → broadcast_to → unary.cast → binary.not_equal` 中 `reindex/void` 未注册；fallback 被拒绝，没有把该失败转为 CPU 执行或通过。原始 oracle、候选栈和脚本位于 `$TASK_STATE/runs/dpo-mask-select-exact-repro-20261008-r1/`，候选日志为 `candidate-1862.jsonl`。

该精确路径确认公开失败至少发生在生成 `labels != -100` 的掩码图，而不仅是后续 bool masked-select/getitem。r1 日志没有记录 ReindexOp 的索引表达式，但源码表明本 case 的 `roll` 对每个维度构造带取模规范化和边界条件的索引表达式；这不是 Expand/identity。通用 `ReindexOp` 仍支持任意索引表达式、overflow 条件和额外索引张量，不能通过无条件名称映射到 `Expand` 处理。随后确认 ACL 专用 `Roll` 仅因 int64 dtype 白名单漏项而未被选中，修复与验证见下节。

## int64 Roll 根因修复与精确链路回归（2026-10-09）

代码检查发现 `python/jittor/ops/shape_ops.py::roll` 先通过 `tensor.roll` dispatch 选择后端 kernel；ACL 已有 `RollACL`，但 `backends/acl/kernels/tensor.py::_roll_acl` 的 dtype 集合漏掉 int64。此前 DPO 的 int64 labels 因此走通用 `ReindexOp`，再与 `labels != -100` 融合，触发未注册节点。CANN 9.0 所用 `aclnnRoll` 在当前设备上可执行 int64：公开 CANN API 对应设备和支持类型见 [aclnnRoll 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-math/aclnnRoll.md)，本轮 NPU 实测也已通过。

提交 `ac515b406c80e33528c6cc8f953ff0497f466474` 将 int64 加入 `_roll_acl` 支持集合，并增加 CPU/加速器值回归与严格 ACL NPU 设备/fallback 回归。运行键 `dpo-mask-select-exact-repro-20261009-r2`、job 1865 在源码基线 `ac515b406` 上先运行原生 torch_npu，再运行 `backend_fallback=error` candidate；完整精确链路成功，选中值 `[0,1,2,4,5,6]`、均值 `3` 一致。Candidate 的 labels、mask、origin、selected、loss 均在 NPU，`fallback_count=0`。原始证据目录 `$TASK_STATE/runs/dpo-mask-select-exact-repro-20261009-r2/`。这只验证具体 Roll 与掩码操作；当时 DPO 公开 L4 训练/完整恢复仍待新公开运行键验证。后续完整结果见下节，不追溯改判任何旧运行。

## DPO 公开入口四步训练与完整恢复（2026-10-09）

目标工作负载为 tiny LLaMA、Swift LoRA、FP32、`adamw_torch_fused`、固定 DPO 数据/seed、真实单机双 NPU HCCL；公开入口由 `torch.distributed.run` 启动真实 `swift.cli.rlhf --rlhf_type dpo`。运行键 `dpo-world2-2x-full-audit-20261009-int64roll-fix01` 使用先行 native torch_npu oracle job 1783（复用 `dpo-world2-2x-full-audit-20261008` 的连续四步与 checkpoint-3 新进程恢复证据；数据及 adapter 输入哈希一致），随后 candidate job 1887 在 ACL strict mode 连续训练至 checkpoint-4，job 1888 在新进程从 checkpoint-3 恢复并续至 step 4。worker 记录源代码基线为 `ac515b406c80e33528c6cc8f953ff0497f466474` 的后代 `bd3ba23ece5bbcedf581600a081f044c5fb7bc19`，且 `src/`、`python/`、`backends/` 与修复提交一致。

Comparator-only job 1905 使用新键 `dpo-world2-2x-full-compare-20261009-fix03`，没有重新运行 native/candidate 模型。比较协议累计 170 项检查、0 失败：连续跨运行时 78 项、恢复跨运行时 22 项、两侧各自恢复轨迹 22+22 项、两侧 step-3 恢复初值同运行时精确检查各 6 项、每侧/阶段/rank 设备审计 8 项、跨运行时 checkpoint 4 项及同侧完整恢复 checkpoint 2 项。最大绝对差 `1.1920928955078125e-07`，在原预锁容差内；没有在观察结果后放宽阈值。两 rank 的 runtime context 均为 HCCL/world-size=2，rank 与 NPU 绑定为 0→`npu:0`、1→`npu:1`。候选 policy 为 `backend_fallback=error`，每 rank 连续及恢复阶段计数均为 0；参数、batch/计算输入、forward output、gradient、optimizer-state 与 buffer 物理驻留均通过。native 为原生 torch_npu，通用回退计数器不可用，记录 `unavailable/unknown`。该锁定配置 L0–L4 通过；L5 真实尺寸稳态性能未运行。

旧失败保持原样：job 1797 的首步 `reindex/void` 失败促成 int64 Roll 根因修复；比较 job 1889 在数值比较前因 comparator 对 fallback 整数调用 `.get()` 而退出；job 1902 暴露 observer frozen/buffer 文件名映射错误及将跨 runtime 恢复初值误设为逐位比较；job 1904 的 16 项失败均为恢复阶段 frozen/buffer 路径漏用映射函数。修正比较器后 job 1905 全部 170 项通过。对应原始日志、trace、checkpoint、manifest 和比较报告均在 `$TASK_STATE/runs/dpo-world2-2x-full-audit-20261009-int64roll-fix01/`、`dpo-world2-2x-full-compare-20261009-fix01/`、`...-fix02/`、`...-fix03/`；不把比较器失败误记为模型数值失败，也不删除失败原件。


## LLaMAPro 单 NPU 训练与 checkpoint 恢复断点（2026-10-09）

运行键 `llamapro-world1-full-audit-20261009-jittor2-fused-r1` 绑定基线 `82263a2abb537626c8e2c3289e950942ea132889`、tiny LLaMA、Swift LLaMAPro、FP32、fused AdamW、固定数据与 seed、HCCL world-size 1（单进程单 NPU）。原生 torch_npu job 1917 先完成，随后严格 ACL candidate job 1922 完成公开 `swift.cli.sft` 三步；worker comparator job 1926 对 30 项进行比较，0 失败。两端 loss 轨迹一致；candidate 以 `backend_fallback=error` 运行且 fallback=0，记录中的参数、输入 batch、forward 输出、gradient、optimizer state 均物理驻留 `npu:0`。跨实现最大绝对差为参数 `5.97909e-7`、optimizer state `2.7940e-9`、logits `8.9407e-8`、gradient `1.4901e-8`、optimizer update `5.9605e-7`，均使用预先锁定容差。故该特定公开训练配置的 L0–L2、L4 通过。

L3 独立遵循 oracle-first。native 新进程恢复 job 1928 失败，尚未启动候选恢复。checkpoint-3 将 adapter 存于 `default/adapter_model.safetensors`，但没有根目录模型 shard/index；Transformers 4.57.6 将 SwiftModel wrapper 当成普通模型，并在 `_load_from_checkpoint` 中调用 `load_sharded_checkpoint` 搜索根目录 index 后退出。它是该锁定恢复路径的 native oracle 失败，不能用 adapter 权重加载代替 optimizer/scheduler/trainer/RNG 的完整恢复。原始证据在 `$TASK_STATE/runs/llamapro-world1-full-audit-20261009-jittor2-fused-r1/` 与 `...-resume-r2/`；resume-r1 的无效 checkpoint 路径及 1918/1919、1924/1925 的启动/比较器错误均保留在其独立目录。L5 未运行。

后续新键 `llamapro-world1-full-audit-20261009-jittor2-fused-resume-index-r1`（native job 2141）仅在原 checkpoint 的副本上添加根目录 shard index，保持 adapter 权重 SHA 不变；这修正了模型 shard 定位，但 native restore 随后仍在 optimizer load 失败。日志显示恢复后的 `SwiftModel` 有 `0.1397M` 参数而可训练参数为 `0.0000M`，接着报 optimizer parameter-group size mismatch，因此未提交候选恢复。对安装版 ms-swift 4.5.2 的只读源码追踪发现 `TunerMixin.prepare_model` 以 `is_trainable=True` 调用 `tuner.from_pretrained`；Swift tuner 路径将该 kwargs 传至 `SwiftModel.from_pretrained`，其签名默认 `inference_mode=True`，未消费 `is_trainable`，构造函数也因此没有启用 adapter 的可训练标记。此证据指向上游训练恢复语义缺陷；尚无当前公开 CLI 可用的上游修复证据，故 L3 保持 failed，不能用 adapter-only 加载绕过 optimizer/scheduler/trainer/RNG 恢复。原生失败和未启动候选记录位于 `$TASK_STATE/runs/llamapro-world1-full-audit-20261009-jittor2-fused-resume-index-r1/`。


## Swift Adapter 双 NPU 训练与 checkpoint 恢复断点（2026-10-09）

运行键 `adapter-world2-full-audit-20261009-jittor2` 基于 Jittor 2.0-refactor，固定 tiny LLaMA、Swift Adapter `adapter_length=16`/GELU、FP32、fused AdamW、公开 `swift.cli.sft` 三步、单机双 NPU/HCCL。native torch_npu job 1930 先完成；严格 ACL candidate job 1931 随后完成，job 1935 在真实 NPU worker 上完成 2092 项比较、0 失败。预锁容差下最大绝对差为 logits `4.76837158203125e-7`、gradient `5.960464477539063e-8`、parameters `4.470348358154297e-8`、updates `4.470348358154297e-8`、optimizer state `7.450580596923828e-9`。每 rank 参数、真实 batch、前向值、梯度、优化器状态、frozen tensors 均记录在 rank 对应 NPU；HCCL ranks 0/1 分别绑定 `npu:0`/`npu:1`；候选 `backend_fallback=error` 且 fallback=0。此锁定配置 L0–L2/L4 通过，L5 未运行。

完整恢复仍遵循原生先行。job 1936 的 native fresh-process oracle 在两 rank 失败：checkpoint 把 Swift Adapter 权重保存在 `default/adapter_model.safetensors`；运行时对象为 `swift.tuners.base.SwiftModel`，而 Transformers 4.57.6 的 `_is_peft_model` 只识别 `PeftModel`/`PeftMixedModel`，所以 Trainer 将其送入全模型加载分支并在 checkpoint 根目录查找 `pytorch_model.bin.index.json` 或 `model.safetensors.index.json` 后退出。候选恢复未启动；不能以 adapter-only reload 替代完整 optimizer/scheduler/trainer/RNG 恢复。比较器 job 1932（bool 减法）、1933（脚本语法）、1934（把汇总日志计成 step）失败均保留；修正后的 worker job 1935 比较 2092 项、0 失败，执行 stdout 与归档 JSON 哈希已逐项核对，artifact path 偏差在 run manifest 中注明。原始运行与恢复断点位于 `$TASK_STATE/runs/adapter-world2-full-audit-20261009-jittor2/`、`adapter-world2-full-compare-20261009-jittor2-fix02/` 至 `fix04/`、`adapter-world2-resume-audit-20261009-jittor2-r1/`。

## Swift Adapter 双 NPU L3 完整恢复复核（2026-10-09）

针对旧 Adapter L3 native checkpoint 分支错误，另用公开 `swift.cli.sft` 进行 native-first 完整恢复复核。保留的原始三步 checkpoint-3 在新进程恢复后续训至 step 4：native torch_npu job 2082，strict Jittor ACL job 2091；两侧各自另有公开 CLI 四步不中断参考，native job 2135、candidate job 2136。恢复和不中断运行均为 world-size 2、真实 HCCL，每 rank 使用固定分区和同一锁定输入。candidate 全阶段启用 `backend_fallback=error` 并记录起止计数 0。

worker comparator job 2139 使用 compare-only key `adapter-world2-resume-audit-20261009-jittor2-compare-r6`，检查 11,637 项、0 失败。每 rank 的 runtime/provider、rank/NPU/HCCL、checkpoint 完整文件、参数与真实 batch、forward、梯度、更新、optimizer 张量设备及数值均通过；恢复首步与连续第 4 步的 batch 精确相等。adapter、optimizer、scheduler、trainer/global step 和 rank RNG 的恢复状态与对应 runtime 连续运行一致；训练浮点轨迹采用预先锁定的类别容差，未调整阈值。最大绝对差：logits `1.1920928955078125e-7`、梯度 `5.960464477539063e-8`、参数/更新 `4.470348358154297e-8`、optimizer `1.1175870895385742e-8`。

协议失败保留且不重判：compare-r4 的 92 项失败来自它要求每一步重复记录只在 step 0 产生的 initial-parameter 事件，以及把同 runtime 恢复/连续的浮点快照错误按 bitwise exact 比较；compare-r5 的数值检查 11,637 项通过，但结果元数据仍写入已失败的旧连续任务号 2132/2133，因此不作为交付报告。compare-r6 独立运行修正 provenance，引用 2135/2136 并复验通过。原始日志、checkpoint、逐 rank 快照和比较产物均未版本化，位于 `$TASK_STATE/runs/adapter-world2-resume-audit-20261009-jittor2-r2/`、`...-r4/`、`adapter-world2-uninterrupted-audit-20261009-jittor2-r2/` 以及 `adapter-world2-resume-audit-20261009-jittor2-compare-r4/` 至 `...-r6/`。此结论只适用于锁定 tiny LLaMA/Swift Adapter 配置；L5 真实尺寸性能仍未运行。

## Swift Adapter 单 NPU 公开推理（2026-10-09）

锁定 tiny LLaMA、已验证的原生/候选 Adapter checkpoint-3、Swift Adapter length=16/GELU、FP32/eager、4 条固定 prompt、greedy 4 token、公开 `swift.cli.main infer`。原生 oracle job 1940 运行键 `adapter-world1-infer-audit-20261009-jittor2-r4`；strict ACL candidate job 1975 使用 candidate-only 运行键 `...-r6` 并复用同配置原生结果；worker comparator job 1976 检查 2461 项、0 失败。比较精确检查输入和 token ID、响应结构、键/形状/dtype/有限值、所有前向输出和每次前向后的完整参数值与设备；参数 atol=`1e-6`，logits atol=`1e-5`/rtol=`1e-4`，均为预先锁定阈值。最大绝对差分别为参数 `4.470348358154297e-08`、logits `1.1920928955078125e-07`。两侧 16 次前向的输入、logits、tokens 均记录在 `npu:0`；全部 29 个参数在每次前向后都在 `npu:0`。Swift 4.5.2 `AdapterModule.forward` 会在第一次收到激活时将 8 个 Adapter 线性参数从 CPU 延迟迁移到激活设备，因此记录了前向前 CPU 状态和迁移后每次前向设备，不把构造前快照混作计算阶段驻留。候选 runtime identity 为 Jittor shim/ACL，`backend_fallback=error`，完整推理过程起止 counter 均为 0。故此锁定公开推理配置 L0/L1/L4 通过；训练 L2/L3 不适用，tiny case 的 L5 未运行。

## ms-swift 模型 registry 盘点（2026-10-09）

只读盘点运行键 `ms-swift-registry-inventory-20261009-r2` 在 Slurm job 2042 的 `cscg-hw01`、Ascend 910B3 worker 上运行原生环境导入与设备探测；读取的是安装版 ms-swift 4.5.2 的 `swift.model.MODEL_MAPPING`（registry 源文件 SHA-256 `78e73070ab92a352ad1ebf34580d2e090a460c95e4661c8722203e50310bf9a9`），共 222 个 model type，其中 118 个 `is_multimodal=true`，无 T5 model type。注册表含 BERT/ModernBERT 编码器及 reranker，也有 118 个多模态 model type。job 2041 的首版盘点在 JSON 序列化 `ModelKeys` 对象时失败并保留；r2 已完整写出 222 条 registry JSON，worker 脚本最后的可选格式化命令因该环境没有 `python` 可执行名而返回非零。原始 JSON 由控制端 JSON::PP 成功解析并校验条目数。此项仅证明公开安装包暴露哪些 registry 类型及无 T5 注册，不执行模型构造、前向或 Swift CLI，也不为任一 L0–L5 授予通过。原始结果、模块路径、版本、设备与失败栈位于 `$TASK_STATE/runs/ms-swift-registry-inventory-20261009-r1/` 和 `...-r2/`。

原始证据位于 `$TASK_STATE/runs/adapter-world1-infer-audit-20261009-jittor2-r4/` 和 `...-r6/`。run r1 的探针钩错 SwiftModel.forward，r2/r3 的路径与协议缺陷、r5 的 fallback 计数缺口和 comparator 部分结果均保留在各自运行目录，不用于本结论；没有重新运行已完成的 native r4 oracle。

## BERT encoder / Swift Adapter 严格恢复对拍（2026-10-09）

在真实 Ascend 910B3 worker 上复核 Swift Adapter direct API 的 BERT sequence-classification 代表面。配置为 hidden=64、2 层、FP32、固定 batch、AdamW 三步，oracle 与候选共享完整初始参数、输入及 adapter。原生 torch_npu jobs 2144/2145 先完成连续三步和独立进程恢复，job 2147 对 95 个状态键、105,206 个值做 exact internal resume 检查。候选使用真实 ACL 和 `backend_fallback=error`。

最初严格比较 job 2157 首先发现 BERT 非持久整数 buffer `position_ids` dtype 不符：native 为 int64，shim 为 int32。根因是 compat 工厂未覆盖 Jittor `arange` 的 int32 默认值，而 Torch 整数边界 `arange` 默认 int64。修复位于 `compat/torch/installers/factories.py`：仅纯 Python 整数边界且调用方未指定 dtype 时显式传 int64；浮点边界和显式 dtype 保持原逻辑。对应 worker 回归 job 2162 在 Jittor 2.0.0 / ACL worker 上 1 passed、0 skipped；测试在 CPU 与可用加速器执行，冷编译串行完成。

修复后严格 ACL candidate job 2165 的连续训练与新进程恢复均成功，损失与 oracle 相同，候选 fallback 起止均为 0；每步参数/buffer、输入、输出、梯度、optimizer tensor 均记录为 NPU 驻留。job 2166 comparator-only 诊断确认初始参数与 buffer（含 int64 `position_ids`）逐项 exact，输入、logits、loss、梯度和 optimizer 状态符合原协议，且两侧各自连续 step 2 与 fresh resume 逐位 exact。跨运行时却在 AdamW 更新上失败：配置仍为 `eps=1e-8`，step 0 的两个 `linear2` 更新最大差分别为 `6.243179e-6` 和 `4.089903e-6`，超过预锁 `atol=1e-6, rtol=1e-5`；梯度最大差 `1.66e-9`、optimizer moments 最大差 `1.66e-10`，loss 精确相同。差异符合近零梯度在过小 epsilon 下被归一化放大的现象，不以放宽容差处理。故该 direct API 配置可记 L0/L1 通过、L2/L3 失败；公开 CLI L4 与稳态 L5 未运行。

后续运行键 `encoder-bert-full-audit-20261009-jittor2-r7-eps1e-6-native` 的 native-first job 2167 已通过：原生 torch_npu 连续三步与 fresh resume 均在 Ascend910B3 执行；内部 exact 检查 95 键通过，loss `0.7017301917076111`，每步模型、输入、前向、梯度、buffer 和 optimizer tensor 均实测 NPU 驻留。

严格 ACL candidate 与 full comparator job 2168 随后通过，使用同一 `AdamW eps=1e-6`。oracle 与 candidate 各自 continuous step2 / fresh resume 均为 95 键、105,206 个值 exact 一致；跨运行时三步轨迹、初始完整参数与 buffers、以及恢复 step 共有 95 键并全部通过既定 dtype/shape 与数值门禁。各步骤最大绝对差：更新 `1.313e-7`、参数 `2.006e-7`、梯度 `4.427e-10`、optimizer tensors `5.554e-11`、logits `1.886e-8`、loss `5.961e-8`；初始参数和 buffers exact。候选运行全阶段 `backend_fallback=error` 且 fallback start/end 均为 0，所有模型、batch、forward、梯度与 optimizer tensor 均为 NPU 驻留。故锁定的 BERT encoder / Swift Adapter direct API case 的 L0–L3 通过。该运行不是公开 CLI，L4/L5 仍未运行，也不推广至其他模型、tuner 或真实尺寸工作负载。

### BERT Adapter 公开 CLI native 边界预检（2026-10-12）

新键 `bert-base-seqcls-adapter-cli-preflight-20261012-r1/r2`：job 2577 是 harness 失败，worker 检查器因脚本漏写 `current-job-id-2577` 标记而退出，未启动 Swift CLI；原始结果保留，不作为适配结论。修正标记后的 job 2578 在 `srun` NPU worker 上确认 torch_npu 设备 `npu:0`、Swift 4.5.2 / Transformers 4.57.6，读取六行固定数据和已校验 BERT-base 权重（SHA256 `2d3ba1ac79e07d1d23ba846ed3d34f3aa49595751a7e0d55aa715b9de135af76`），并调用公开 `swift.cli.main sft` 的 `tuner_type=adapter, adapter_length=16, task_type=seq_cls` 配置。CLI 成功解析数据并进入 `SwiftSft.prepare_model()`，随后在安装版 `swift/pipelines/train/tuner.py::prepare_adapter` 访问 `model_arch.mlp` 时因其为 `None` 抛出 `AttributeError`；训练、checkpoint 写出均未开始。未启动 strict ACL candidate。故原 direct API 的 L0–L3 通过结论不变；本公开 CLI 格子的 L4 仍是 `not-run`，当前 native oracle 前置被 Swift 上游架构路径阻塞，而不是 Jittor/ACL 失败。是否扩展 BERT encoder 的 Swift Adapter 注入语义需先定支持范围；本轮未改源码。

所有运行脚本、快照和日志均未版本化，位于 `$TASK_STATE/runs/encoder-bert-full-audit-20261009-jittor2-r1/` 至 `...-r8-eps1e-6-candidate/`，比较诊断在 `...-r6-numeric-diagnostic/`。

2026-10-10 开始探测该 BERT 配置的公开训练 CLI。预检键 `bert-seqcls-cli-preflight-20261010-r1`（job 2196）在 Python 启动前因脚本变量未传入子 shell 失败；r2（job 2197）成功记录原生 torch_npu / Ascend910B3 身份，但误调用内部 `swift.cli.sft` 模块帮助，该模块仅显示 `--tuner_backend`，随后选项 grep 以 1 退出。r3（job 2198）确认安装的 `oracle/bin/swift` shebang 指向容器绝对路径 `/usr/local/python3.11.15`，主机命名空间无法解析，退出 126。r4（job 2199）通过隔离 runtime 调用已安装 `swift.cli.main.cli_main`，其确实分发到 `swift/cli/sft.py`；但该脚本中的 `try_init_unsloth()` 先以仅声明 `--tuner_backend` 的 argparse 解析器处理 `--help`，截断了完整训练参数帮助，grep 再次返回 1。四项失败均为 CLI 探测/harness 证据，不含模型构造或训练，本身不授予 L0–L4 通过；随后完成的公开训练验收见下段。r1–r4 的原始 manifest、输出和 worker runtime 记录未版本化，分别位于 `$TASK_STATE/runs/bert-seqcls-cli-preflight-20261010-r1/` 至 `...-r4/`。结果报告更新后，Slurm job 2203 在真实 NPU worker 完成 `JITTOR_TORCH_SHIM=1` 的结构门禁：1387 passed、8 skipped、0 failed；两个 skip 来自环境未安装 pytest-xdist。job 2200、2201、2202 的 ccache 初始化缺失、冷编译超时和子进程 600 秒超时均保留原始证据，最终 r4 通过。

## BERT sequence-classification 公开 SFT CLI 对拍（2026-10-10）

使用公开 `swift sft` / `swift.cli.main.cli_main`，tiny BERT sequence-classification、full finetuning、FP32、batch=1、固定 6 条本地 JSONL、3 步训练并从 checkpoint-2 在新进程恢复至 step 3。锁定 AdamW fused、lr=`1e-3`、weight decay=0、预置容差为 forward/gradient/updates/parameters `atol=1e-6, rtol=1e-5`、optimizer `atol=1e-7, rtol=1e-5`，初始权重/buffers 与 batch 精确比较。所有作业在 Ascend 910B3 Slurm worker 上，源基线为远端同步后的 `c9d8355b4a963067925eccb5c8ac7a34df5ee119`，工作树干净；oracle 先于 strict ACL candidate。

运行键 `bert-seqcls-cli-world1-20261010-r9/r10`（jobs 2215/2216）锁定 `eps=1e-6`。修正前的比较器 r2/r3 错误断言 continuous 初始参数等于 checkpoint-2 resume 初始参数，保留其失败记录；比较器 r3 修正来源关系后检查 5,917 项，剩余 8 项严格数值失败。forward/gradient/optimizer state 与初始状态通过；更新最大差 `1.5564728528261185e-6`，最终参数最大差 `2.2095628082752228e-6`，超过预锁容差，集中在 `bert.encoder.layer.0.intermediate.dense.weight`。checkpoint-2 恢复初始参数两侧均与各自 checkpoint 精确相等，同 runtime 连续/恢复轨迹精确。Numpy worker-only 诊断显示该张量某坐标的 gradient 差约 `2.62e-9`、梯度约 `3e-7`、二阶矩约 `9e-17`；这是 epsilon 敏感性定位证据，不改变失败判定，也不把不同 epsilon 的结果互相替代。candidate strict ACL fallback=0 且计算张量驻留 NPU；native 通用 fallback counter `unavailable/unknown`。该锁定配置的公开 CLI 已执行但 L2–L4 未通过，L5 未运行。

独立运行键 `bert-seqcls-cli-world1-20261010-eps1e5-native-r1` / `...-eps1e5-acl-r1`（native job 2221 → strict ACL candidate job 2222 → read-only comparator job 2223）使用新锁定配置 `eps=1e-5`，其它输入、模型权重、训练/恢复命令、seed 和容差不变。5,917 项检查、0 失败；forward 最大差 `5.960464477539063e-8`，gradient `1.7601996660232544e-7`，optimizer state `1.4566467143595219e-8`，更新 `4.6938657760620117e-7`，更新后参数 `8.55419784784317e-7`。两侧 continuous/resume 内部轨迹及恢复初始参数对 checkpoint-2 均 exact；初始参数/buffers、batch 和训练数据也 exact。candidate runtime/provider 为 Jittor shim/ACL，`backend_fallback=error` 且全阶段起止 fallback=0；参数、真实 batch、forward、gradient 与 optimizer state 的物理位置均为 `npu:0`。native runtime/provider 为 torch_npu，物理 NPU 驻留通过，通用 fallback counter 仍记 `unavailable/unknown`。因此仅此 tiny BERT full-finetuning 公开 CLI 锁定配置 L0–L4 通过；L5 未运行。epsilon 是显式锁定超参数，不能由此推广到其它值、encoder、tuner 或整体 ms-swift。

真实尺寸 L5 前置盘点（worker runs `bert-real-size-cache-inventory-20261010-r1` 至 `-r4`）没有找到可用的完整模型权重：常用 Hugging Face/ModelScope 缓存未见模型 snapshot，ms-swift runs 下没有大于 50 MB 的 `.safetensors`/`.bin`；另查的共享 `project3_310b` / `project3_e2e_20261009` 模型目录仅 12–38 MB 且未发现匹配模型权重。worker 上对公开 `google-bert/bert-base-uncased` 配置文件的无凭证 HTTPS 探测（`bert-real-size-network-probe-20261010-r1` / Slurm job 2229）返回 HTTP 000，curl exit 28，连接 Hugging Face 约 10.2 秒超时。因此没有下载权重，也没有声称该真实尺寸模型通过正确性或 L5；当前 L5 仍 `not-run`，解除条件是 worker 获得可用的真实尺寸权重和可达的受信模型源。盘点 job 2226 因 `find | sort | head` 的 SIGPIPE 退出 141，原样保留；随后不截断的 r3 和定向 r4 结果用于上述结论。

追加 dropout=0/eager 的单步公开入口激活诊断（运行键 `bert-base-eager-layertrace-public-cli-20261010-r4`，native 2247 → strict ACL 2248 → worker comparator 2249）：两侧 203 项初始参数精确，layer-0 attention 输出和 intermediate dense 输出通过；最早失败位于 dense 后的激活输出，logits 最大差 `1.0939e-3`。固定输入 GELU 探针 `bert-gelu-isolation-20261010-r2` 使用相同输入 SHA256 `30d7776e…a8ca474e`，native torch_npu job 2251 先于 strict ACL candidate job 2252；两侧输入和输出物理位置均为 `npu:0`，候选对拍 max abs `4.7350e-4`、RMSE `2.5823e-4`。该探针没有采集候选 fallback 计数，只是激活边界定位证据，不单独证明最终根因或兼容通过。其 r1 启动作业 2250 因缺少 Slurm 当前作业标记，在执行计算前退出，保留于原目录。

后续路由隔离运行键 `bert-gelu-route-isolation-20261010-r1`（native 2262 → strict ACL candidate 2263 → worker comparator 2264）复用相同输入 SHA。native torch_npu 的 `approximate='none'` 与 `'tanh'` 输出位级相同；candidate 默认快速路径与 ACL dispatch 位级相同，关闭快速路径并屏蔽 dispatch 后的 portable erf 输出与 ACL dispatch 最大差 `9.537e-7`。三条 candidate 路径相对 native 的最大差仍约 `4.735e-4`，所有输入/输出在 `npu:0`，严格策略 error，四次 fallback 计数均为 0。故差异不是 `_FAST_GELU` 绕过 ACL 注册：关闭它仍复现；也不是 portable 公式与 ACL 算子之间的可见精度差。

为排除前一输入分布饱和导致 native 两个模式相同，独立运行键 `bert-gelu-contract-probe-20261010-r1` 在 Ascend worker 上用 2001 个 `[-10,10]` 均匀 float32 值重新对拍，native job 2265 先于 strict ACL candidate job 2266，worker comparator job 2267。oracle 的两个模式仍位级相同，且 native 输出与 tanh 数学式最大差 `4.768e-7`、与 erf 数学式最大差 `4.733e-4`；输入中 `|x|>1` 占 89.96%，但结论由公式对拍而非单看饱和区域得出。candidate ACL dispatch 与 portable erf 最大差 `9.537e-7`，ACL 输出相对 native 超出锁定容差，最大差 `4.733e-4`；portable tanh 与 native tanh 在预锁容差内，最大差 `4.768e-7`。candidate 全部输出位于 `npu:0`，`backend_fallback=error` 且完整窗口计数保持 0。结合当前 ACL 注册源码明确以 `approximate=0` 调用 `aclnnGeluV2`，并由 Python 实现只在 `approximate='none'` 走该精确 ACL 路径，已把行为断点收敛为：此安装的 torch_npu 在 Ascend 上对 `F.gelu` 的 `none` 和 `tanh` 都呈现 tanh 结果，而 strict ACL candidate 的 `none` 呈现 erf 结果。此为真实 NPU 上观测到的运行时行为差异；目前没有改写 shim/API 或 ACL 的语义来迎合该差异，也不把 GELU 或 BERT 训练格记为通过。路由探针原始数组、runtime identity、fallback 计数与脚本未版本化，分别位于 `$TASK_STATE/runs/bert-gelu-route-isolation-20261010-r1/` 和 `$TASK_STATE/runs/bert-gelu-contract-probe-20261010-r1/`。

本次文档集成的仓库布局检查在真实 NPU worker 上通过。结构门禁首个 job 2255 因 45 分钟时限在 pytest 15% 处被 Slurm 终止；复用其 JIT 缓存的新运行键 job 2256 完成 `tests/structure`：1,395 项收集、1,393 项执行，1,383 passed、8 skipped、4 failed。四项失败来自 `test_pool_legacy.py` 的 average/max/unpool 执行和 `test_pool_structure.py` 的 adaptive execution，严格 ACL 报错均为 fused `reindex/void` 变体未注册并拒绝回退；pytest-xdist 未安装导致两个 skip 被测试策略计入 `other`。这组门禁结果记录为失败，不归咎于文档变更，也不提升任何 ms-swift 验收格。job 2254 仅完成布局检查，因 wrapper 未在 overlap step 中启动而未进入 pytest；上述各次原始日志均保存在 `$TASK_STATE/runs/bert-activation-report-gates-20261010-r1/` 至 `...-r3/`。

完整 BERT CLI 命令、脚本、日志、逐阶段快照、运行时身份、设备 metadata、fallback 计数和 comparison report 均未版本化，分别位于 `$TASK_STATE/runs/bert-seqcls-cli-world1-20261010-r9/`、`...-r10/`、`...-eps1e5-native-r1/`、`...-eps1e5-acl-r1/`、`$TASK_STATE/runs/bert-seqcls-cli-diagnose-20261010-r1/` 与 `...-r2/`、`$TASK_STATE/runs/bert-seqcls-cli-compare-20261010-r2/`、`...-r3/` 及 `...-eps1e5-r1/`。r1 JSON 序列化失败保留；r2 和 r3 诊断/比较结果为最终有效证据。资源盘点 raw output 位于 `$TASK_STATE/runs/bert-real-size-cache-inventory-20261010-r1/` 至 `...-r4/` 与 `$TASK_STATE/runs/bert-real-size-network-probe-20261010-r1/`。

## BERT-base 真实尺寸公开 SFT 新收敛协议验收（2026-10-11）

完整模型权重来自 `AI-ModelScope/bert-base-uncased`，结构为 12 层、hidden size 768；序列分类头构造后共有 109.4838M 参数并全部训练。公开入口为 `swift sft`，FP32 full finetuning、fused AdamW `eps=1e-6`、最大长度 128、固定六条本地样本、batch=2、12 个 optimizer steps、seed/data_seed=3407；checkpoint 每 4 步保存。两侧共用已锁定的初始权重、tokenizer 和输入 fixture，按原生 torch_npu 先跑、严格 ACL candidate 后跑。

L0–L4 证据如上。L5 使用独立运行键 `bert-base-real-size-l5-world1-20261011-r1`、相同公开入口/模型/精度/优化器/batch/max_length/data/seed 配置，训练到 step 13；任务基线为 `f582d8afc2163f42a77558c685c976428cff8918`、工作树干净。native oracle job 2364 先于 strict ACL candidate job 2365；每侧均完成 steps 1–3 warmup，并在真实 worker `cscg-hw01` 上同步设备后测 steps 4–13 共 10 个 optimizer step。计时边界覆盖 forward、backward、optimizer 更新及更新后 NPU 同步。计时器检查每个训练参数均在 `npu:0`；candidate 为 Jittor 2.0.0 shim/ACL，`backend_fallback=error`，起止计数均为 0；native 为 torch_npu 2.10.0，通用 fallback counter unavailable/unknown。两侧每步 batch 均为 2 个有效样本，固定六样本无 shuffle 的 audited attention-mask 周期合计 440 个有效 token。

10 个单步延迟的算术均值/中位数分别为 native `70.354/70.093 ms`、candidate `172.812/173.011 ms`；nearest-rank p95 分别 `73.118/174.968 ms`，min–max 为 native `67.764–73.118 ms`、candidate `170.602–174.968 ms`。以 20 个有效样本和 440 个有效 token 除以 10 步同步时间总和，native 为 `28.428 sample/s`、`625.410 token/s`，candidate 为 `11.573 sample/s`、`254.612 token/s`；candidate/native 平均延迟为 `2.456x`，candidate/native 吞吐为 `0.407x`。统计使用 10 个原始单步样本的算术均值、中位数和 nearest-rank p95，不排除测量点。Swift Trainer 汇报的 `memory(GiB)` 两边均为 `1.77`；精确 allocator 同步边界 live/reserved 峰值不可观测。L5 已满足测量门槛，但此结果显示该 candidate 配置慢于 native，不能解释为性能优于原生。测量回调、脚本、计时 JSON 和 job 日志位于 `$TASK_STATE/runs/bert-base-real-size-l5-world1-20261011-r1/`，JIT cache 复用已完成 correctness run 的同 ABI/源/编译器配置缓存。

本次基线为 `64afae887dc52ef19811f446685d954bf98eccc9`，工作树干净；真实 NPU worker 的 oracle job 2356 与候选 job 2360 均从 step 1 完成至 step 12；job 2363 的只读 comparator 检查全部 12 步及双方完整恢复轨迹，标记 `BERT_SEQCLS_L3_PAIRED_INPUT_INITIAL_PASS`。native 初段/末段 step 1–3 与 10–12 平均 loss 为 `0.84811014 → 0.68229463`；candidate 为 `0.72807539 → 0.60170611`。初始参数和 buffers 精确一致，12 步逐批 `input_ids`、`attention_mask`、labels 完全一致。每侧都从 checkpoint-4 在新进程恢复到 step 12；模型、optimizer、scheduler、trainer、RNG 文件齐备，global step、optimizer/scheduler 和 806 个恢复状态数组与各自保存状态相符，恢复后的下一个 batch 与不中断 step 5 精确一致，step 5–12 各自运行时轨迹精确续接。

两侧的参数、实际计算 batch、前向输出、梯度和 optimizer state 设备记录均为 `npu:0`。candidate runtime/provider 为 Jittor shim/ACL，两个完整运行阶段均启用 `backend_fallback=error`，fallback start/end 均为 0；native 身份为 torch_npu，通用 fallback counter unavailable/unknown。跨运行时 logits、gradient 与 optimizer 浮点差异（最大绝对差分别约 `0.7260`、`0.13237`、`0.02189`）保留在比较 JSON 中作诊断；按当前收敛协议不将其单独作为 L1–L3 失败阈值。该结论只适用于此锁定的 BERT-base、full-tuning 和公开 SFT 配置，不能推及其它 encoder、模型或 tuner。此训练带全量 observer，步时不代表稳态性能；L5 未运行。较早 tiny BERT 的 GELU 严格对拍失败及其根因记录仍保持原判。

运行键 `bert-base-real-size-seqcls-world1-20261010-r1` 因 candidate 被重复提交为 jobs 2357/2358 并共享输出/JIT 目录而失效；2358 在审计目录创建处报 `FileExistsError`，2357 随后取消，依赖 job 2359 也取消。新的 `...-20261011-r2` 使用独立输出及 JIT 缓存，并有原子 owner lock。只读比较器 2361 缺少 pair 入口，2362 的 shell 检查带入字面引号；这两次 harness 失败未重跑训练。修正后的 2363 完成配对检查。所有失败原始日志保留在 state，没有追溯改判。真实权重盘点及下载、原生与候选训练/恢复快照、检查器和 JSON 均未版本化，位于 `$TASK_STATE/runs/bert-base-real-size-seqcls-world1-20261010-r1/` 与 `...-20261011-r2/`。

## arange dtype 与 NPU 设备语义回归（2026-10-10）

针对 BERT `position_ids` 暴露出的 Torch 整数 arange 默认 int64，新增了纯整数边界时传入 int64 的工厂适配，并将测试设备名改为真实的 `npu`。原生 torch_npu oracle job 2181 通过 NPU `.is_cuda == False` 与 float64 arange 检查；job 2185 在 `npu:0` 上通过 9 种 arange 语义（浮点 start/stop/step、NumPy 浮点 step、tensor 边界/step、float64、float16、NumPy int64 stop）。原生侧 fallback 计数不可得。

严格 ACL job 2186 使用 `backend_fallback=error`，float64 之前的 6 个 float32 arange 构造断言通过；float64 执行在 CANN 9.0 `Expand` workspace 查询失败，运行时列出的支持类型不含 DOUBLE。失败之后的 float16 与 NumPy int64 stop 未执行。候选 fallback 计数为 0，但整体失败，不能记为兼容通过。为验证 DOUBLE 临时增加的 ACL dtype 映射与白名单代码已撤回，不能宣称该 ACL 路径支持 float64 arange。

另一个严格 ACL 回归 job 2184 中，NPU `.is_cuda` 断言通过，随后 median 梯度触发未注册 fused `reindex_reduce/add`，严格模式捕获 11 次 fallback 尝试，整组结果为 33 passed、10 failed。该证据说明当前 median backward ACL 缺少该 fused kernel；没有声称 median 梯度兼容。job 2180 首次整组回归还记录了 `.is_cuda` 语义错误与异步失败；job 2182 编译中止，未运行测试。这些失败均保留在原始运行记录中。

以上是窄范围兼容回归，不赋予 ms-swift 功能面 L0–L5 通过。运行目录：`$TASK_STATE/runs/torch-npu-device-semantics-oracle-20261010-r1/`、`torch-arange-int64-acl-regression-20261010-r5/` 至 `...-r7/`、`torch-arange-full-semantics-oracle-20261010-r1/` 与 `torch-arange-full-semantics-acl-20261010-r1/`。

## IA3 双卡 optimizer state 驻留补证（2026-10-08）

标准 `adamw_torch` 的 IA3 双卡审计已证明 optimizer moments 在各自 NPU，但每 rank 的 14 个 scalar `step` tensor 在 CPU，未满足“所有 optimizer state 物理驻留 NPU”。独立运行键 `ia3-world2-fused-optimizer-audit-20261007` 使用公开 Swift SFT 的 `adamw_torch_fused` 配置，native job 1630 先于严格 ACL candidate job 1631 完成。两 rank 的参数、真实 batch、梯度、更新前后参数与 optimizer state 均有设备记录：原生与候选各自 42/42 optimizer state tensors 全在对应 NPU；候选 HCCL world size 为 2，两个 rank 的 backend 为 ACL，完整观察窗 fallback count 均为 0；原生通用 fallback counter 不可用，记 `unavailable/unknown`。

只读数值审计 job 1665 确认每 rank 的初始 14 个 IA3 参数 exact，batch 与冻结参数/buffers exact，14 个梯度最大差 `3.7253e-9`，更新前后及最终参数最大差均为 0，14 个 trainable tensor 都产生非零梯度和更新。该运行仍不能授予完整 L2：探针未保存 forward output 数组；候选 `optimizer.pt` 是该 Torch shim 的 portable pickle 容器而非 PyTorch ZIP，原 comparator 错把容器格式当作 optimizer 数值缺失，未能通过 shim loader 恢复并比较 optimizer state 值。报告因此保留一个失败项，optimizer checkpoint 数值未比较；没有推出 checkpoint 跨运行时可移植、L3 恢复或完整 L0–L2 通过。原始审计 JSON 位于 `$TASK_STATE/runs/ia3-world2-fused-optimizer-audit-20261007/numeric-audit-v3.json`，日志与逐 rank 快照在同目录。

## IA3 fused AdamW 前向/optimizer 数值与驻留补证（2026-10-10）

运行键 `ia3-world2-fused-value-audit-20261010-r2` 在真实单机双 NPU 上使用公开 `swift.cli.sft`、IA3、FP32 和 `adamw_torch_fused`。native torch_npu oracle job 2189 先运行，strict ACL candidate job 2190 后运行；比较器 job 2193 只读取已生成的快照，不重跑模型。此前 r1 仅在 native harness preflight 失败、未运行模型；比较器 2191/2192 的路径及 rank 记录错误继续保留，不改判。

在两 rank 上，14 个初始可训练参数、22 个冻结参数/buffer 与 4 个实际 batch 张量逐项精确一致。每 rank 2 个前向输出最大绝对差 `4.76837158203125e-7`，14 个梯度最大差 `3.725290298461914e-9`，42 个 optimizer state 数值最大差 `3.4924596548080444e-10`；预先锁定的浮点容差为 `atol=1e-6, rtol=1e-5`。更新前、更新后和最终参数快照均在预置容差内（最大绝对差为 0）。job 2193 的 `numeric-audit.json` 汇总 20 类 rank/阶段张量比较，0 失败。

候选 observer 对每个 rank 的参数、计算输入、前向输出、梯度及 optimizer tensor 执行 live-device 检查，确认位于该 rank 绑定的 NPU；候选 runtime/provider 为 Jittor shim/ACL，HCCL world size 为 2，严格 `backend_fallback=error` 且完整观察窗 fallback count 为 0。native runtime/provider 为原生 torch/torch_npu；没有通用 fallback counter，故记 `unavailable/unknown`。该补证关闭旧 fused optimizer 单步探针“forward 数组缺失、optimizer 数值未比较”的缺口，但只覆盖单步 L1/L2 证据；四步训练及 checkpoint-3 全新进程恢复仍由 `ia3-world2-fullstate-fused-l3-r3-20261008` 独立证明，不能拼接成新的跨运行 checkpoint 对拍。标准 `adamw_torch` profile 的 CPU scalar step 状态问题仍未解决，L5 真实尺寸稳态性能未运行。原始证据未版本化，位于 `$TASK_STATE/runs/ia3-world2-fused-value-audit-20261010-r2/` 和 `$TASK_STATE/runs/ia3-world2-fused-value-compare-20261010-r2/`。

## PEFT DoRA 公开 SFT 历史严格对拍记录（2026-10-10）

运行键 `dora-world1-public-audit-20261010-r5` 基于 integration HEAD `2a8736d755dec83f392110017686f8704d6257a0`，使用 ms-swift 4.5.2、PEFT 0.17.1、tiny LLaMA、FP32、公开 `swift.cli.sft`、PEFT LoRA `use_dora=true`、q/v targets、rank 8、alpha 16、fused AdamW 和固定三步数据。原生 torch_npu oracle job 2269 先完成，严格 Jittor ACL candidate job 2277 随后完成三步，worker comparator job 2278 检查并判失败；前面 r1–r4 候选均在有效模型训练前退出，保留为 harness/JIT 启动失败，不计作兼容结果。

两侧 12 个可训练初始参数、22 个冻结参数/buffers 和三步实际 batch 精确相等。step-0 前向值最大绝对差 `4.76837158203125e-7`，首步 loss 差 `4.76837158203125e-7`；但首步梯度最大差 `0.019399959594011307`，后续轨迹扩大，最终参数最大差 `0.0054405564442276955`、前向最大差 `0.00379335880279541`、更新最大差 `0.0020000040531158447`、optimizer state 最大差 `0.002043604850769043`，逐项比较不满足固定容差。完整 loss、参数、梯度、更新、optimizer 差异和失败条目见 worker `comparison.json`。两侧安装的 `peft/tuners/lora/variants.py` 与 `layer.py` 源码哈希相同；现有证据尚不能确定梯度偏差的算子根因，不能放宽阈值或把该运行改判通过。

候选实际使用 shim/ACL、HCCL world size 1、`backend_fallback=error`，起始与结束 fallback 计数均为 0；observer 记录的参数、计算输入、梯度和 optimizer 张量均在 `npu:0`。但两侧 `devices.json` 都缺少三步 `forward-output` 设备事件，尽管另有三份前向值快照，因此输出的物理设备证据不完整。这是旧严格数值对拍协议下的历史结果，保留其差异和失败结论；新协议不再仅因跨实现浮点差异判训练 L1–L3 失败，但旧运行因缺少前向输出驻留事件仍不能补判 L1/L2 通过。完整恢复 L3 未运行，公开端到端 L4 未通过，L5 未运行。PEFT DoRA 属于单独互操作面，不代表 ms-swift 自有 LoRA 或整个 ms-swift 的状态变化。

原始日志、插件、比较器及快照未版本化，位于 `$TASK_STATE/runs/dora-world1-public-audit-20261010-r5/`；r3 已用修正后的 forward hook 在新运行键补齐设备事件。若后续继续定位旧运行中的跨实现梯度差异，需在新运行键记录裁剪前梯度和裁剪系数，结果仅作诊断；不得因协议变化追溯改写 r5 的严格对拍失败。候选 JIT 首次编译后正常完成，不需为启动问题重跑 r1–r5。

DoRA 反向路径诊断（2026-10-10）：运行键 `dora-gradient-path-audit-20261010-r4` 的 worker 比较显示，四个 q/v 投影的输入和完整模块输出在两侧逐位一致，输出 cotangent 相对 L2 差低于 `3e-7`；当时看到的投影参数梯度差集中在 v_proj。随后 `dora-shape-gradient-probe-20261010-r2`（job 2297）在 `[1,22,64]` 输入上先跑原生 torch_npu，再跑严格 ACL，按 PEFT 0.17.1 `DoraLinearLayer` 公式重算四个投影。两侧各 52 个公式张量设备记录均为 `npu:0`；候选 `backend_fallback=error`，完整公式运行 fallback 计数为 0。worker 比较 job 2299 与 2300 只读取快照：两侧重算输出均与 R4 实际 q/v 模块输出逐位相同，重算的 B 与 magnitude 梯度两侧也逐位相同。

这还没有定位原始 L2 偏差。R4 的梯度快照由 `on_pre_optimizer_step` 记录；该回调在 Transformers Trainer 的 `max_grad_norm=1.0` 全局裁剪之后执行。因而将独立公式的未裁剪梯度直接与 R4 快照比较不能作为反向根因证据。下一步需在新运行键中记录裁剪前梯度及实际裁剪系数，再用同一公开训练步比较；此前“偏差来自投影层反向算子”的推断撤回。R1/job 2296 候选路径映射错误、R2 配套比较 job 2298 数组路径错误均在模型/公式计算前失败并原样保留；它们没有改变 DoRA 的失败判定。原始产物未版本化，位于 `$TASK_STATE/runs/dora-gradient-path-audit-20261010-r4/`、`$TASK_STATE/runs/dora-shape-gradient-probe-20261010-r1/`、`...-r2/`、`$TASK_STATE/runs/dora-shape-gradient-compare-20261010-r1/` 与 `...-r2/`。

### 新协议下的公开八步收敛和设备审计

运行键 `dora-world1-convergence-audit-20261010-r3`（native torch_npu job 2321 先于 strict Jittor ACL candidate 同一公开 SFT 作业中的 candidate 阶段；worker comparator job 2322 只读取快照）复用 r1/r2 锁定的模型、数据、初始 DoRA adapter 和八步训练窗口，修正 full-model forward hook 的设备记录，并让预检与 launcher worker 使用相同 rank cache。oracle/candidate 两侧初始可训练状态均与 immutable lock 精确相同，22 个冻结参数/buffer 和八个逐步实际 batch 结构、shape、dtype、有限值检查通过；每步 forward、梯度、更新与 optimizer state 快照结构及有限值检查通过。两侧均通过各自收敛门槛：oracle 初段三步平均 loss `4.846355597`、末段三步 `4.702533407`；candidate 分别为 `4.845908323` 和 `4.702208517`。

两侧完整设备事件均覆盖 `npu:0` 上的输入、输出、梯度、optimizer state、训练参数初末态及可训练参数；候选确认 Jittor shim/ACL、严格 `backend_fallback=error` 和完整观察窗 fallback count 0。补充快照审计 job 2329 对原 r3 产物只读检查：两侧八步每一步均有 12/12 个有限且结构匹配的梯度/参数，12/12 个可训练参数均发生实际更新，36 个 optimizer state 数值均有限。跨实现 loss、前向、梯度、更新与 optimizer 浮点差异均保留作诊断，不作训练失败阈值；比较器最终标记 `PUBLIC_SFT_EIGHT_STEP_AUDIT_PASS`。因此该锁定的 DoRA 公开 SFT case 当前有 L0–L2 证据，旧三步运行不追溯改判。L3 完整 checkpoint 新进程恢复仍未运行，故 L3 为 not-run、L4 未通过；L5 未运行。原始证据未版本化，位于 `$TASK_STATE/runs/dora-world1-convergence-audit-20261010-r3/`、`$TASK_STATE/runs/dora-world1-convergence-compare-20261010-r4/` 与 `$TASK_STATE/runs/dora-world1-convergence-docgate-20261010-r1/`。

本次结构门禁 job 2330 的布局检查通过；worker 的 `tests/structure` 结果为 1,382 passed、8 skipped、5 failed、24 warnings，历时 19:30。四项执行失败为 pool/adaptive pool 的 fused `reindex/void` ACL variant 未注册，严格模式拒绝 CPU fallback；另一项 `test_process_mode_contract` 的子进程触发 `jit_utils` 首次重建后以退出码 3 结束，pytest 未收集该子进程用例。这些门禁问题不改变 DoRA 运行结论；完整日志和快照补验记录位于 `$TASK_STATE/runs/dora-world1-convergence-docgate-20261010-r2/`。r1 job 2329 仅触发 JIT cache 更新后退出，未运行 pytest。


### 新协议下 PEFT DoRA 完整恢复

锁定公开入口为 `swift.cli.sft`，PEFT DoRA q/v、FP32 fused AdamW，固定 tiny LLaMA、数据和初始化，batch=1、累积步数 1、连续 8 步，checkpoint-2 保存完整状态；全新进程从 checkpoint-2 恢复并继续到 step 8。训练运行键 `dora-world1-resume-audit-20261010-r5` 的 Slurm job 2336 先完成原生 torch_npu oracle 连续/恢复，再完成 strict ACL candidate 连续/恢复；只读 worker comparator `dora-world1-resume-compare-20261010-r2` job 2338 标记 `DORA_PUBLIC_SFT_FULL_RESUME_L3_PASS`。前一比较器 job 2337 的事件 schema 假设错误保留为 harness 失败，没有改动训练产物；此前恢复运行键 r1–r4 也各自保留失败断点。

四条轨迹的锁定 loss 窗口都各自下降。连续训练首/末三步平均值：oracle `4.846355753 → 4.702533407`，candidate `4.845908323 → 4.702208517`；恢复 step 3–5 与 step 6–8 的均值：oracle `4.782141207 → 4.702533247`，candidate `4.781852880 → 4.702208517`。两侧 checkpoint-2 均有 adapter、optimizer、scheduler、trainer 与 RNG 文件，global step 为 2。恢复时 optimizer、scheduler 及 Python/NumPy/CPU/NPU RNG 与各自 checkpoint 精确匹配，RNG 在首个恢复 forward 前确认；后续六个 batch 和 checkpoint 初始可训练/冻结状态与各自不中断轨迹精确相同，每个 runtime 的连续 8 步与恢复 6 步路径分别检查 14 次实际参数更新，均通过。

两侧连续阶段各记录 8 组、恢复阶段各记录 6 组实际输入、forward 输出、梯度、optimizer state 和可训练参数事件；全部张量设备为 `npu:0`。候选 602 个连续阶段与 468 个恢复阶段张量均报告物理 location=device；候选全阶段 `backend_fallback=error`、fallback=0。native torch_npu 没有已验证通用 fallback counter，记为 unavailable/unknown。跨 runtime 浮点差异仅保存为诊断，不作 L3 判据。该锁定用例现有 L0–L4 证据；L5 未运行，不推广到其它 PEFT tuner、模型或完整 ms-swift。原始产物保存在 `$TASK_STATE/runs/dora-world1-resume-audit-20261010-r5/` 与 `$TASK_STATE/runs/dora-world1-resume-compare-20261010-r2/`；中间失败运行均保留各自目录。

本次结果文档提交前，NPU worker 结构门禁 job 2339 运行 9:03：1,383 passed、8 skipped、4 failed。四项均为 pool/unpool 结构用例触发 ACL 未注册的 fused `reindex/void` variant，严格策略拒绝 CPU fallback；另有 2 项因 worker 缺少 `pytest-xdist` 被标为 `other` skip。该门禁失败与 DoRA checkpoint 恢复无关，未以通过报告。布局检查与 `git diff --check` 通过。

## BERT-base sequence-classification 公开推理 GELU 分歧（2026-10-11）

使用真实尺寸 BERT-base sequence-classification 的 checkpoint-12 与同一 tokenizer，按公开 `swift infer --infer_backend transformers --task_type seq_cls` 跑 6 条样本、batch=2、max_length=128。Native torch_npu job 2366 先于 strict ACL candidate job 2367；两侧各返回 6 条结果，输入/tokenization 与预测标签一致，参数、输入和 logits 均记录为 `npu:0`。candidate 的 `backend_fallback=error`，观察窗口 fallback 起止均为 0。worker comparator job 2368 检查 102 项，其中 20 项 logits/log-probability 数值超过固定 `atol=1e-5, rtol=1e-4`，故这条锁定公开推理的 L1 与 L4 均记 failed；训练 L2/L3 不适用，L5 未运行。

为定位首个差异，独立 trace 键 `bert-base-seqcls-infer-trace-world1-20261011-r3` 的 native/candidate jobs 2371/2372 在 110 个节点中确认 embedding 输出精确，`layer00.intermediate_dense` 的差异仍在阈值内，而首个超阈值位置是 `layer00.intermediate_activation`（max abs `4.73976e-4`）。随后 GELU 隔离键 `bert-base-gelu-isolation-world1-20261011-r1` 将 native trace 中完全相同的 `[2,22,3072]` float32 dense 输出直接送入两侧：native job 2377 先于 strict ACL candidate job 2378；两个输入 SHA256 均为 `e55e287d8b49a647bd63a4d20eb8340530ce15c720ac4ec020d231ef49faacf3`，输入和输出在 `npu:0`。candidate fallback policy 为 `error`，fallback 0→0。worker comparator 2380 在同一输入上得到 max abs `4.7330186e-4`、RMSE `2.5827478e-4`，超出锁定容差；首次 comparator job 2379 因结果目录写错而在读取数组前退出，作为 harness 失败保留，不纳入判定。

源码路由核对显示候选 `nn.gelu` 的 ACL 注册实现只接受 `approximate='none'`，native runner 调用 `aclnnGeluV2` 且传参 `0`；已有独立 NPU GELU route/contract 探针进一步观测到本机 torch_npu 的 `approximate='none'` 与 `'tanh'` 输出相同，并符合 tanh 公式，而候选 ACL 路径按 exact/erf 语义工作。因此根因收敛为两运行时 GELU 数值语义不一致，且该差异能单独解释当前 BERT-base 首个层内分歧。没有更改 shim/API、没有调宽容差；历史严格失败仍有效。运行脚本及原始证据位于 `$TASK_STATE/runs/bert-base-seqcls-public-infer-world1-20261011-r1/`、`bert-base-seqcls-infer-trace-world1-20261011-r3/`、`bert-base-gelu-isolation-world1-20261011-r1/` 与 `bert-base-gelu-compare-world1-20261011-r2/`。
### 2026-10-11：IA3 公开推理与 Swift Prompt 路由盘点

当前基线 `5c4d291ba2b213fa8b90dd652e9dd6336d7b05d5` 的新键 `ia3-public-infer-current-world1-20261011-r3` 用 ms-swift 4.5.2 公开 `swift infer` 加载 tiny 2-layer LLaMA 与 Swift IA3 checkpoint-1。原生 torch_npu job 2386 先于候选；候选 job 2387 使用 ACL strict `backend_fallback=error`；worker 只读 comparator job 2389。双方完成 4 条固定请求、每条 greedy 生成 4 token。37 个数组快照全部匹配：加载后的全部参数、16 个 forward 输入和 4 个生成 token ID 精确相等；2,048 个 logits 值最大差 `0`（固定门槛 `atol=1e-5, rtol=1e-4`）。两侧每次 forward 的模型参数、真实输入、输出 logits 和 token tensor 均记录 NPU；候选同步物化后参数/计算张量 `location=device`，process observer 记录 ACL、world-size=1、起止 fallback=0。此锁定用例 L0/L1/L4 通过，训练 L2/L3 不适用，L5 未运行。候选首次生成在缓存中编译耗时约 77 秒，此值不属于稳态计时。

失败证据保留：job 2383 缺少 `srun` overlap step，模型未加载；r2 candidate job 2385 的 observer 在 Jittor lazy graph 物化前要求 `location=device`，输入已为 `npu:0/device`，根因是读取顺序；r3 comparator job 2388 把前向前的 `loaded-parameters.json` 初始惰性元数据纳入物理驻留检查。observer 已在 `jt.sync_all(True)` 后审计，比较器仅检查 forward 后的参数文件并在 job 2389 通过。r1/r2/r3 原始产物与每次错误日志在 state 运行目录保留；旧 strict 数值失败未改判。

ms-swift 4.5.2 tuner source inventory 发现 `swift.tuners.mapping` 注册 `SwiftTuners.PROMPT`，但 `swift.tuner_plugin.mapping` 没有 Prompt，SFT `TunerMixin` 因而把该 tuner 交给 `prepare_adapter()`，该函数没有 Prompt 分支并以 `Unknown tuner_type` 结束。故当前版本没有公开 SFT Prompt 路由。单独的 `Swift.prepare_model` direct API 已在当前源基线上通过 tiny NPU 训练定位：运行键 `swift-prompt-direct-api-current-world1-20261011-r5`，native job 2403、strict ACL candidate job 2404、worker 数值/结构诊断 job 2405。Prompt 参数由 `prepare_model` 在 CPU 构造，探针随后显式 `model.to('npu:0')`；因此该 L0–L2 结论不覆盖未显式迁移的配置路径。两侧在同一固定四样本 NPU batch 上训练 12 步，各自 loss 首三步/末三步均值从 native `4.8643740` 降到 `4.8478611`、candidate `4.8643740` 降到 `4.8478613`。candidate ACL `backend_fallback=error`，fallback 0；每步参数、batch、logits、梯度、更新后参数和 momentum state 的已记录张量均驻留 NPU，冻结参数不变、Prompt 参数有更新。比较器保存的 loss/gradient/parameter/momentum 数值差只作诊断，没有数值阈值；原 runner 未保存 forward logits 数组，故 logits 差异不可回溯。该 12 步记录形成时 L3 未运行；SFT CLI Prompt route 缺失，所以该 direct API 不构成 L4。旧 2026-10-05 SHA `8ce6996` 探针不继承。

此前独立尝试均保留在 state：r1 native job 2395 证实 Prompt 参数在 `prepare_model` 后仍位于 CPU，L0 未通过；r2 native job 2396 完成训练但审计器混用 `state_dict()` 与 `named_parameters()` 键，末尾审计失败；r3 strict candidate job 2400 在惰性 batch 物化前读取 `location()`，未进入前向；r4 strict candidate job 2402 在惰性 loss 物化前读取 `location()`，未进入反向。r3/r4 的 native oracle jobs 2397/2401 均先通过。r5 只调整物化/审计顺序后两侧均通过，未改数值门槛或回溯改判旧失败。

### Swift Prompt direct API 新收敛协议 checkpoint 恢复（2026-10-12）

Native oracle `swift-prompt-direct-api-checkpoint-world1-20261012-r2`（jobs 2538/2549）与 strict ACL candidate 的 continuous 段（key `...-acl-r1`，job 2553）均使用 tiny LLaMA、Prompt dim=64/length=2、固定四条 token 序列、FP32、SGD momentum 0.9、constant scheduler、16 steps 和 step-8 checkpoint。Native gate 在 candidate 启动前确认 continuous 与 fresh-process resume 各自 loss 窗口下降、step-8 checkpoint 中的 model/optimizer state 精确对应保存边界、fixed-batch cursor 与 RNG/scheduler 恢复检查通过；两侧的 cross-runtime 与同 runtime 浮点轨迹只记录诊断，不设精确或容差门槛。

Candidate job 2553 在 ACL NPU 上完成并保存了 16 步 continuous 轨迹、step-8 checkpoint 和 step 快照；其后 fresh-process 阶段先因 probe 调用 shim `_OptState.clear()` 而失败，这是 harness 错误。修正版不重跑 continuous，而以新 resume-only key `swift-prompt-direct-api-checkpoint-world1-20261012-acl-r2`（job 2558）读取既有 step-8 candidate checkpoint，在新进程续训 step 9–16。只读配对审计 job 2560 通过：native/candidate 的初始 model state 与 batch identity 相同；每侧 checkpoint 的 step-8 model/optimizer arrays 与其连续轨迹快照精确对应；continuous loss 首末三步均值分别为 native `4.88290278→4.85498110`、candidate `4.88290278→4.85498047`，resume 分别为 native `4.86900727→4.85498095`、candidate `4.86900727→4.85498047`。Candidate ACL 使用 `backend_fallback=error`，起始及结束 fallback 计数均为 0；NPU state/device、scheduler、batch cursor 与 RNG 恢复断言通过。

此配置达到 direct-API 锁定 case 的 L0–L3；它不提供公开 SFT CLI 端到端证据，因 ms-swift 4.5.2 当前无 Prompt SFT route，L4 不适用。既有 Prompt generation native oracle 在 rotary position 长度处失败的 L1 结果不变；L5 未运行。candidate job 2551 的 shim import 失败、gate jobs 2540/2544 的 harness 断言失败以及 job 2553 的 resume harness 失败均原样保留，未改写历史严格对拍结论。完整脚本、checkpoint、日志与逐步数组未版本化，位于 `$TASK_STATE/runs/swift-prompt-direct-api-checkpoint-world1-20261012-r2/`、`...-acl-r1/`、`...-acl-r2/`。

### Swift Prompt direct API 确定性生成失败（2026-10-11）

新运行键 `swift-prompt-direct-api-infer-world1-20261011-r2` 的 native torch_npu oracle job 2408 在 Ascend worker 上以固定四行 token 输入调用 `Swift.prepare_model(PromptConfig(...))` 后执行 greedy `model.generate(max_new_tokens=4, use_cache=False)`，在首个生成前向失败。Llama decoder 收到长度为 6 的 prompt 注入后 hidden/query 序列，但 rotary position embedding 只有原输入的 4 个位置，错误为 `The size of tensor a (6) must match the size of tensor b (4) at non-singleton dimension 2`。只读源码核对 ms-swift 4.5.2 `swift/tuners/prompt.py`：Prompt wrapper prepend embeddings 并扩展 attention mask，没有扩展 `position_ids`；Transformers 4.57.6 `GenerationMixin.prepare_inputs_for_generation` 从原 attention mask 构造长度 4 的位置，因此与 Prompt 后的长度 6 不匹配。

这是 native oracle 对该 direct generation 调用的失败，不能归因于 Jittor/ACL。按 oracle-first 规则，candidate job 2409 因 `afterok` 前置失败未启动，比较器 job 2410 未执行并已取消；没有 Prompt 生成的 candidate 通过证据。更早的 r1/job 2407 同样观察到位置长度栈，但 runner 在作业运行期间被改写，故只留作 harness 失败，不作正式运行证据。该结果不改变上文已完成的 Prompt 训练 L0–L2；Prompt `model.generate` 推理 L1 记 failed（native oracle），公开 Prompt CLI 路由仍不存在，L4 不适用，L5 未运行。原始脚本、哈希、日志与 Slurm 状态均未版本化，位于 `$TASK_STATE/runs/swift-prompt-direct-api-infer-world1-20261011-r1/` 与 `...-r2/`。未改生成配置、容差或 ms-swift 源码，也未启动 ACL 候选。


### IA3 新收敛协议 native oracle 失败（2026-10-11）

运行键 `ia3-world2-convergence-resume-20261011-r2` 采用新验收合同：tiny LLaMA、Swift IA3、FP32 fused AdamW、固定 fixture `9230f3017c58f8cb5190bb18d4cbd49f5cc0fd94d206163561bfaa94442cbcca`、HCCL world-size 2，连续 8 步并保存 checkpoint-4，再由新进程恢复至 step 8。Native torch_npu job 2412 在 `cscg-hw01` 实际完成连续与恢复两段公开 `swift.cli.sft`，并生成两段 checkpoint 与 rank 快照。

新协议要求两侧各自先证明训练窗下降。Native 连续训练 step 1–3 均值 `4.84470685`、step 6–8 均值 `4.85686668`；恢复轨迹 step 5–6 均值 `4.84205961`、step 7–8 均值 `4.84973407`，两段均未下降。此锁定学习率/数据配置因此不能进入 candidate 对拍或授予新协议下的 L2/L3/L4；矩阵保留原四步严格数值协议 L0–L4 结果，不追溯改判。Candidate job 2413 因 native 前置失败处于 `DependencyNeverSatisfied`，随后取消；比较 job 2414 随之取消。

Native 前置 gate 同时发现 harness 计数问题：事件表末尾另有一个 `train_loss` 汇总日志，按 `step` 做字典覆盖会遮掉实际 step-8 `loss`；进程目录有一个 torchrun 父进程和两个 rank 记录，不能用文件总数等于 2 判失败。两者均为 gate 逻辑缺陷；逐 rank 设备事件检查未发现设备驻留错误，runtime context 记录 native torch_npu 与 HCCL。独立于 gate 缺陷，原始逐步 loss 已清晰证明此训练配置不满足新收敛条件，故不启动 candidate 是正确的 oracle-first 结果。

先前 r1/job 2411 因调度脚本写入共享 `current-job-id` 而未满足 `host-env.sh` 所需 `current-job-id-<jobid>`，在 Python/模型启动前退出；保留其日志，不复用该运行键。r2 的脚本、原始日志、checkpoint、事件和设备快照均未版本化，位于 `$TASK_STATE/runs/ia3-world2-convergence-resume-20261011-r1/`、`...-r2/`。

### IA3 独立收敛与恢复验收通过（2026-10-11）

此前 r2 的 oracle loss 不下降失败仍按原锁定配置保留。新键 `ia3-world2-convergence-resume-20261011-r3` 使用 32 步固定公开 SFT：native torch_npu oracle 2420 完成 continuous 与 checkpoint-16 fresh-process resume；严格 ACL candidate 2421 完成 continuous，候选 resume job 2424 从同一 checkpoint-16 全新进程续训至 step 32。r3 candidate resume harness 因 continuous 与 resume 共用 `JT_LAUNCH_LOGDIR`、launcher 日志以独占模式创建而在模型加载前报 `FileExistsError`；comparator job 2422 因 candidate 前置作业失败而取消。r4 job 2425 使用旧 strict numeric comparator，失败 1,313 个 exact float mismatch（resume trajectory 与 cross-runtime checkpoint step16 参数），该 `verification.json` 原样保留，不改成旧协议通过。

用户确认的新合同要求每个 runtime 各自证明预选 loss windows 下降；跨实现和训练轨迹浮点数值仅作诊断，不用阈值决定通过。只读审计键 `ia3-world2-convergence-resume-20261011-r15-acceptance-audit-v5`、worker job 2436 对 r3/r4 immutable 产物执行完整检查，`failure_count=0`：native continuous steps 1–3/30–32 均值 `4.844706853/4.791386445`，native resume 17–19/30–32 `4.838513374/4.791386445`；candidate continuous `4.844707171/4.791386922`，candidate resume `4.838513374/4.791386922`。初始模型/buffer 及 rank-local step 16–31 batch identity 精确；两个 rank 的 candidate 与 native 参数、输入、forward、梯度和 optimizer state 设备事件均在各自真实 NPU，HCCL world-size=2。候选 ACL strict `backend_fallback=error`，起止 fallback=0。连续与恢复 checkpoint 的 optimizer step/group topology、scheduler/trainer counters、rank RNG/sampler state 与恢复批次续接通过；same-runtime RNG/scheduler continuation max difference 为 0。422 项浮点差异仅入诊断，最大轨迹绝对差约 `4.77e-7`；不调容差，也不追溯重写 2425 的 strict failure。

审计器修正过程的失败和诊断键（2426、2427–2431、2432–2435）各自保留在 state；其中包括 candidate checkpoint 使用 Jittor protocol-4 pickle/`__jt_var__` 数组而不是原生 torch zip 序列化，以及 audit wrapper/参数组假设缺陷。它们不属于训练失败，也不替代 job 2436 的最终验收；r15 完整运行清单、脚本与 report 均在其 state 目录中。此结果只授予此 tiny LLaMA/IA3/FP32/AdamW/HCCL 两卡公开 SFT 锁定用例的 L0–L4；L5 未运行，不能外推至其它 tuner、模型或真实尺寸。


### Swift Adapter 新收敛协议下的恢复状态审计（2026-10-11）

锁定用例仍为 tiny LLaMA、Swift Adapter length=16/GELU、FP32 fused AdamW、公开双 NPU `swift.cli.sft`。Native torch_npu 的 r3 连续/恢复轨迹先运行并各自通过固定 loss 窗口；candidate 使用已有 r5 连续轨迹与 checkpoint-16，R10/job 2453 在新 Slurm worker 进程恢复后续训至 step 32。未重跑 r5/r8/r9。R10 bootstrap `--check` 与 checkpoint read-only extractor 成功；worker gate `resume-state-gate.json` 为 `passed`, `failure_count=0`。

恢复开始时（`on_train_begin`, global step 16、首次续训 optimizer step 前），每个 rank 均捕获实际恢复的 optimizer、scheduler 与 RNG。两组 optimizer group、8 个参数 slot、24 个 tensor state、超参数及状态值与该 rank 的 checkpoint reference 精确匹配；scheduler state 与 rank-local Python/NumPy/CPU/NPU RNG state 精确匹配。两 rank 的 step 16–31 实际 batch 均与各自 uninterrupted 轨迹逐项相同。每 rank 8 个 Swift Adapter trainable 参数；整个恢复窗口的 trainable 参数、计算输入/输出、forward 输出、梯度和 24 个 optimizer tensor state 均物理驻留 `npu:0`/`npu:1`。运行时报告 Jittor shim、ACL、实际 HCCL world size 2，`backend_fallback=error`，起止计数为 0。Candidate loss 窗口 step 17–19 到 30–32 均值为 `4.696953138 → 4.666621208`；native 为 `4.696953297 → 4.666621208`。checkpoint-32 的 trainer global step 为 32，optimizer/scheduler、adapter、rank RNG 文件均存在。此锁定公开训练 case 按新合同 L0–L4 通过；L5 未运行。

旧协议与 harness 失败原样保留：R8 的 strict gate 有 1,298 个 floating `not-exact` 项，涉及连续/恢复浮点轨迹；新合同把这些值用于诊断，不作训练验收阈值，R8 原 gate 仍记录 failed。R9 因 native PyTorch `torch.load` 不支持候选 Jittor pickle，在模型启动前的只读 checkpoint extraction 失败，没有训练。R10 按已验证的 `pickle.load` + `__jt_var__` normalization 路径修正解码并在 worker 端完成本次审计。未调数值阈值，也未改写历史运行结果。原始脚本、日志与状态快照未版本化，目录为 `$TASK_STATE/runs/adapter-world2-convergence-resume-20261011-r8/`、`...-r9/`、`...-r10/`。 报告提交门禁由独立 worker job 2454 完成：布局检查通过，`tests/structure` 为 1,389 passed、6 skipped、0 other skips（1,395 collected，19:53.62）。

### ORPO 新收敛协议双 NPU训练与完整恢复（2026-10-11）

新运行键 `orpo-world2-convergence-resume-20261011-r1` 使用公开 `swift.cli.rlhf --rlhf_type orpo`，tiny LLaMA、Swift LoRA、FP32 fused AdamW、两个 HCCL rank、固定两样本数据、batch=1/rank、max length 32、lr `1e-4` constant scheduler、seed/data_seed 42。native oracle job 2456 与 gate job 2458 先于 candidate；candidate job 2477 在同一 Slurm worker job 中以独立训练进程完成 32 步 continuous 轨迹及 checkpoint-16 全新进程恢复到 step 32。运行源 HEAD 为 `1874c55f47eca28f8e8e2179b1310f202c6a6f0c`，本地源码差异 SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`（JIT include-tree cache filter），目标远端当时可信祖先 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`。Oracle/candidate 使用的固定模型、adapter 和数据 SHA 见两侧 manifest。

两侧分别通过预选收敛窗：native continuous steps 1–3 到 30–32 的 loss 均值 `4.881912→4.390640`，resume steps 17–19 到 30–32 为 `4.612972→4.390640`；candidate continuous `4.881913→4.390640`，resume `4.612972→4.390640`。native gate 与 candidate acceptance audit 均为 0 失败。两 rank 的实际 batch 与锁定 oracle identity、初始 trainable/frozen 参数、每步 loss、前向、梯度、参数更新及 optimizer-state 快照完整且有限；resume 初始参数匹配连续轨迹 checkpoint-16 边界，rank-local step 16–31 batch 与连续路径对应。candidate 全程 `backend_fallback=error`，HCCL world-size 2，输入/输出/参数/梯度/optimizer state 的设备记录均在各自绑定 NPU，完整观察窗 fallback=0。native runtime/provider 为 torch_npu，通用 fallback counter unavailable/unknown。

只读 checkpoint audit job 2490 解析两侧各自 optimizer checkpoint-16，并与同 runtime 连续轨迹更新 16 后的 observer 快照逐项比较：28 个 optimizer state slot、84 个 step/exp_avg/exp_avg_sq 字段、34,844 个元素全部精确一致；所有 optimizer step 和 trainer global step 均为 16，两个 rank RNG 文件以及 scheduler/trainer/adapter checkpoint 文件存在。numeric diagnostics job 2484 检查 480 组跨 runtime 前向/梯度/参数/optimizer 数值数组和 388 组连续到恢复的同 runtime 快照，未发现键/shape/dtype 或非有限值缺陷。批次及 checkpoint 边界按各自语义精确；浮点快照之间存在最多约 `1.49e-7` 的绝对差，跨 runtime 与同 runtime 恢复差均只作诊断，不设通过阈值。L0–L4 仅适用于此固定公开 ORPO 配置；L5 未运行。

原始失败也保留，均不是训练不收敛或 ACL 兼容失败：候选预检 jobs 2475/2476 分别缺 `adapter-zero-dropout` 子目录和 launch 预期的顶层 adapter 文件，均在训练前退出；job 2478 比较器拼错 `trace-shim-*` 路径，job 2479 缺 per-job Slurm marker；job 2480 用 torch_npu legacy reader 解码 JTorch portable pickle，误报 invalid magic；job 2481 导入 shim 时触发 compiler-path 探测失败，未读数据；job 2482 虽已通过数组/optimizer 检查，却到 scheduler wrapper 日志寻找 Swift 完成标记，误报未完成。最终 gate job 2483 改为逐 rank launcher 日志并通过。checkpoint probe job 2485 的 `inspect.py` 遮蔽 Python 标准库模块；job 2486 假设 native/candidate checkpoint 共用根目录；job 2488/2489 的 Slurm 输出目录配置错误。逐项修复后的 checkpoint state audit job 2490 通过。所有脚本、错误报告与作业输出未版本化，保存在 `$TASK_STATE/runs/orpo-world2-candidate-convergence-audit-20261011-r2/` 至 `...-r12/`；training 与 oracle 原始证据分别在各自 run 目录。

后续针对现有 Jittor cache include-tree 修复作了独立 candidate 续验，使用新键 `orpo-world2-candidate-convergence-resume-20261012-r1`，不重跑 native oracle job 2456/2458，也不重跑此前 candidate job 2477。该候选基于 integration HEAD `a79e5706916bbb7da4ddb42e21624514c60bf5f9` 与 dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；同步目标 `origin/2.0-refactor` 的 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该 HEAD 的祖先。候选 job 2570 完成公开双 rank ORPO 连续 32 步与 checkpoint-16 新进程恢复至 step 32；只读 gate-only job 2576 对既有 native/candidate 证据检查 0 失败，不训练模型。Candidate continuous steps 1–3 到 30–32 的 loss 均值为 `4.88191255→4.39063994`，resume steps 17–19 到 30–32 为 `4.61297178→4.39063994`。两 rank 均通过 strict ACL device audit，HCCL world-size=2、`backend_fallback=error` 且 fallback=0；gate 核实精确初态与 batches、checkpoint-16 模型/optimizer/scheduler/RNG/trainer 状态和 optimizer step=16，以及终态 step 32。按新合同这是该候选锁定配置的 L0–L4 通过，不覆盖 L5、其它 tuner/模型或真实多机。早先严格跨实现对拍的结论及其失败条目仍保持原样，仅作独立历史记录。未版本化脚本和原始产物位于 `$TASK_STATE/runs/orpo-world2-candidate-convergence-resume-20261012-r1/` 与 `$TASK_STATE/runs/orpo-world2-candidate-convergence-gate-recheck-20261012-r5/`。

### Qwen3 tiny 新收敛合同下的公开 SFT 与 checkpoint 恢复（2026-10-12）

新运行键 `qwen3-sdpa-convergence-resume-20261012-r2` 以当前集成基线 `9d2aa047bb6abbd4c38ec847606c6045bde701a6`、锁定的本地两层 Qwen3 fixture 与固定 8 行 JSONL 执行公开 `swift sft`。配置为 FP32 bool-mask SDPA、Swift LoRA q/k/v/o rank 4/alpha 8/dropout 0、batch 2、无 shuffle、32 步连续训练并保存 checkpoint-16，再由新进程恢复至 step 32；优化器为 fused AdamW（lr `5e-4`、weight decay `.01`、betas `.9/.999`、eps `1e-6`），constant scheduler，seed 0/data_seed 1。native torch_npu oracle job 2532 先完成，严格 ACL candidate job 2534 仅在其验收产物存在后运行。两侧公开训练的完整 raw evidence 与脚本未版本化，位于 `$TASK_STATE/runs/qwen3-sdpa-convergence-resume-20261012-r2/`。

新合同要求每个 runtime 分别通过预选 loss 窗口，不以跨 runtime 浮点差作训练通过阈值。Native 连续训练 step 1–3/30–32 loss 均值为 `4.828080177 → 4.444386959`，checkpoint 恢复 step 17–19/30–32 为 `4.535408656 → 4.444386959`；candidate 连续训练为 `4.828080336 → 4.444387277`，恢复为 `4.535408656 → 4.444387118`。两侧各 32/16 个训练步均有 forward batch、梯度及 optimizer state evidence；全部数值有限且更新非零。两侧 checkpoint-16 均含 adapter、optimizer、scheduler、RNG、trainer state 与 training args，trainer global step 和恢复起点为 16，optimizer/scheduler 状态与各自运行时保存边界匹配，恢复边界精确。

只读配对审计通过 `QWEN3_PUBLIC_SFT_L0_L3_PASS`：42 组初始全模型参数/buffer（94,160 个值）与 16 组 trainable 初始状态（3,584 个值）均逐项精确相同；continuous 32 批及 resume 16 批 identity 精确。补充的 `resume-state-audit.py` 以各运行时自己的 checkpoint loader 读取 checkpoint-16 scheduler，并与恢复进程首次 optimizer step 前记录的 scheduler 状态逐项匹配；trainer global step 为 16。Native 与 candidate 各自 continuous/resume checkpoint-32 RNG 文件字节相同，分别见 SHA256 `26c79405…e071`、`fe61eddd…ca0d`；next-batch identity 精确，数据不 shuffle 且 LoRA dropout 为 0。Candidate 的 continuous/resume 分别记录 1,700/900 条设备事件，实际训练与 optimizer 状态均在 `npu:0`；launcher 使用 ACL 与 `backend_fallback=error`，观察窗 fallback 为 0。跨 runtime logits、gradient、pre/post-update 与 optimizer state 的最大绝对差约为 `5.98e-7`、`7.64e-8`、`5.98e-7` 与 `2.98e-8`，只作为诊断值，无阈值判定。此结果只覆盖该 fixture、数据、超参数与公开入口，不外推至大权重推理、其他 tuner、L5 或多卡配置。

此前运行键 `qwen3-sdpa-current-2.0-20261009-*` 中三步严格数值对拍的 AdamW 更新差异仍保留为 failed；本次新协议通过不改变旧结果、不调整旧容差。新收敛合同的首次 r1 在模型启动前因 observer 父目录未创建而失败（native job 2528），其 afterok candidate 2530 未运行模型；该 harness 失败保留。r2 为独立新键，不复用 r1 或既有模型运行键。Slurm accounting 当前不可用，作业在队列消失后以 native/candidate acceptance JSON 与配对审计产物核验完成。

### tiny Mistral Swift Adapter checkpoint 恢复门禁（2026-10-12）

仅推进既有 `mistral-adapter-public-eps1e6-20261006` 的 L3，不重跑其 native/candidate 连续 SFT、严格更新比较或推理。新恢复键为 `mistral-adapter-public-resume-20261012-r1/r2/r3`：r1/job 2579 在 worker 上确认 torch_npu `npu:0`、固定 32 行数据摘要、tiny Mistral `model.safetensors` 摘要，以及实际公开目录名 `native-observed/output/checkpoint-2`（trainer global step=2；optimizer、scheduler、RNG、Adapter 权重/config 全在）；随后因 callback launcher 缺失 `ADAPTER_EVIDENCE` 环境变量而在训练前失败。r2/job 2580 修复变量后仍因 launcher 预先创建 evidence 目录，与 callback 的独占创建冲突而在训练前失败。两项为 harness 缺陷，原日志保留，不判为 Swift/Jittor 恢复失败。

r3/job 2581 的 worker 预检通过，真实公开 `swift.cli.main sft --resume_from_checkpoint <.../checkpoint-2>` 进入 Transformers `Trainer.train()`，然后由 `transformers.Trainer._load_from_checkpoint()` 调用 `load_sharded_checkpoint()`；该函数要求 resume 目录根部有 `pytorch_model.bin.index.json` 或 `model.safetensors.index.json`，但 ms-swift Adapter checkpoint 实际保存的是 `default/adapter_model.safetensors` 加 optimizer/scheduler/RNG/trainer 状态，根目录没有完整 base-model shard index，因此在任何恢复更新前抛出 `ValueError: Can't find a checkpoint index`。这是 native oracle 的公开 checkpoint 恢复路径失败，不是 Jittor/ACL 结果；按 oracle-first 未启动 candidate。此 Mistral Adapter 锁定配置的 L3 failed、端到端恢复 L4 failed，原先三步训练和推理的严格通过结论不变。进一步支持需要 Swift checkpoint loader 先重建 base model 与 Adapter，再恢复 Trainer/optimizer/scheduler/RNG；本轮没有用伪造 shard index 改写 checkpoint，也没有改源码。完整 manifest、worker checker、launcher、callback 和输出位于 `$TASK_STATE/runs/mistral-adapter-public-resume-20261012-r1/` 至 `...-r3/`。

### tiny LLaMA 浮点 `inputs_embeds` direct API 训练更新（2026-10-12）

新收敛验收键 `llama-lora-float-embeds-train-20261012-r2` 将此前已通过的浮点 `inputs_embeds.requires_grad` L1 direct API 梯度格推进至 L2。同步前的 target `origin/2.0-refactor` SHA 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是本次运行基线 HEAD `a79e5706916bbb7da4ddb42e21624514c60bf5f9` 的祖先；运行还绑定 manifest 中的 dirty worktree 与候选 source-scope hash。先完成 explicit bootstrap job 2588，再由 worker no-build 预检 job 2589 检查 native 与 ACL 环境、模型/输入/初态清单，随后 native job 2590、严格 ACL candidate job 2591 及 worker 只读比较 job 2592 全部退出码 0。实际小模型 checkpoint 目录名为 `base`；模型文件 SHA manifest、固定 token/input manifest 与 LoRA 初态 manifest 均在 worker 可读；没有下载权重。r1/job 2587 曾在加载模型前因旧候选 cache build stamp 被 `bootstrap --check` 拒绝；这是 harness/cache 预检失败，不判为兼容性失败，本次先显式重建并通过 no-build 检查。

该锁定配置是两层 tiny LLaMA、PEFT q/v LoRA rank 2、FP32 `[1,8,64]` 可微 `inputs_embeds`，标签为输入 token 循环移位；AdamW fused、16 次真实 optimizer 更新，lr `5e-4`、betas `(.9,.999)`、eps `1e-8`、weight decay 0。两侧初始 trainable SHA 精确相同；16 步的参数、输入、梯度及 optimizer state 均在 `npu:0`，native 与 candidate 各自 step 1–4 到 step 13–16 均值 loss 分别从 `4.887218356` 降至 `4.865373850`、从 `4.887218236` 降至 `4.865373611`。ACL candidate 固定 `backend_fallback=error`，起止 fallback 均为 0。worker comparator 核验 48 个初态键精确一致、1008 项结构/键检查通过；1000 个跨 runtime 浮点比较仅作为诊断记录，不设置比较阈值。按新协议，本行 L0–L2 通过；L3 恢复、L4 公开 Swift CLI、L5 性能均未运行。此前的严格对拍失败记录不因本次独立训练配置通过而改判。所有脚本、作业输出与原始 tensor 数据未版本化，位于 `$TASK_STATE/runs/llama-lora-float-embeds-train-20261012-r2/`。


### Qwen3 tiny 公开 greedy batch=2 推理与实测分组（2026-10-12）

本增量继续使用本机已校验的两层 Qwen3 fixture 和 native checkpoint adapter，不下载模型。短预检在 Slurm worker 中核实实际目录名 `base` / `checkpoint-3`、6 个模型文件与 10 个 adapter 文件、输入/初态清单及哈希、`2.0-refactor` 同步基线 `7a18abf295668d9b19da5fa1657f5606e84b65a0`、当前 HEAD `a79e5706916bbb7da4ddb42e21624514c60bf5f9` 和 dirty source diff；shim core hash `7c2d5edd…3294bc` 与既有同指纹缓存相同，`bootstrap --check` 在 `JITTOR_NO_BUILD=1` 下通过，无新编译。worker preflight job 2626 确认 torch 2.10.0/torch_npu 2.10.0 原生侧及 Jittor/ACL 严格 `backend_fallback=error`、fallback=0 候选侧均可用。所有 preflight、gate 与 comparator 检查均经 `srun` worker 执行。

前一运行键 `qwen3-public-infer-batch2-greedy-world1-20261012-r1` 的 native 2615、gate 2622、candidate 2623、read-only comparator 2625 均留下证据：4 条结果、37 个 NPZ、16 个 logits 快照，greedy token、解码文本、输入及加载参数精确相同，logits 最大差 0，ACL 结束 fallback=0。worker 观察到 16 次前向的 batch 维度全部为 1；原因是该键设置 `--write_batch_size 1`，公开 infer 将数据拆成单行 shard，`--max_batch_size 2` 因此未实际覆盖 batch=2。candidate job 2623 只在推理后检查器调用不存在的裸 `python` 时以 127 退出；模型输出完整，比较器通过，不判为模型失败。该键只证明公开单行 greedy 路径，batch=2 仍为未运行。

新键 `qwen3-public-infer-batch2-greedy-world1-20261012-r2` 将 `--write_batch_size` 与 `--max_batch_size` 均锁为 2，以修正前述分组。Native job 2627 的 worker 原生门禁 2628 通过：四条输入形成两个实际 batch，每次 `input_ids` 的 batch 维度均为 2，共 8 次 forward、2 个 batched token snapshot、19 个 NPZ；模型仍是 `base`、adapter 仍是 `checkpoint-3`。Strict ACL candidate job 2629 完成相同公开入口及 4 条结果；两份 process-end 记录均为 ACL、`fallback_policy=error`、`fallback_count=0`。worker comparator job 2630 通过 19 个快照：输入、已加载参数、greedy token ID 和解码响应精确一致，16 个 logits 组最大绝对差 0。候选作业末尾的额外 stdin 检查器以 `NameError: error` 退出 1；推理、结果及严格回退证据均已产生，worker comparator 独立验证后通过，因此保留为 harness-only 失败，不重跑候选。

按本合同，此锁定的单 NPU public greedy batch=2 推理格 L0/L1/L4 通过，L2/L3 对推理不适用，L5 未运行。R1 与 R2 运行键及其退出结果均保留；R1 的单行实测不被追溯写成 batch=2。该证据不覆盖 beam search、sampling、额外 cache 配置、大模型或训练/恢复。原始脚本、worker 日志、NPZ 与 manifest 未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-greedy-world1-20261012-r1/` 和 `...-r2/`。

独立延长生成格使用新键 `qwen3-public-infer-batch2-greedy16-world1-20261012-r1`，将相同四条固定 prompt 的 greedy decode 从 4 延至 16 token。Slurm worker 预检 job 2631、native oracle job 2632、worker 原生 gate 2633、strict ACL candidate job 2634 与 worker 只读 comparator job 2635 均退出码 0；所有检查器在 `srun` worker 内运行。预检确认模型与 adapter 的实际目录名 `base` / `checkpoint-3`、模型下载关闭、输入及初态清单可读；ACL core cache 指纹与已校验缓存相同，no-build 检查未触发新编译。两侧各运行四条输入、64 个生成 token，实际形成两个 batch 且每批大小为 2。比较器核验 67 个快照和 32 次前向：输入及 41 个加载参数精确一致，greedy token IDs 与解码响应精确，最大 logits 绝对差 0。Candidate 两个进程均为 ACL 严格 `backend_fallback=error`，全程结束 fallback count=0。此锁定公开推理格 L0/L1/L4 通过；不推广至 sampling、其他 cache 配置、大模型或 L5。全部原始日志、脚本、NPZ 与 manifest 未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-greedy16-world1-20261012-r1/`。

随后从矩阵中推进公开 sampling 分支，使用新键 `qwen3-public-infer-batch2-topk1-sample-world1-20261012-r1`：预检 job 2636、native oracle 2637、native gate 2638、strict ACL candidate 2639、worker comparator 2640 均退出码 0。Swift 4.5.2 把 `temperature=1, top_k=1` 解析为 `do_sample=True`，因此这条路径会进入 sampling 分支，但每一步只有一个可能 token；它提供可精确比较的采样调用路径，不能外推到多候选 token 的一般采样或跨运行时 RNG 等价。两侧仍是四条 prompt、两个实际 batch=2、16 token/样本。worker comparator 对 67 个快照、32 次 logits/input 前向检查全部通过：输入与加载参数精确、token/响应精确，logits 最大差 0；candidate 两个进程为 ACL、`backend_fallback=error` 且观察窗 fallback count=0。实际模型/adapter 目录仍为 `base` / `checkpoint-3`，离线未下载权重，复用已核验的同指纹 ACL cache。此锁定 sampling top-k=1 公开推理格 L0/L1/L4 通过；一般 top-k/top-p sampling、额外 cache 配置、大模型与 L5 仍未运行。原始证据未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topk1-sample-world1-20261012-r1/`。

随后单独验证非整除数据分组，运行键 `qwen3-public-infer-batch2-partial-final-greedy-world1-20261012-r1` 使用三条已锁定 prompt、`write_batch_size=max_batch_size=2` 和 greedy 4-token decode。短预检 job 2641 通过，核实模型/adapter 目录 `base` / `checkpoint-3`、输入与初态清单可读、离线不下载和 ACL strict no-build cache；native oracle 2642 与 worker gate 2643 成功。Gate 记录两个数据 shard及 8 次前向，observed `input_ids` batch sizes 是 `[2, 2, 2, 2, 1, 1, 1, 1]`，证明末尾单行分组被真正执行。Candidate job 2644 的公开 CLI 已生成 3 条结果、两个 token batch、8 个输入/logits snapshot 和两份进程结束记录；之后复制 runner 的 stdin 后置检查器触发 `NameError: error`，所以 Slurm job 没有成功退出标记。只读 comparator job 2645 改为在 candidate 结束后运行，验证 19 个 snapshot 全部通过：初始参数和对应输入精确、greedy token/响应精确、最大 logits 差 0；两 candidate process 记录为 ACL、严格 error policy 且 fallback=0。该 wrapper 错误归为 harness-only；未重跑 candidate。此锁定不满尾批公开推理格 L0/L1/L4 通过；L5 未运行。未版本化证据位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-partial-final-greedy-world1-20261012-r1/`。

### Qwen3 tiny 公开 top-k=4 单步 sampling（2026-10-12）

从未覆盖矩阵推进多候选采样分支，新键 `qwen3-public-infer-batch2-topk4-sample-step1-world1-20261012-r1` 固定四条 prompt、实际 batch=2、`temperature=1`、`top_k=4`、`max_new_tokens=1`。预检 job 2652、原生 oracle 2653、worker 原生门禁 2654、strict ACL candidate 2655 与 worker 只读 comparator 2656 均退出码 0。预检确认模型目录 `base`、Adapter `checkpoint-3`、输入和初态清单可读，关闭下载；candidate 复用 SHA256 为 `7c2d5edd…3294bc` 的已校验 Jittor core/cache，并启用 `backend_fallback=error`、fallback count=0。所有预检、门禁和比较器均在 `srun` worker 内运行。

Comparator 精确核验加载参数与 prefill 输入；native/candidate prefill logits 最大绝对差为 0。四个 sampled token 两侧恰好相同（`[53,84,79,46]`），但判据不要求跨 runtime 的 RNG/token 精确一致，只分别验证每个 token 属于该侧对应 prefill logits 的 top-4 候选。故此锁定的一步 top-k=4 sampling 公开入口 L0/L1/L4 通过；结果不代表多步 RNG 等价、top-p、其它 cache 配置、大模型或 L5。原始脚本、日志、快照与 manifest 未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topk4-sample-step1-world1-20261012-r1/`。本运行基线 HEAD 为 `a9605c85d738f5f89acf4931913e3050e7153757`，dirty source patch SHA256 见该 run manifest；同步目标仍为祖先 `7a18abf295668d9b19da5fa1657f5606e84b65a0`。

### Qwen3 tiny 公开 near-context 推理（2026-10-12）

从未覆盖的上下文长度边界继续验收。新键 `qwen3-public-infer-batch2-nearctx-greedy4-world1-20261012-r1`：worker 预检 2658、原生 torch_npu oracle 2659、worker 原生门禁 2660、strict ACL candidate 2661、worker 只读 comparator 2662 均通过。短预检核验实际模型目录 `base`、Adapter 目录 `checkpoint-3`、输入/初态清单与已校验权重可读；模型声明 `max_position_embeddings=128`，两条原始 tokenizer 序列各 115 token，CLI `max_length=128`。原生 gate 观察到实际 public-infer 首次输入长度 124、四次前向的 batch 均为 2；因此没有越过 fixture 上限。candidate 复用已校验的 ACL core，严格 `backend_fallback=error`、fallback=0，未下载模型。Comparator 比较 4 次 logits/input 快照、参数、greedy tokens 和公开响应；输入及参数精确一致，生成结果精确一致，logits 最大绝对差 0。此锁定 near-context 推理格 L0/L1/L4 通过；不能推广到长于 128 的模型上下文、其它 cache 配置或 L5。

前置探索键 `qwen3-public-infer-batch2-longctx-greedy4-world1-20261012-r1` 的 job 2657 在短预检器因 manifest 缺少 `temperature` 字段而退出，没有执行模型；该失败属于 harness。预检同时读取到拟用提示的 raw tokenizer 长度 285，而实际 fixture `max_position_embeddings=128`，所以该超长输入不适用于此模型。保留该未运行探索键，不重跑；后续另建上述模型支持范围内的 near-context 键。


### Qwen3 tiny 公开 top-p=0.8 单步 sampling（2026-10-12）

新键 `qwen3-public-infer-batch2-topp08-sample-step1-world1-20261012-r1` 固定四条 prompt，public infer 实际分成两个 batch=2 请求，设置 `do_sample=true`、`temperature=1`、`top_k=0`、`top_p=0.8`、每条生成一个 token。worker 短预检 job 2665 确认实际模型/Adapter 目录为 `base` / `checkpoint-3`，输入与初态清单可读，关闭下载，torch_npu 可见 Ascend910B3；candidate 复用已校验 ACL core SHA256 `7c2d5edd…3294bc`，strict `backend_fallback=error`、fallback=0。预检、原生 oracle 2666、worker 原生门禁 2667、strict ACL candidate 2668 和 worker 比较器 2669 均成功，后四项按 `afterok` 串行。

比较器确认两侧加载参数与 prefill 输入精确相同，prefill logits 最大绝对差 0；每侧四个采样 token 均属于该侧对应 logits 的 top-p nucleus。两侧本次观测到的 token 序列相同，但验收不要求跨 runtime RNG 等价，也不以此宣称一般采样轨迹一致。此锁定的单步 top-p 公开推理格 L0/L1/L4 通过；多步 top-p/general sampling、其它 cache 配置及 L5 未覆盖。离线权重未下载。原始日志、快照与 manifest 未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topp08-sample-step1-world1-20261012-r1/`。本次运行基线 HEAD 为 `f036413f14d163ad7465df8bda86430607d82b4b`，dirty source patch SHA256 为 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。新任务开始时的 upstream HTTPS fetch 卡在网络读取阶段；保留最后可信目标 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`，不将其表述为实时同步结果。


### Qwen3 tiny 公开 top-p=0.8 四步 sampling（2026-10-12）

继续覆盖多步采样分支，新键 `qwen3-public-infer-batch2-topp08-sample-multistep4-world1-20261012-r1` 固定四条 prompt、两个实际 batch=2 请求，设置 `do_sample=true`、`temperature=1`、`top_k=0`、`top_p=0.8`、最多生成 4 token。worker 预检 2671、原生 torch_npu oracle 2672、worker 原生门禁 2673、strict ACL candidate 2674 与 worker 比较器 2675 全部退出码 0，后续阶段以 `afterok` 串联。原生门禁观察到 8 次前向输入，batch 均为 2；每侧实际产生 16 个 token。预检核验实际模型/Adapter 目录 `base` / `checkpoint-3`，输入/初态清单与权重可读，离线不下载；candidate 使用已核验 ACL core SHA256 `7c2d5edd…3294bc`，`backend_fallback=error` 且 fallback=0。

worker comparator 发现两侧加载参数与 prefill 输入精确一致，prefill logits 最大绝对差 0；两侧四步采样序列本次观测恰好相同，但不把它作为跨 runtime RNG 判据。逐步检查每个 runtime 的采样 token 都在本侧对应 top-p nucleus 内，下一步输入均消费前一步输出；对应输入一致的各步 logits 也满足既有数值容差。故该锁定四步 top-p=0.8 public inference case L0/L1/L4 通过；结论不推广至其它采样配置、cache 配置、长上下文或 L5。原始证据未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topp08-sample-multistep4-world1-20261012-r1/`。运行基线 HEAD `9d09cfc89bdc02a39f9592064e403c87e8141125`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。开始该格时 upstream fetch 在 30 秒有界超时内未完成；上次可信目标 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 仅作为已知基线记录，不称为实时远端状态。


### Qwen3 tiny 公开 top-k=16 单步 sampling（2026-10-12）

新模型运行键 `qwen3-public-infer-batch2-topk16-sample-step1-world1-20261012-r1` 固定四条 prompt、两个实际 batch=2 请求、`temperature=1`、`top_k=16`、每条生成一个 token。worker 预检 2678、原生 torch_npu oracle 2679、worker 原生门禁 2680、strict ACL candidate 2681 均退出码 0；真实模型/Adapter 目录为 `base` / `checkpoint-3`，输入与初态清单可读，未下载权重。candidate 使用已校验 ACL core SHA256 `7c2d5edd…3294bc`，`backend_fallback=error`、fallback=0。

原比较器 job 2682 因模板残留的 top-4 数组切片只检查了 4 个候选，误拒绝 token 34；这是 checker harness 缺陷，不能判为兼容失败。比较器专属新键 `qwen3-topk16-sample-step1-comparator-fix1-20261012-r1` 的 job 2683 只读复用了上述原始两侧产物并在 marked srun worker 通过：加载参数和 prefill IDs 精确相同，prefill logits 最大绝对差 0，每侧样本均在自身 top-16 候选集，ACL fallback=0。本次两侧 sample token 相同不构成 RNG 一致性要求。因此该锁定 top-k=16 单步 public inference case L0/L1/L4 通过；不推广到其它 top-k、一般 RNG 等价、多步轨迹或 L5。原始材料位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topk16-sample-step1-world1-20261012-r1/`，修正比较证据位于 `$TASK_STATE/runs/qwen3-topk16-sample-step1-comparator-fix1-20261012-r1/`。基线 HEAD `8be53a565641e2c6ce4d14cb54fef7ea336b5c94`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；开始该格前 25 秒有界 upstream fetch 超时，最近可信 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 不称为实时同步结果。

### 推理 cache 开关的公开 CLI 约束（2026-10-12）

新预检键 `qwen3-public-infer-batch2-nocache-greedy4-world1-20261012-r1` 的 srun job 2677 只检查 CLI、未启动模型；worker 显示此版本公开 `swift infer` 不暴露 `--use_cache`，故没有运行独立的 cache-on/off 对照。已核验的 tiny fixture `generation_config.json` 将 `use_cache` 设为 `false`，但这不替代成对 cache 配置验收。该尝试是公开控制参数未暴露的前置限制，不是 ACL/模型兼容失败；保留运行键且不重跑。

后续以本地只读模型副本（只修改 `config.json` 与 `generation_config.json` 的 `use_cache=true`，其余已校验权重和 tokenizer 使用符号链接）补齐 cache-enabled 公开入口格。初始键 `qwen3-public-infer-batch2-cacheon-greedy4-world1-20261012-r1` 的 worker 预检 job 2691 通过，但随后发现尚未执行模型前的 gate 草稿仍按两个 batch 请求计数；没有 native/candidate 模型作业在该键运行。新键 `...cacheon...-r2` 的预检 job 2692 通过：实际模型目录 `cache-on-base`、adapter `checkpoint-3`，两条原始提示各 115 token，权重与 adapter 清单/hash 可读、离线不下载；native 身份为 torch_npu/Ascend910B3，candidate 为 ACL，核心 SHA256 `7c2d5edd…3294bc`、`backend_fallback=error` 且预检 fallback=0。Native public infer job 2693 退出 0；gate job 2694 暴露 gate 自身把单个 batch-2 请求误数成 8 次前向的 harness 断言错误，不启动 ACL candidate。

修正计数后的 gate-only 键 `qwen3-public-infer-batch2-cacheon-greedy4-gatefix1-20261012-r1` 以只读链接复用 jobs 2692/2693 的产物，worker gate job 2695 通过：有效 `GenerationConfig.use_cache=true`，实际 batch=2，4 次模型输入长度为 `[124,1,1,1]`，确认 prefill 后为三步单 token decode。候选 job 2696 在模型前因复制的 observer 输出目录已被 setup 预创建、而 observer 使用 `mkdir(exist_ok=False)` 退出；这是 harness 目录冲突，不是 ACL 兼容结论。新键 `qwen3-public-infer-batch2-cacheon-greedy4-aclfix1-20261012-r1` 删除该预创建目录并只读复用原生/预检/gate 证据；strict ACL public infer job 2697 与 worker comparator 2698 均通过。Comparator 检查了两侧完整有效 GenerationConfig 相同且 `use_cache=true`、加载参数和所有输入精确相同、4 组 logits 满足 `atol=1e-5, rtol=1e-4`（最大绝对差 0）、greedy token ID 与响应精确相同；candidate 全程 NPU、ACL `backend_fallback=error`、fallback=0。未下载权重。

因此该锁定 cache-enabled tiny Qwen3 单 NPU、batch-2、公开 `swift infer` case 的 L0/L1/L4 通过；通过模型本地配置启用缓存，不代表公开 CLI 提供 `--use_cache` 开关，也不构成配对 on/off 数值/性能比较或 L5 结论。历史 job 2677 的 CLI 探针和两次 harness 失败均保留，不追溯改写。原始清单、脚本、worker 日志与数组快照未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-greedy4-world1-20261012-r1/`、`...-r2/`、`...-gatefix1-20261012-r1/`、`...-aclfix1-20261012-r1/`。


### Qwen3 cache-enabled top-p 多步 sampling（2026-10-12）

新锁定键 `qwen3-public-infer-batch2-cacheon-topp08-sample-multistep4-world1-20261012-r2` 完成真实 public `swift infer` native/strict ACL 对照。旧 `...-r1` 仅有未提交 builder 的语法错误，无 Slurm 作业，证据保留且不复用。r2 worker 预检 job 2700 通过：模型目录实际名 `cache-on-base`、adapter 为 `checkpoint-3`，weight/tokenizer 与 adapter manifests 可读，离线模式且不下载；model 与 generation config 的 `use_cache=true`，原生身份为 torch_npu/Ascend910B3，candidate ACL core SHA256 `7c2d5edd…3294bc`、`backend_fallback=error`、fallback=0。Native public inference job 2701 和只读 worker gate 2702 通过；4 条固定 prompt 分两个 batch=2 请求，每条生成 4 token，gate 观测两组 batch、共 8 次前向，每个请求序列长度为 `[prefill,1,1,1]`。Strict ACL candidate job 2703 与只读 worker comparator job 2704 通过，未下载模型。比较器确认初始参数及 prefill IDs 精确，prefill logits 最大绝对差 0，六组匹配 decode logits 在预置 `atol=1e-5, rtol=1e-4` 内；两侧各自采样 token 均在本侧 top-p=0.8 nucleus 且被下一步消费。观测 token 序列恰好相同，但验收不要求也不据此宣称跨运行时 RNG 等价。

该锁定 cache-enabled tiny Qwen3 单 NPU、batch-2、top-p=0.8 四步公开推理配置 L0/L1/L4 通过，L5 未运行。缓存由只修改 `config.json` 与 `generation_config.json` 的本地模型副本启用；不代表 CLI 暴露 `--use_cache`，不构成 on/off 配对、性能结论或其它 sampling/cache 配置的泛化。之前 no-cache CLI 开关缺失和 cache-on greedy 的 harness 失败详见上一节，历史结论不变。

### Qwen3 tiny 公开 top-k=4 与 top-p=0.8 组合 sampling（2026-10-12）

新增锁定键 `qwen3-public-infer-batch2-topk4-topp08-sample-step1-world1-20261012-r1`，固定四条 prompt、两个 batch=2 请求，`temperature=1`、`top_k=4` 与 `top_p=0.8` 同时启用，每条生成一个 token。预检 job 2685、原生 torch_npu oracle 2686、worker 原生门禁 2687、strict ACL candidate 2688、worker 比较器 2689 均成功。实际模型/Adapter 目录 `base` / `checkpoint-3`、清单与文件 hash 均通过，离线未下载；ACL core SHA256 `7c2d5edd…3294bc`，`backend_fallback=error`、fallback=0。

比较器按 top-k 先筛选再对保留 logits 计算 top-p nucleus，逐侧验证采样 token 落在该组合候选集合。两侧参数、prefill IDs 精确相同，prefill logits 最大绝对差 0；本次 sample token 序列碰巧相同，不作为 RNG 一致性判据。因此该锁定组合公开推理格 L0/L1/L4 通过，不推广到多步组合采样、其它阈值、cache-on/off 对照或 L5。原始证据未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-topk4-topp08-sample-step1-world1-20261012-r1/`。基线 HEAD `4a09308c2defc4c5e839df98088589c552bdf50c`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；本格开始时同步的 upstream `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是当前 HEAD 的祖先。

### Qwen3 tiny cache-enabled top-k=2 四步 sampling（2026-10-12）

本格补齐 cache-enabled 多步 sampling 的另一种筛选配置。模型运行键 `qwen3-public-infer-batch2-cacheon-topk2-sample-multistep4-world1-20261012-r2` 按 worker 预检 job 2717、原生 torch_npu oracle 2718、worker 原生门禁 2719、strict ACL candidate 2720 的顺序执行；只读比较器专属新键 `qwen3-topk2-sample-multistep4-comparator-fix1-20261012-r1` 的 worker job 2722 通过。原 comparator job 2721 错保留 top-k=4/top-p=0.8 的模板配置断言，在比较前失败；修正键只读复用了既有产物，没有重跑模型。

Worker 预检确认模型实际目录名 `cache-on-base-topk4`、Adapter `checkpoint-3`，模型/adapter 权重、输入与初态 manifest 可读，下载关闭；cache 在两侧有效 generation config 中均为 true。四条 prompt 实际分为两次 batch=2 public infer 请求，各生成最多四步，原生门禁观测解码序列 `[prefill,1,1,1]`。Comparator 确认初始加载参数和 prefill 输入精确相同，prefill logits 最大绝对差为 0；六个匹配 decode 步 logits 满足 `atol=1e-5, rtol=1e-4`。每侧每一步采样 token 均属于该侧 top-2 集合并由下一步消费。Candidate 使用严格 ACL `backend_fallback=error`，fallback=0。两侧本次 token 序列虽相同，仍不宣称跨 runtime RNG 等价；L5 未运行。

前置运行键 `qwen3-public-infer-batch2-cacheon-topk2-cacheon-sample-multistep4-world1-20261012-r1` 的 job 2716 因复制的 sbatch 保留旧 `R` 路径，只运行旧 top-k=4 预检，没有执行模型，不作为 top-k=2 证据；其 harness 错误及 job 2721 均保留。原始日志、脚本与快照未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk2-sample-multistep4-world1-20261012-r2/` 和 `$TASK_STATE/runs/qwen3-topk2-sample-multistep4-comparator-fix1-20261012-r1/`。本格基线 HEAD `f46b3bd6813648495486b23252b52837bfb0f930`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；`origin/2.0-refactor` 同步 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 为该 HEAD 的祖先。

### Qwen3 tiny cache-enabled top-k=16 四步 sampling（2026-10-12）

新锁定键 `qwen3-public-infer-batch2-cacheon-topk16-sample-multistep4-world1-20261012-r1` 将 seeded top-k sampling 扩展到 cache-enabled 四步公开 `swift infer`。worker 预检 job 2724、native torch_npu oracle job 2725、只读 gate 修正 job 2727、strict ACL candidate job 2728、worker comparator job 2729 全部退出 0。原始只读门禁 job 2726 在检查数据前因该新键的 manifest 未填 `jobs.native_oracle` 而抛出 `KeyError`；保留为 gate harness 失败。修正键 2727 只读取 2724/2725 既有产物，没有重跑预检或 native inference；候选依赖其通过后才启动。

预检验证模型实际目录 `cache-on-base-topk4`、Adapter `checkpoint-3`，input/initial-state manifests 与模型清单可读、离线未下载；两侧有效 GenerationConfig 均 `use_cache=true`。四条 prompt 被分为两个真实 batch=2 请求，最多各生成 4 token；原生 gate 记录 8 次 batch=2 forward，两个请求长度均为 `[prefill,1,1,1]`。Comparator 检查初始参数和 prefill IDs 精确一致、prefill logits 最大绝对差 0；六组匹配 decode logits 在 `atol=1e-5, rtol=1e-4` 内。两侧每个 token 均属于各自 top-16 候选集并由下一步消费；Candidate ACL `backend_fallback=error` 且 fallback=0。观测到 token 序列相同，但不据此声称跨 runtime RNG 等价。此锁定配置的 L0/L1/L4 通过，L5 未运行；不推广到其它 top-k/cache/模型或 RNG 语义。

原始运行证据未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk16-sample-multistep4-world1-20261012-r1/`；gate 修正及只读对照位于 `$TASK_STATE/runs/qwen3-topk16-native-gatefix1-20261012-r1/` 与 `$TASK_STATE/runs/qwen3-topk16-sample-multistep4-comparator-fix1-20261012-r1/`。基线 HEAD `38131e2e24af130fcc596ffacc00343982b1b947`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；开始本格前由 mirror 获取的 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该基线祖先。

### Qwen3 tiny cache-enabled top-k=8 四步 sampling（2026-10-12）

新锁定键 `qwen3-public-infer-batch2-cacheon-topk8-sample-multistep4-world1-20261012-r1` 完成 public `swift infer` 原生 torch_npu 与 strict ACL 对照。预检 job 2731、原生 oracle job 2732、只读原生 gate job 2733、strict ACL candidate job 2734、worker 比较器 job 2735 均退出 0。实际模型/Adapter 目录为 `cache-on-base-topk4` / `checkpoint-3`；输入与初态 manifest 可读且摘要锁定，权重摘要验证通过，离线未下载；有效 `GenerationConfig.use_cache=true`。两侧 worker 分别确认原生 torch_npu/Ascend910B3 与 Jittor ACL、`backend_fallback=error`、fallback=0，candidate core SHA256 `7c2d5edd…3294bc`。

四条固定 prompt 以两个真实 batch=2 请求运行，每条最多生成四 token。原生门禁记录 8 次 NPU 前向，序列长度分别为 `[15,1,1,1]` 和 `[17,1,1,1]`，覆盖 prefill 及单 token cache decode。比较器确认加载参数和 prefill 输入完全相同，prefill logits 最大绝对差为 0；六个匹配 decode 步 logits 满足 `atol=1e-5, rtol=1e-4`。两侧每一步采样 token 均属于各自 top-k=8 集合，并由后续 decode 输入消费。观察到两侧 token 序列相同，但不将偶然一致解释为跨 runtime RNG 等价。

此锁定 cache-enabled tiny Qwen3、单 NPU、batch-2、top-k=8 四步公开推理配置 L0/L1/L4 通过，L5 未运行；不覆盖 cache-off、beam search、其他 top-k/模型配置或训练/恢复。原始日志、脚本、清单和快照未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk8-sample-multistep4-world1-20261012-r1/`。基线 HEAD `436627f655d5093ab6d09d19002a16f1d4ee8c6b`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；本格开始前从 mirror 同步的 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。

### Qwen3 tiny cache-enabled top-k=4 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk4-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 的 torch_npu 与 strict ACL 对照。worker 预检 job 2738、原生 oracle 2739、只读原生门禁 2740、候选 2741 与只读 worker 比较器 2742 全部退出 0。实际模型和 Adapter 目录为 `cache-on-base-topk4` 与 `checkpoint-3`；模型、Adapter、输入及初态清单摘要可读，下载关闭，`GenerationConfig.use_cache=true`。候选复用已校验 core SHA256 `7c2d5edd…3294bc`，运行于 ACL 且 `backend_fallback=error`、fallback=0。

固定四条 prompt 分两个实际 batch=2 请求，每条最多生成四 token。原生 gate 观察到八次 NPU 前向，两个请求输入长度分别为 `[15,1,1,1]` 和 `[17,1,1,1]`。比较器确认两侧加载参数和 prefill IDs 精确，prefill logits 最大差为 0；六个匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`。每侧采样 token 都落在各自 top-4 候选中并由后续输入消费。两侧本次生成序列相同，但不据此声称跨 runtime RNG 等价。故该锁定配置 L0/L1/L4 通过，L5 未运行；不外推到其他采样配置、cache-off、beam search、训练或其他模型。原始日志、manifest、脚本和快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk4-sample-multistep4-world1-20261012-r1/`。
本格基线 HEAD `2fafb1d0f9a73c954cdbe21a43a2ff1ebe78bf1f`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。开始本次结果集成前从 mirror 获取的目标分支 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是该基线祖先。


### Qwen3 tiny cache-enabled top-k=4、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk4-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 的 torch_npu 与 strict ACL 对照。预检 job 2746、原生 oracle 2747、只读原生门禁 2748、候选 2749、只读 worker 比较器 2750 全部退出 0。模型/Adapter 实际目录为 `cache-on-base-topk4` / `checkpoint-3`，输入与初态 manifest 摘要匹配，下载关闭，模型及 generation config 均启用 KV cache。native 为 Ascend910B3 `torch_npu`；候选为 ACL，`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。

四条 prompt 分两个真实 batch=2 请求，每条最多生成四 token。原生 gate 观察到 8 次 NPU 前向，两组请求的输入序列长度分别为 `[15,1,1,1]`、`[17,1,1,1]`。比较器确认加载参数与 prefill 输入精确一致，prefill logits 最大绝对差为 0；六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`。每侧每步采样均落入各自 top-k=4 候选集，后续 decode 输入消费前一步 token。两侧本次 token 序列恰好相同，但不据此声称跨 runtime RNG 等价。该锁定配置 L0/L1/L4 通过，L5 未运行；不外推到其他温度、采样策略、beam search、训练/恢复或其他模型。

原始日志、manifest、脚本和快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk4-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `5cfcf691d251327d7bf3dd0a0c1e7f33255511c1`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。该格开始前从 mirror 获取的 `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。


### Qwen3 tiny cache-enabled top-k=8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk8-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 的 torch_npu 与 strict ACL 对照。预检 job 2754、原生 oracle 2755、只读原生门禁 2756、候选 2757、只读 worker 比较器 2758 全部退出 0。模型/Adapter 实际目录为 `cache-on-base-topk4` / `checkpoint-3`，模型权重、Adapter、输入与初态 manifest 均与预检摘要匹配，离线未下载；模型和 generation config 都启用 KV cache。native 运行于 Ascend910B3 `torch_npu`；candidate 使用 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。

四条 prompt 分两个真实 batch=2 请求，每条最多生成四 token。原生 gate 记录 8 次 NPU 前向，解码输入长度分别为 `[15,1,1,1]`、`[17,1,1,1]`。比较器确认参数和 prefill IDs 精确一致，prefill logits 最大绝对差为 0，六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；每侧采样 token 都属于自身 top-k=8 集合，后续输入消费前一步 token。观察到两侧序列相同，不据此声称跨 runtime RNG 等价。此锁定配置 L0/L1/L4 通过，L5 未运行；不外推到其他温度/采样配置、beam、训练/恢复或其他模型。

原始日志、manifest、脚本和快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk8-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `2958af2cd24f0f3deae0648508bbcc860e335fc1`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。该新格开始前同步得到的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。


### Qwen3 tiny cache-enabled top-k=2、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk2-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。worker 预检 job 2762、原生 oracle 2763、原始只读 gate 2764、修复 gate 新键 job 2765、candidate 2766、修复后的只读比较器新键 job 2767。原 gate 2764 因复制模板遗留 `top_k==8` 与本格 top-k=2 不符而在检查数据前失败；保留为 harness 失败。gatefix 2765 与 comparator-fix 2767 复用成功的 native/candidate 产物，没有重跑模型。

实际模型/Adapter 目录为 `cache-on-base-topk4` / `checkpoint-3`；权重、输入和初态 manifest 摘要与预检相符，离线不下载。有效配置 cache=true；native 为 Ascend910B3 `torch_npu`，candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。四条 prompt 分两个实际 batch=2 请求，每条最多生成 4 token。native gate 观察到八次 NPU 前向，序列长度为 `[15,1,1,1]`、`[17,1,1,1]`。比较器确认加载参数/pre-fill IDs 精确相同，prefill logits 最大绝对差为 0；六个匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`。每侧每个 token 均位于自身 top-k=2 集合，且由下一步输入消费。两侧本次 token 序列相同，不据此声称 RNG 等价。该锁定配置 L0/L1/L4 通过，L5 未运行；不外推其他温度、采样策略、beam、模型或训练/恢复。

原始日志、清单、脚本和快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk2-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `1c566269a669c6d99f4ec4fc824766e623d0d8d8`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步到的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。


### Qwen3 tiny cache-enabled top-k=16、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk16-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。预检 2771、原生 oracle 2772、原始原生 gate 2773、修复 gate 新键 2774、候选 2775、原始 comparator 2776、comparator 修复尝试 2777、最终只读比较器新键 2778。2773 因模板保留 `top_k==8` 断言、2776 因比较器的 stale `inputs['top_k']==8`、2777 因一次未生效的字符串替换而分别在数据判断前失败，均保留为 harness 失败。修复 gate 2774 和 comparator fix3 2778 均只读复用成功的 native/candidate 产物，没有重跑模型。

模型/Adapter 真实目录为 `cache-on-base-topk4` / `checkpoint-3`，清单摘要一致且权重下载关闭；两侧 cache=true。native 为 Ascend910B3 `torch_npu`；candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。四条 prompt 分为两个真实 batch=2 请求，每条最多生成四 token。修复 gate 观察到八次 NPU 前向，两组输入长度 `[15,1,1,1]`、`[17,1,1,1]`。Comparator 确认参数与 prefill IDs 精确、prefill logits 最大绝对差为 0，六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；每侧采样 token 属于自身 top-k=16 集合并由下一步输入消费。两侧 token 恰好相同，不声称 RNG 等价。该锁定配置 L0/L1/L4 通过，L5 未运行，不外推至其他温度、采样方式、beam、训练/恢复或其他模型。

原始日志、清单、脚本和快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk16-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `5725268ae1defd430b3c12ed39336cc2df34de5b`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前从 mirror 同步的目标 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。


### Qwen3 tiny cache-enabled top-k=4 + top-p=0.8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk4-topp08-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。预检 2782、原生 oracle 2783、原生 gate 2784、candidate 2785、只读比较器 2786 全部退出 0。模型/Adapter 实际目录 `cache-on-base-topk4` / `checkpoint-3`；模型和 generation config 均 cache-on，清单校验通过，权重下载关闭。native 为 Ascend910B3 `torch_npu`；candidate 使用 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。

四条固定 prompt 分为两个真实 batch=2 请求，每条最多生成四 token。native gate 观察到八次 NPU 前向，序列长度为 `[15,1,1,1]` 和 `[17,1,1,1]`。比较器确认初始参数与 prefill IDs 精确相同，prefill logits 最大绝对差为 0，六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`。各 runtime 每步 token 都符合其自身 top-k=4 后 top-p=0.8 的组合候选集合，并由后续 cache decode 输入消费。两侧本次 token 序列相同，不据此声称 RNG 等价。该锁定配置 L0/L1/L4 通过，L5 未运行；不外推到其他温度/采样配置、cache-off、beam、训练/恢复或其他模型。

原始日志、清单、脚本与快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk4-topp08-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `ed6b602896cb108045106a6246dd5747932d599b`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前从 mirror 同步的目标 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。

### Qwen3 tiny cache-enabled top-k=2 + top-p=0.8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk2-topp08-temp07-sample-multistep4-world1-20261012-r2` 完成公开 `swift infer` 的原生 torch_npu 与 strict ACL 对照。r1 的短预检 2790 通过，但原生 job 2791 因 run setup 预建了插件自身要求创建的 `observations` 目录而 `FileExistsError`；这是 harness/setup 错误，不是 ACL 结论。保留 r1 及部分产物，不复跑该键。r2 预检 2792、原生 oracle 2793、原生 gate 2794、strict ACL candidate 2795 均退出 0。2796 的数值比较断言通过，但比较 JSON 把真实模型目录误报为 `cache-on-base-topk2`；2797 因 checker 目录缺原生 gate 状态链接、2798 因 checker-only manifest 不完整，分别在读取数据前失败。均为只读 harness 错误；最终 checker-fix3 job 2799 复用原生/candidate 产物并通过，输出正确目录名，没有重跑推理。

实际模型/Adapter 目录为 `cache-on-base-topk4` / `checkpoint-3`；权重、输入与初态清单摘要一致，离线未下载。两侧启用 KV cache；native 为 Ascend910B3 `torch_npu`，candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。四条固定 prompt 分两个实际 batch=2 请求，每条最多生成四 token。native gate 观察到八次 NPU 前向，序列长度分别为 `[15,1,1,1]`、`[17,1,1,1]`。fix3 比较器确认加载参数与 prefill IDs 完全一致、prefill logits 最大绝对差为 0，六组相同 decode 输入的 logits 满足 `atol=1e-5, rtol=1e-4`。每侧每步样本都属于各自 top-k=2 后 top-p=.8 的组合候选集合，后续 cache 输入消费上一 token。两侧这次序列相同，不声称 RNG 等价。此锁定配置 L0/L1/L4 通过，L5 未运行；不外推至其它采样参数、cache-off、beam、训练/恢复或其它模型。

日志、manifest、脚本及快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk2-topp08-temp07-sample-multistep4-world1-20261012-r2/`，只读比较器修复证据位于 `$TASK_STATE/runs/qwen3-topk2-topp08-temp07-comparator-fix3-20261012-r1/`。基线 HEAD `c547d29cecedd5a1bed7f630cd880a89bd2eb034`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该基线祖先。

### Qwen3 tiny cache-enabled top-k=8 + top-p=0.8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk8-topp08-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 的原生 torch_npu 与 strict ACL 对照。预检 2803、原生 oracle 2804、原生 gate 2805、candidate 2806、只读 worker 比较器 2807 均退出 0。实际模型/Adapter 目录 `cache-on-base-topk4` / `checkpoint-3`；权重、输入和初态清单摘要匹配，下载关闭。native 为 Ascend910B3 `torch_npu`；candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`；两侧 cache-on。

四条固定 prompt 分两个真实 batch=2 请求，每条最多生成四 token。native gate 观察到八次 NPU 前向，解码长度分别为 `[15,1,1,1]` 与 `[17,1,1,1]`。比较器确认加载参数及 prefill IDs 精确一致，prefill logits 最大绝对差为 0；六组相同 decode 输入的 logits 满足 `atol=1e-5, rtol=1e-4`。两侧每步 token 均在各自 top-k=8 后 top-p=.8 的组合候选集内，下一步 cache 输入消费上一 token。观察到的采样序列相同不代表跨 runtime RNG 等价。此锁定配置 L0/L1/L4 通过，L5 未运行；不外推到其它采样参数、cache-off、beam、训练/恢复或其它模型。

原始日志、manifest、脚本与快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk8-topp08-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `8ceffd78a835876503816c351528461f40a9d600`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该基线祖先。

### Qwen3 tiny cache-enabled top-k=16 + top-p=0.8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk16-topp08-temp07-sample-multistep4-world1-20261012-r2` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。r1 的预检脚本残留 `top_k==8` 检查，与 top-k16 CLI/清单冲突，因此 job 2811 在 worker 检查前取消；未执行模型。修订为新键 r2 后，worker 预检 2812、native oracle 2813、只读 native gate 2814、strict ACL candidate 2815、只读比较器 2816 均退出 0。

模型/Adapter 实际目录为 `cache-on-base-topk4` / `checkpoint-3`；本地权重、Adapter、输入与初态摘要可读且匹配，下载关闭。两侧启用 KV cache。native 为 Ascend910B3 `torch_npu`；candidate 使用 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。四条固定 prompt 分两个真实 batch=2 请求，每条最多生成四 token。native gate 观察到八次 NPU 前向，输入长度为 `[15,1,1,1]` 与 `[17,1,1,1]`。比较器确认加载参数、prefill IDs 精确相同，prefill logits 最大绝对差为 0；六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`。每侧 token 均属于自身 top-k=16 后 top-p=.8 的组合候选集，并由下一步 cache 输入消费。两侧本次 token 恰好相同，不据此声称 RNG 等价。此锁定配置 L0/L1/L4 通过，L5 未运行；不外推其他采样配置、cache-off、beam、训练/恢复或其他模型。

原始日志、清单、脚本与快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk16-topp08-temp07-sample-multistep4-world1-20261012-r2/`。基线 HEAD `0d6d1c095`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。

### Qwen3 tiny cache-enabled top-k=32 + top-p=0.8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk32-topp08-temp07-sample-multistep4-world1-20261012-r3` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。两次未进入 worker 的 harness 尝试已保留：job 2820 模板仍断言 top-k=16 且结果 metadata 错写模型目录名，job 2821 的 staging 复制遗漏 JSON 清单；修正后 r3 的预检 2822、native oracle 2823、native gate 2824、strict ACL candidate 2825、只读比较器 2826 均退出 0。

模型/Adapter 实际目录为 `cache-on-base-topk4` / `checkpoint-3`；权重、Adapter、输入与初态清单摘要匹配且下载关闭。两侧 KV cache 启用。native 为 Ascend910B3 `torch_npu`；candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`。四条固定 prompt 分为两个实际 batch=2 请求，每条最多生成四 token。native gate 观察到八次 NPU 前向，序列长度为 `[15,1,1,1]`、`[17,1,1,1]`。比较器确认加载参数和 prefill IDs 精确一致，prefill logits 最大绝对差为 0，六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；每侧采样 token 均属于自身 top-k=32 后 top-p=.8 组合候选集，后续输入消费上一 token。本次两侧序列相同，不声称 RNG 等价。此锁定配置 L0/L1/L4 通过，L5 未运行；不外推至其他采样配置、cache-off、beam、训练/恢复或其他模型。

原始日志、manifest、脚本与快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk32-topp08-temp07-sample-multistep4-world1-20261012-r3/`。基线 HEAD `4879c84be`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。

### Qwen3 tiny cache-enabled top-k=64 + top-p=0.8、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk64-topp08-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。worker 预检 2830、native oracle 2831、只读 native gate 2832、strict ACL candidate 2833、只读比较器 2834 均退出 0。预检确认模型/Adapter 实际目录 `cache-on-base-topk4` / `checkpoint-3`，权重与输入/初态清单可读且匹配、离线没有下载。native 是 Ascend910B3 `torch_npu`；candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`；两侧均启用 KV cache。

四条固定 prompt 分两个真实 batch=2 请求，每条最多生成四 token。native gate 观察到 8 次 NPU 前向，解码长度分别为 `[15,1,1,1]`、`[17,1,1,1]`。比较器确认初始参数与 prefill IDs 精确，prefill logits 最大绝对差为 0，六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；两侧每步 sample 都符合自身 top-k=64 后 top-p=.8 的组合候选集，后续 cache 输入消费上一 token。本次两侧序列相同，不据此声称 RNG 等价。该锁定配置 L0/L1/L4 通过，L5 未运行；不外推其他采样配置、cache-off、beam、训练/恢复或其他模型。

原始日志、manifest、脚本与快照位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk64-topp08-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `6aede08bf`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是基线祖先。

### Qwen3 tiny cache-enabled top-k=64 + top-p=0.9、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk64-topp09-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。worker 预检 2838、native oracle 2839、只读 native gate 2840、strict ACL candidate 2841、只读比较器 2842 均退出 0。预检核实模型/Adapter 实际目录 `cache-on-base-topk4` / `checkpoint-3`、配置与清单一致、无下载；native 为 Ascend910B3 `torch_npu`，candidate 为 ACL、`backend_fallback=error`、fallback=0，core SHA256 `7c2d5edd…3294bc`，两侧均启用 KV cache。

四条固定 prompt 分为两个真实 batch=2 请求，每条最多生成四个 token。native gate 检查到八次 NPU 前向，两请求解码长度为 `[15,1,1,1]` 与 `[17,1,1,1]`。比较器确认加载参数与 prefill IDs 精确一致，prefill logits 最大绝对差为 0，六组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；采样 token 按比较器中的 top-k64 后 top-p=.9 候选集合规则检查，下一步 cache 输入消费前一 token。两侧本次 token 序列恰好相同，不据此声明 RNG 等价。该锁定配置 L0/L1/L4 通过，L5 未运行；不外推到其他采样设置、cache-off、beam、训练/恢复或其他模型。

原始日志、manifest、脚本与快照未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk64-topp09-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `14804c5f3`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。本格开始前同步的 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该基线祖先。


### Qwen3 tiny cache-enabled top-k=32 + top-p=0.9、temperature=0.7 四步 sampling（2026-10-12）

新独立格 `qwen3-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261012-r1` 完成公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。所有检查器均在标记的 Slurm `srun` worker 执行：短预检 job 2882、native oracle 2883、只读 native gate 2884、strict ACL candidate 2885、只读比较器 2886 均通过。预检确认真实目录名 `cache-on-base-topk4` 与 `checkpoint-3`、有效 cache 配置、模型/adapter/input/initial-state 清单可读且哈希匹配；复用本地已校验文件，无下载。Native 为 Ascend910B3/torch_npu；candidate 为 Jittor 2.0.0/ACL，core SHA256 `7c2d5edd…3294bc`，`backend_fallback=error` 且完整运行 fallback=0。

四条固定 prompt 分为两次真实 batch=2 请求，每条最多生成四 token。原生 gate 观察到 8 次 NPU 前向，两个请求的序列分别为 `[15,1,1,1]`、`[17,1,1,1]`；有效 GenerationConfig 为 cache=true、sampling、top-k=32/top-p=.9、temperature=.7。比较器确认加载参数和 prefill IDs 精确一致，prefill logits 最大差为 0，六个匹配 decode 步 logits 满足 `atol=1e-5, rtol=1e-4`；两侧采样 token 各自符合 top-k 后 top-p 的截断集合，且 cache 输入消费前一步 token。两侧本次 sampled sequence 相同不作为 RNG 等价证据。该锁定配置 L0/L1/L4 通过，L5 未运行；不外推到 cache-off、beam、训练/恢复、其他采样设置或其他模型。

原始日志、哈希清单、运行 manifest、checker 与快照未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `15e2099b9e0e964dba1b5c18a08b1a6b1f169806`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；集成前同步 mirror `2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该基线祖先。

### Mistral cache-enabled top-k=64 + top-p=0.8 四步公开 sampling（2026-10-12）

锁定配置 `mistral-public-infer-batch2-cacheon-topk64-topp08-temp07-sample-multistep4-world1-20261012-r1` 完成单 NPU 公开 `swift infer` 原生 torch_npu 与 strict ACL 对照。所有检查均由 Slurm worker 的 `srun` 执行：预检 job 2872、native oracle 2873、native gate 2874、ACL candidate 2875、只读比较器 2876 均成功。预检复核实际模型目录 `cache-on-base`、Adapter 目录 `checkpoint-3`，4 条原始提示 tokenizer 长度为 20，模型与输入/初态清单可读且下载关闭；native torch_npu/Ascend910B3 与 candidate Jittor/ACL 身份、core SHA 和 strict fallback 策略通过。

原生门禁实际观察到两个 batch=2 请求，各有 4 次前向且输入序列长度 `[29,1,1,1]`；有效 GenerationConfig 为 cache=true、top-k64、top-p=.8、temperature=.7。候选 public infer 返回 4 行结果；比较器确认两侧初始参数和 prefill IDs 精确相同，prefill logits 最大差 `8.940696716308594e-08`，8 组匹配 decode logits 均满足 `atol=1e-5, rtol=1e-4`；各 runtime 的样本均属于本侧 top-k 后 top-p 候选集，下一步 cache 输入消费前一 token。两侧观测到的 token 序列相同，不作为 RNG 等价证据。Candidate `backend_fallback=error`、完整观察计数 0；adapter 叶子计算输入/输出在 `npu:0`。native 观察到 8 个 adapter 参数位于 CPU，而 candidate 参数执行设备有别；这一差异如实保留，未测性能。故本锁定推理格 L0/L1/L4 通过、L5 未运行；不推广至 cache-off、其他采样参数、训练/恢复或其他模型。原始日志、输入与快照未版本化，位于 `$TASK_STATE/runs/mistral-public-infer-batch2-cacheon-topk64-topp08-temp07-sample-multistep4-world1-20261012-r1/`。基线 HEAD `2b607f5ac31e079de8af465f4164c6a9a681c5a0`；本格开始前同步的 `origin/2.0-refactor` SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0` 是该基线祖先。

### Swift LoRA 双卡公开 SFT 短程路径复验（2026-10-13）

在原生公开 oracle 键 `040da30a6c84e9a5f7adfcc5fa0b937dbd14ffc21224ef8c57a9dc67955fff98` 与 strict ACL 候选键 `976b0318eef6d3e920c02ecd36b523941b90d076af3d07b922126cd165c8f06f`（Slurm job 2981）的既有产物上复核短程公开 `swift sft` 路径；没有重跑 oracle 或候选。两者都是单机 HCCL/world-size=2、FP32、同一锁定 fixture SHA `c5d16d3215c84b6c2047fd3b9be1ed0246420eebfc5b45c01197ee6e6ffc47b7`、同一数据/初始 adapter 清单和相同三步训练参数。native torch_npu 使用可见设备 2,3；candidate 使用 ACL 可见设备 4,5、`backend_fallback=error`，两个 rank 的 fallback count=0。job 2979 的 srun-worker 预检核对了固定文件 SHA、native checkpoint 与恢复证据、公开 CLI 解析路径和命令；job 2981 随后经串行 cache 预热、两个 launcher/rank singleton、真实 HCCL world-size=2 微检后运行公开 `swift sft`，两 rank 均到 global_step=3，退出码 0，并生成 `checkpoint-2` 与 `checkpoint-3`。

三步 loss 在 native 与 candidate 中分别完全相同：`4.82627869, 4.84576845, 4.85396099`，末段没有下降。按新合同，三步烟测不证明收敛，且 native oracle 也未通过此训练窗口；因此不能把此结果判为 ACL 特有 L2 回归，L2 保持未通过/待新窗口验收。此次只证明候选公开训练命令实际启动、完成并写出 checkpoint；没有做候选新进程恢复或推理对照，L3/L4 仍不授予通过。旧矩阵行和历史严格对拍失败原样保留。原始产物及 worker 证据未版本化，位于 `$TASK_STATE/runs/swift-trainer-world2-acl-candidate-20261013-r2/` 和 `$TASK_STATE/delivery/runs/040da30a6c84e9a5f7adfcc5fa0b937dbd14ffc21224ef8c57a9dc67955fff98/`、`$TASK_STATE/delivery/runs/976b0318eef6d3e920c02ecd36b523941b90d076af3d07b922126cd165c8f06f/`；candidate 基线 HEAD `6cfac80ad26ff1112713f9e8d4fa84c164cea660`，同步目标 SHA `4896af611650d1436302c06b0653ac61726fd10c` 是该基线祖先。后续需要新运行键锁定能下降的训练窗口，先跑 native oracle，通过后才运行 strict ACL candidate 和恢复验收。


### Qwen3 cache-enabled greedy 16-token decoding（2026-10-13）

从未完成矩阵选择新的单 NPU 推理格：公开 `swift infer`、Qwen3 tiny、FP32/SDPA、batch=2、cache-enabled greedy 16-token。新键 `qwen3-public-infer-batch2-cacheon-greedy16-world1-20261013-r2` 的 worker 预检 job 2984、torch_npu oracle job 2985、只读 native gate job 2986、严格 ACL candidate job 2987、worker comparator job 2988 全部退出 0；预检与所有 gate/comparator 均在标记的 Slurm `srun` worker 内执行。模型目录实际名 `cache-on-base`，Adapter 目录 `checkpoint-3`；锁定输入两行，原始 token 长度 `[2,6]`，模型权重、tokenizer、Adapter、输入和初态清单在 worker 可读，离线且未下载权重。模型参数和 tokenizer 文件复用本地已校验 Qwen fixture，cache-on 只改模型与 generation 配置副本。

Native gate 观察到公开 `GenerationConfig.use_cache=true`、实际 batch 2、16 次 NPU forward；输入序列长度为 prefill 15 后连续 15 个单 token cache decode。比较器检查两侧加载参数、输入、token 与响应精确相同，16 组 logits 满足锁定容差且 max abs 差为 0。Candidate runtime/provider 为 Jittor shim/ACL，使用既有持久 JITTOR_HOME 与 `backend_fallback=error`，完整进程结束 fallback 计数为 0；candidate 所有记录的模型输入、logits 和参数驻留 `npu:0`。本锁定推理格 L0/L1/L4 通过，L5 未运行；不外推到其他 prompt、模型规模、生成设置或 cache on/off 性能比较。

前置问题按原样留存：初始新键 `qwen3-public-infer-batch2-cacheon-greedy16-world1-20261013-r1` 的预检 job 2982 识别到当前 JITTOR_HOME 缺少本构建配置的 `jittor_core`，`JITTOR_NO_BUILD=1` 按预期阻止隐式编译；模型阶段未运行。独立 cache-prewarm key `qwen3-cacheon-greedy16-cachebuild-20261013-r1` 的 job 2983 在 worker 显式 bootstrap/build 并 no-build 验证通过，core SHA256 `24b99842d51e481b79c45163c95fd8fcc43c1ad1955cf9127c8c3f8fd9f379ad`；r2 预检确认相同 SHA 后才继续 oracle-first。r1 与 2982 作为 harness/cache 前置失败保留，不记兼容失败，也未在旧键上重跑。

原始 manifest、运行脚本、worker 日志、NPZ 与模型文件清单未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-greedy16-world1-20261013-r1/`、`...-r2/` 与 `$TASK_STATE/runs/qwen3-cacheon-greedy16-cachebuild-20261013-r1/`。本运行基线 HEAD `b94b05a5aa6227214091938b601999a7a0dd9115`；同步目标 `origin/2.0-refactor` SHA `4896af611650d1436302c06b0653ac61726fd10c` 为该 HEAD 祖先。dirty source diff SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；历史严格失败不追溯改判。

本轮报告提交前的仓库检查：`bash tools/check_repo_layout.sh` 通过，`git diff --check` 通过。按仓库要求在 Slurm `srun` worker 运行 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`，结果为 1,383 passed、8 skipped、4 failed。四个失败均发生在结构测试启动的隔离子进程中，私有编译器 wrapper 因子进程未提供该 run 的 `CCACHE_DIR` 而拒绝编译（`CCACHE_DIR: private run cache required`）；失败项为 `test_process_mode_contract` 两项、`test_root_domain_structure::test_native_cold_start_preserves_legacy_module_surfaces` 和 `test_nn_structure::test_first_import_paths_are_cycle_free_in_fresh_processes`。这不是本次文档差异导致的回归；本次只提交报告，不修改测试或编译器代码。


### Qwen3 cache-enabled near-context prefill 与 greedy decode（2026-10-13）

独立运行键 `qwen3-public-infer-batch2-cacheon-nearctx-greedy4-world1-20261013-r2` 增加了 near-context prefill 与 cache decode 的交叉覆盖。Slurm worker 预检 job 2991 确认公开 CLI 的 `--model` 明确指向实际 `cache-on-base` 目录、Adapter 为 `checkpoint-3`，模型与 generation config 均启用 cache；两个输入 raw token 长度为 115，模型/Adapter/input/initial-state 清单可读，下载关闭；native torch_npu 和 candidate ACL 可用，candidate core SHA256 `24b99842d51e481b79c45163c95fd8fcc43c1ad1955cf9127c8c3f8fd9f379ad` 与已验证 cache 匹配，strict `backend_fallback=error` 且预检 fallback=0。

Native oracle job 2992、只读原生门禁 job 2993、strict ACL candidate job 2994 与 worker comparator job 2995 全部退出 0；所有检查器均在标记的 `srun` worker 中。原生门禁观察到一个实际 batch=2 请求，四次 NPU forward 输入长度 `[124,1,1,1]`：prefill 长度 124，随后三个 cache decode 均只输入一个 token。两侧加载参数、输入 IDs、greedy token 和解码响应精确相同；四组 logits 最大绝对差为 0（锁定容差 `atol=1e-5, rtol=1e-4`）。candidate 计算在 NPU，结束 fallback 计数 0。此锁定格 L0/L1/L4 通过、L5 未运行；不能推广到其他上下文位置或 cache 性能比较。

前置键 `...-r1` 的 job 2990 发现 runner 的 native/candidate `--model` 仍指向旧 cache-off fixture，而预检未检查 CLI 模型路径；该键未运行 oracle/candidate，作为 harness-only 失败保留。新键 r2 加入 worker 内命令路径断言后再做预检和 oracle-first。原始 manifest、脚本、快照和日志未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-nearctx-greedy4-world1-20261013-r1/` 与 `...-r2/`；没有重下已校验模型或复用旧运行键。


### Qwen3 cache-off near-context 公开推理配置合同失败（2026-10-13）

新键 `qwen3-public-infer-batch2-cacheoff-nearctx-greedy4-world1-20261013-r5` 的 srun-worker 预检 job 3000 通过：公开命令指向本地模型目录 `base` 与 Adapter `checkpoint-3`，两条输入的 raw token 长度均为 115，模型和输入/初态清单可读，下载关闭。Native job 3001 进入公开推理配置后，observer 读到实际 `GenerationConfig.use_cache=true`，而本地 `config.json` 与 `generation_config.json` 都是 false，因此 gate 停止；没有运行 strict ACL 候选。此格的公开请求语义失败，不是 ACL 兼容性失败；L5 未运行。

尝试历史按原样保留：r1 在准备脚本时报 TypeError；r2 的 worker 预检 job 2996 错指向旧 run root；r3/job 2997 暴露 CLI 实际仍用 cache-on 模型路径；r4/job 2998 预检通过，但 native job 2999 因预先创建 `native/observations` 与 observer 的 `mkdir(exist_ok=False)` 冲突、在推理前停止；均未形成模型兼容性结论。r5 的 generation-config 快照、日志和 manifest 位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheoff-nearctx-greedy4-world1-20261013-r5/`。

下一步需要修正 Swift 推理配置路径，使有效 generation config 继承模型本地 cache-off 设置，或提供明确的公开 `--use_cache` 参数；这涉及 Swift 源码行为变更，先等待用户确认，不在当前 Jittor 仓库代改，也不把历史 Mistral cache-off 失败改写。

### Qwen3 公开 `swift export --merge_lora` safetensors parity（2026-10-13）

独立新键 `qwen3-public-export-merge-lora-world1-20261013-r5` 完成公开导出路径的原生/strict ACL 工件比较。所有检查器均在 srun worker：预检 job 3008、native export 3009、native artifact gate 3010、strict ACL export 3011、candidate artifact gate 3012、只读 comparator 3013 均通过。复用本地已校验 `base` 与 `checkpoint-3`，无下载。两个导出均由公开 `swift export --merge_lora` 产生单一 `model.safetensors`，包含 25 个 tensor keys；配置和键/形状/dtype/有限值门禁一致，逐 tensor `atol=1e-5, rtol=1e-4` 对照通过，最大绝对差 0；Adapter 工件已合并/移除。ACL 运行使用 `backend_fallback=error`，完整进程结束计数为 0，日志记录 `device_map='npu:0'`。

此证据只支持该 Qwen3 fixture 的公开导出与序列化 parity，不覆盖导出物重新加载/推理或参数直接设备快照，不授予通用 L0–L5 模型行为结论。前置 r1/job 3002（symlink 路径断言）、r2/job 3003（嵌套 help 命令退出 87）、r3/jobs 3004/3005（shell 输出引号错误）、r4/jobs 3006/3007（template 参数歧义）未生成导出工件，均作为 harness/config 失败保留。成功键的未版本化日志与产物位于 `$TASK_STATE/runs/qwen3-public-export-merge-lora-world1-20261013-r5/`。


### Qwen3 merged export 公开重载推理（2026-10-13）

新独立键 `qwen3-public-export-reload-infer-world1-20261013-r5` 只读复用已通过的父导出键 `qwen3-public-export-merge-lora-world1-20261013-r5`。所有门禁与比较器均在标记的 Slurm `srun` worker 执行：预检 job 3018、原生 public `swift infer` job 3019、原生 gate 3020、strict ACL candidate 3021、worker comparator 3022 全部通过。模型从原生和 ACL 两个 `output` 目录重载，未再加载 Adapter、未下载权重；两个导出目录的七个文件逐文件 SHA 在 worker 验证，父键 artifact comparator 结果为 pass。

两条固定 near-context prompt 形成真实 batch=2，raw token 长度 `[115,115]`；有效 GenerationConfig cache=true，gate 观察到四次输入长度 `[124,1,1,1]`。两侧的已加载参数、请求输入、greedy token 与响应精确一致；四组 logits 最大绝对差为 0（容差 `atol=1e-5, rtol=1e-4`）。Candidate runtime 为 ACL，`backend_fallback=error`，结束 fallback 计数 0；两侧参数、模型输入和 logits 记录均驻留 `npu:0`。

本格证明此 Qwen3 merged export 可由公开 infer 重载并完成锁定推理对照，L0/L1/L4 通过，L5 未运行；不推广到其它模型或导出设置。父导出文件的 `generation_config.json` 记为 use_cache=false，但 Swift 公开推理产生的有效 GenerationConfig 为 true，故此格不代表 cache-off。前置 r1/job 3014 因初态清单摘要不匹配、r2/job 3015 因缺少两侧导出 SHA 清单，在模型加载前停止；r3 只在准备阶段发现源目录未生成待复制清单、没有提交 Slurm；r4/job 3017 因 observer 拒绝预建 observation 目录，在推理前停止。这些均为 harness 准备失败，保留且未复用键。r5 manifest、日志、快照与比较结果位于 `$TASK_STATE/runs/qwen3-public-export-reload-infer-world1-20261013-r5/`；基线 HEAD `f45c31cb0214e14e454cc9e0217df90bfa3adeaa`，source dirty diff SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。

### Mistral tiny 公开 `swift export --merge_lora` safetensors parity（2026-10-13）

新独立键 `mistral-public-export-merge-lora-world1-20261013-r1` 完成公开导出路径的原生/strict ACL 工件比较。所有检查均在标记的 Slurm `srun` worker：预检 job 3023、native export 3024、native artifact gate 3025、strict ACL export 3026、candidate artifact gate 3027、只读 comparator 3028 均通过。复用本地 Mistral fixture 目录 `base` 与公开训练 checkpoint `checkpoint-3` Adapter，模型、Adapter、输入和初态清单哈希在 worker 复核通过；执行离线，没有下载权重。Native runtime 为 torch_npu/Ascend910B3；candidate 为 ACL、`backend_fallback=error`，完整进程结束 fallback=0。

两侧公开 `swift export --merge_lora` 输出均为单一 `model.safetensors`，包含 29 个 tensor keys；原生/候选配置一致，tensor keys、shape、dtype、有限值门禁通过，逐 tensor `array_equal` 全部精确一致，最大绝对差 0，导出目录中没有 Adapter 工件。此格仅证明该 Mistral fixture/Adapter 锁定设置下的公开导出和序列化 parity；不证明 merged 输出重载/推理、参数设备快照、其它模型/Adapter或性能。L5 未运行。cache-off 公开推理 configuration 失败继续保留，和本导出格无关，未追溯改判。

原始 manifest、模型/Adapter 清单、脚本、日志、快照和产物未版本化，位于 `$TASK_STATE/runs/mistral-public-export-merge-lora-world1-20261013-r1/`。基线 HEAD `fb4e0ebd06d512e1a57f98de97a6d125c32bc074`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。

本次文档提交门禁：`bash tools/check_repo_layout.sh` 通过。job 3032 在 worker 执行 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`，1383 passed、8 skipped、4 failed。四个失败均来自 pool 计算触发的 ACL fused `reindex/void` 未注册变体并因 `backend_fallback=error` 被拒绝：`test_average_pooling_direct_surface`、`test_maximum_and_minimum_pooling_direct_surface`、`test_unpooling_direct_surface`、`test_adaptive_execution_fields_and_pickle`。这是本次验证发现的核心算子覆盖缺口，不由本次文档差异引入；本轮未改动相关实现，也未放宽回退策略。失败调用的最早共同层为 fused reindex/void 注册/执行路径，若继续修复，应先为这些 pool 形状做最小原生与 ACL 复现，再评估注册新变体或改写融合路径。随后文档门禁续验使用新运行键 `result-doc-verify-qwen-topk8-20261013-r4/r5/r6`：r4/job 3064 在子进程冷编译超过 600 秒时有 2 项超时、其余 1385 passed/8 skipped；r5/job 3066 将同模块子进程超时调为有实测依据的 1800 秒，4 项均通过；当前文档版本的 r6/job 3072 在同指纹缓存上重新执行完整 worker `tests/structure`，1387 passed、8 skipped、0 failed，布局检查也通过；新增 Mistral 采样记录后，run key `result-doc-verify-mistral-topk32-20261013-r1` / job 3080 再次在 worker 完成全套结构门禁，1387 passed、8 skipped、0 failed（569.55 s），布局检查通过。原始状态未版本化，位于 `$TASK_STATE/runs/result-doc-verify-qwen-topk8-20261013-r4/` 至 `...-r6/`。

### Mistral tiny 导出物重载完整性失败（2026-10-13）

新键 `mistral-public-export-reload-infer-world1-20261013-r3` 复用已验证的 Mistral `base` 与公开 `checkpoint-3`，离线从父导出目录 `output` 启动公开 `swift infer`；所有预检、原生 gate、strict ACL candidate 和比较器均由 Slurm `srun` worker 执行。预检 job 3036 通过，native oracle 3037、gate 3038、ACL candidate 3039、comparator 3040 均退出 0。Native 为 torch_npu/Ascend910B3，candidate 为 ACL/`backend_fallback=error` 且 fallback=0；输入与初态 manifest 可读，未下载权重。

这不是完整 merged Adapter 的通过结果。父键 `mistral-public-export-merge-lora-world1-20261013-r1` 输出两侧皆有 29 个相同 safetensors keys，但源 Adapter 的 `adapter_config.json` 标记 `swift_type=ADAPTER`，导出 `args.json` 的 `tuner_type=adapter`，同时父运行命令确实请求 `merge_lora=true`。标准 `MistralForCausalLM` 加载器警告并忽略 8 个 `adapter_default` state keys；只读 worker 审计 job 3044 将这 8 个输出键与源 Adapter 状态逐一核对，shape 也一致。因此该导出不满足 adapter 参数已合并/不再依赖 Adapter 层的完整性门禁，矩阵的 L0/L1/L4 标为 failed。

推理观察仍可作为范围有限的 ACL 对照：native 与 ACL 各只加载了共同的 21 个命名参数；这些参数及两批输入完全相同，4 次 logits 比较在 `atol=1e-5, rtol=1e-4` 内且最大绝对差 0，greedy tokens/responses 精确相同。有效 GenerationConfig 的 cache 为 true，故不代表 cache-off。比较器 job 3040 退出 0，但其 stdout 与比较 JSON 共用文件，导致报告整体 JSON 格式损坏；job 3044 独立复算快照、核验首行摘要后确认上述有限范围的比较。

历史 harness 失败保留：r1/job 3033 的 native preflight 没有写 JSON；r2/job 3035 的 observer 假设 generate 绑定到 Adapter 内层模型，在第一次 forward 前错误停止；只读审计 r1/job 3041 在解析损坏报告时失败，r2/job 3042 已通过数据断言、只在组装报告时误读不存在的 `args.json.merge_lora` 字段；修正后的审计 r3/job 3044 完成。原始日志、manifest、权重清单和快照未版本化，位于 `$TASK_STATE/runs/mistral-public-export-reload-infer-world1-20261013-r3/`、`$TASK_STATE/runs/mistral-public-export-merge-lora-world1-20261013-r1/` 与 `$TASK_STATE/runs/mistral-export-reload-weight-coverage-audit-world1-20261013-r3/`。该根因是 Swift ADAPTER/`merge_lora` tuner-config 不匹配，不是 ACL 数值分歧；修复需要验证适用的公开导出入口并新建运行键，未在本轮改源码。此前严格对拍结论均保持原状态。

### Qwen3 tiny cache-enabled top-k=8 sampling（2026-10-13）

从未完成采样矩阵选择三个独立的公开 `swift infer` 单 NPU 格。均复用本地已校验 Qwen fixture `cache-on-base-topk4` 与 Adapter `checkpoint-3`、四条固定 prompt、实际 batch=2、每条至多四个生成 token；模型/Adapter/输入/初态清单在 worker 可读，离线未下载。所有预检、原生门禁和比较器均由 Slurm 标记的 `srun` worker 执行；native 为 Ascend910B3 `torch_npu`，候选为 Jittor 2.0.0/ACL、`backend_fallback=error`，三格结束 fallback count 均为 0，复用的 core SHA256 为 `24b99842d51e481b79c45163c95fd8fcc43c1ad1955cf9127c8c3f8fd9f379ad`。

格 A `qwen3-public-infer-batch2-cacheon-topk8-sample-multistep4-world1-20261013-r2` 锁定 top-k=8、top-p=1、temperature=1；worker 预检 job 3050、native oracle 3051、native gate 3052、strict ACL candidate 3053、worker comparator 3055 均成功。native gate 观察到两批 batch=2、八次前向及每个请求的 `[prefill,1,1,1]` cache 输入长度。比较器确认初始参数和 prefill 输入精确一致，prefill logits 最大绝对差 0（`atol=1e-5, rtol=1e-4`），16 个/每侧采样 token 都在各自 top-k=8 支持集内且 EOS-aware；两侧本次四个输出序列相同。L0/L1/L4 通过，L2/L3 不适用，L5 未运行。

格 B `qwen3-public-infer-batch2-cacheon-topk8-topp08-temp07-sample-multistep4-world1-20261013-r1` 锁定 top-k=8、top-p=0.8、temperature=0.7；worker 预检 3056、native oracle 3057、native gate 3058、strict ACL candidate 3059、worker comparator 3060 全部成功。比较器确认相同输入和初态、有效 cache=true、逐 runtime top-k 后 top-p 候选集合、prefill logits 最大绝对差 0（同一容差），两侧本次四条生成序列相同。L0/L1/L4 通过，L2/L3 不适用，L5 未运行。两个格仅覆盖各自锁定采样参数、微型 Qwen fixture 与短生成窗口，不证明 RNG 等价、训练/恢复、性能、cache-off 或其它模型。

格 C `qwen3-public-infer-batch2-cacheon-topk8-topp1-temp07-sample-multistep4-world1-20261013-r1` 固定 top-k=8、top-p=1、temperature=0.7，以与格 A 只差 temperature 的方式隔离该参数。worker 预检 job 3067、native oracle 3068、worker 原生 gate 3069、strict ACL candidate 3070 和 worker comparator 3071 全部 `COMPLETED 0:0`。两侧实际运行四条 prompt、两个 batch=2 请求、各生成 16 个 token；gate 观察到 8 次前向，cache 输入长度分别为 `[15,1,1,1]` 与 `[17,1,1,1]`。Comparator 核验初态参数与 prefill 输入精确、prefill logits 最大绝对差 0，匹配 decode 步 logits 满足 `atol=1e-5, rtol=1e-4`，采样 token 分别属于各自温度调整后 top-k=8 支持集；本次两侧序列相同，但不将其作为 RNG 等价判据。ACL 两个进程均为 `backend_fallback=error`、完整观察窗 fallback=0。此锁定格 L0/L1/L4 通过、L2/L3 不适用、L5 未运行；结果不推广至其它温度、top-p/采样组合、训练恢复、cache-off 或其它模型。原始证据未版本化，位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk8-topp1-temp07-sample-multistep4-world1-20261013-r1/`。运行基线 HEAD `e5902af38151ab3b53fee819364ec6ce5c78faa3`，dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；远端同步 SHA 仍引用本轮前已记录的祖先 `4896af611650d1436302c06b0653ac61726fd10c`，本轮未重新 fetch。

格 A 的前键 `qwen3-public-infer-batch2-cacheon-topk8-sample-multistep4-world1-20261013-r1` / job 3049 在模型加载前因预检将 `--model` 目录误写为不存在的 `cache-on-base-topk8` 而失败；锁定 fixture 的真实目录名为 `cache-on-base-topk4`。这是 harness 路径错误，不是适配失败；该键未运行原生推理或 ACL 候选并保留为 L0 harness failure，修正后使用新键 r2，没有重跑 job 3049。两个通过格运行 manifest、清单、日志、worker 快照与比较结果位于 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk8-sample-multistep4-world1-20261013-r2/` 与 `$TASK_STATE/runs/qwen3-public-infer-batch2-cacheon-topk8-topp08-temp07-sample-multistep4-world1-20261013-r1/`；前键日志保留在其 `...-r1/` state 目录。运行基线 HEAD `66862e89c1bcff69fda6de3c91785e9c1dde184c`，同步目标 SHA `4896af611650d1436302c06b0653ac61726fd10c` 是该基线祖先；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`。


### Mistral cache-enabled top-k=32 + top-p=0.9、temperature=0.7 四步 sampling（2026-10-13）

独立公开 `swift infer` 格 `mistral-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261013-r1` 使用四条固定 20-token 提示，分两个真实 batch=2 请求，每条最多生成四 token。worker 预检 job 3073、torch_npu/Ascend910B3 oracle job 3074、只读原生门禁 job 3075、strict ACL candidate job 3076 均通过；实际模型目录为 `cache-on-base`、Adapter 为 `checkpoint-3`，模型、输入及初态清单在 worker 可读且未下载权重。候选使用已校验 core SHA256 `24b99842d51e481b79c45163c95fd8fcc43c1ad1955cf9127c8c3f8fd9f379ad`、ACL 与 `backend_fallback=error`，两进程结束 fallback 均为 0。原生门禁观察到 cache=true、top-k=32/top-p=.9/temperature=.7、两个 batch=2 请求各四次 NPU 前向，输入长度均为 `[29,1,1,1]`；base 参数在 NPU，adapter 参数快照有 8 个 CPU 参数，但其 32 个 adapter forward event 的输入/输出均位于 `npu:0`。

原比较器 job 3077 使用旧 top-k=64/top-p=.8 常量，审查结果先失败；只读 audit-r1 job 3078 修正采样常量后，又错误地将 EOS=2 后的 pad=0 当作采样 token。这两次均是 comparator harness 错误，没有重跑原生或 ACL candidate。只读 audit-r2 新键 `mistral-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261013-audit-r2` 的比较器 job 3079 通过，并只复用父键已完成的四组产物。最终比较确认两侧加载参数及 prefill 输入精确相同，prefill logits 最大绝对差 `8.940696716308594e-08`，8 组匹配 decode logits 满足 `atol=1e-5, rtol=1e-4`；26 个实际采样位置分别属于对应 runtime 经温度缩放、top-k=32 后 top-p=.9 的支持集，EOS 后 6 个位置均为 pad=0 并从采样检查中排除。cache 下一步输入消费前一 token。两侧本次生成序列相同，不据此声称 RNG 等价。该锁定配置 L0/L1/L4 通过，L2/L3 不适用、L5 未运行；不外推到 cache-off、其它采样参数、训练/恢复或其他模型。

原始 manifest、日志、哈希清单、快照与两次只读审计位于 `$TASK_STATE/runs/mistral-public-infer-batch2-cacheon-topk32-topp09-temp07-sample-multistep4-world1-20261013-r1/` 和同前缀 `...-audit-r1/`、`...-audit-r2/`。运行基线 HEAD `aebe22d6c159f3614c0c31c74099d273feb94016`；dirty source patch SHA256 `192171d2616b3494cd3fe94f91141e168827e471d7519c56362b7c1d269c04b3`；本轮沿用此前已记录的同步祖先 SHA `4896af611650d1436302c06b0653ac61726fd10c`，未重新 fetch。
