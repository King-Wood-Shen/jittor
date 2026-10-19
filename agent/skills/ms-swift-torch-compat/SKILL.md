---
name: ms-swift-torch-compat
description: 基于干净 ms-swift 注册表和公开入口，验证 Jittor Torch shim 在 CUDA/Ascend 上的算法、推理、并行及开关；复用有效证据并修复底层根因。
---

# ms-swift × Jittor Torch 兼容

## 目标和证据边界

目标是尽可能广地支持 ms-swift 的公开训练和推理入口。所有结论绑定 ms-swift、Jittor 完整 SHA、设备、模型、配置、验证命令和原始日志。一个 tiny 模型通过不代表整库支持；注册可见、导入成功、脚本启动不代表算法通过。

本轮优先功能和正确性、训练收敛及目标设备峰值显存；不要求速度、吞吐或耗时验证。未测的性能等级仍记“未测”，不能沿用旧数值。按 downstream-library-adaptation/SKILL.md 分别报告 L1 导入/前向，L2 独立 PyTorch 对照的前向与全部请求梯度，L3 多步训练轨迹，L5 真机零 fallback。训练业务验收另记录 loss 是否稳定收敛；收敛不能冒充严格 L2 对拍，L3 也不能自动推出 L2。推理按真实调用路径对照输出、logits、KV cache 等适用结果。


## 本轮主要优化与开关

按[主要优化与开关的验收路径](references/priority-optimizations.md)选择代表场景；本轮记录训练收敛、功能正确性、零 fallback 和设备峰值显存，不把耗时或吞吐作为验收门槛。旧性能数据只按原基线解释。

## 起步和复用

1. 阅读本分支 AGENTS.md、github-collaboration-commit/SKILL.md、downstream-library-adaptation/SKILL.md、飞书项目要求和设备端已有日志。记录官方最新 2.0-refactor 的 Jittor SHA、当前修复分支 SHA、ms-swift SHA、依赖和硬件。
2. 从干净上游源码用 registry_inventory.py 枚举模型族、训练方法、Trainer/CLI 及推理入口。模型、算法、设备、并行、开关分别写 X/N 和未抽查数，不能相加成一个支持率。登记项不等于可执行项。
3. 每项先查既有代码 SHA、配置、输入、checkpoint、日志、断言和零 fallback 证据。同一源码和执行路径已通过则直接继承，不重跑；源码改动触及依赖图时仅重验受影响项。无可核验证据标“缺证据”。
4. 修复前保存 Git 节点与最小复现。以官方最新 2.0-refactor 的独立工作树为基线；旧补丁仅作线索，不盲目覆盖。定向回归后再测受影响路径。同一根因连续五次失败，记录后转下一项。

## 覆盖轴

| 轴 | 逐项登记和验收 |
| --- | --- |
| 训练算法 | SFT/全参、LoRA/其他 tuner、预训练、DPO、CPO、SimPO、ORPO、KTO、GRPO、PPO、RM、GKD 及当前注册表其他可执行方法。用小模型和固定输入走公开入口多步更新；记录 backward、优化器、loss 轨迹、checkpoint 恢复和 fallback。可选依赖及外部引擎单列。 |
| 并行方式 | 单卡、真实双卡数据并行；FSDP、ZeRO/DeepSpeed、序列、张量、流水并行等模式按实际归属判定。适用模式检查两 rank 实际执行、同步、梯度、checkpoint 保存和恢复。双卡不能推广四卡或多机；缺硬件/依赖写未验证。 |
| 推理和生成 | greedy、采样、batch、LoRA、cache 开关、生成配置、量化、投机解码等从源码枚举的开关。验证开关实际改变执行路径及适用的输出、logits、KV cache、接受率。投机解码只有经 ms-swift 公开入口且目标/草稿模型走 shim 才计入；vLLM/SGLang 等独立引擎路径应标外部所有权。 |
| 设备和数值 | CUDA、Ascend 分开；用同权重同输入的独立 PyTorch/torch_npu 同设备 oracle。核验 dtype、实际 device、fallback_policy=error、fallback_count=0。CPU 只用于定位共享 API 根因，不代替目标卡。 |

不跑全组合笛卡尔积。核心模型族 × 公开算法 × 目标设备走必要路径，架构、精度、并行及开关差异分层抽样，未跑格子可见。小模型验证兼容和收敛，不做性能结论。真实单卡和双卡为当前目标；四卡、八卡和多机如无独立需求仅登记。

## 定位、修复和记录

先在原生 PyTorch 同版本配置复现，再判断环境/版本、Jittor 核心、Torch compat、后端算子、分布式运行时或 ms-swift 私有 adapter 的归属。修共享语义根因，避免按算法加旁路；跨库改动检查其他调用方及 dtype/device 组合。严重或跨架构问题按飞书格式写现象、最小复现、根因、影响、已尝试次数、候选方案和证据；可控问题直接修复并继续。每个功能或根因单独提交，依 github-collaboration-commit/SKILL.md 验证后推到自己的分支。

每项结果至少记录入口/开关、模型、算法、并行、设备、双方 SHA、命令、日志、L1/L2/L3/L5 各自状态、loss 或推理输出、fallback、修复提交及未覆盖边界。历史证据保留在 docs/results/2026-10-15-swift-trainer-installed-source.md 和 docs/results/2026-10-14-downstream-oracle-scorecards.md，仅在源码与路径一致时继承。