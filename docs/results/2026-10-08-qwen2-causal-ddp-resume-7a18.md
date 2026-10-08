# Qwen2-0.5B 双卡公开 SFT checkpoint 恢复对拍

- 状态：L3 恢复证据为 partial；本配置的完整 L0–L5 未通过验收。
- 日期：2026-10-08。
- 基线：Jittor `f502887dd507688651628090fbdb91bc8b21a665`，已同步上游基线 `origin/2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B 全参数 FP32、公开 `swift sft` CLI、双 RTX 4090 NCCL、SGD、固定四条样例、每卡 batch 2、`max_steps=4`；从 step 3 checkpoint 恢复到 step 4。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补齐此前同配置的输入/参数映射及逐参数梯度门槛后再提升 L3；Torch shim checkpoint、Swift Trainer、NCCL 或 RNG/dataloader 恢复语义变化时复验。

## 运行与证据

Slurm 13020 在 `cscg-qh04` 依次运行原生 oracle 和 strict shim。GPU 为 RTX 4090，UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea` 与 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`；NCCL header 版本 `2.21.5`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、Jittor 2.0.0、ms-swift 4.6.0.dev0。模型文件 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，数据 SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。

每个 runtime 都执行了从初始模型连续训练四步，以及从自身既有 step 3 checkpoint 恢复并训练到 step 4。原生使用 `20261008-qwen2-causal-ddp-02498-merge-fb73-v1/native-output/checkpoint-3`；shim 使用 `20261008-qwen2-causal-ddp-02498-7a18-v4/shim-output/checkpoint-3`。所有 CLI 子进程均以两个 rank 运行。候选 rank 0/1 日志记录 Torch 与 Jittor rank 对应、两张 GPU UUID、`use_cuda=1`、shim 已启用，启动和退出 `fallback=0`。训练与恢复顺序执行，复用已完成双卡运行的 Jittor cache。

Slurm 13020 保存了四个 step 4 checkpoint。每份 `trainer_state.json` 的 `global_step=4`；模型、optimizer、scheduler 和 rank 0/1 RNG 文件均存在。Slurm 13024 在 GPU worker 上比较了模型与恢复状态：原生连续/恢复 290 个参数键、494,032,768 个元素完全相同；shim 连续/恢复最大绝对差 `3.7253e-9`、相对 L2 `3.3988e-11`。step 4 loss 原生连续与恢复均为 `3.8139588833`；shim 分别为 `3.8139617443` 与 `3.8139622211`。两个 runtime 各自的 optimizer 状态 306 项、scheduler 状态 8 项及 rank 0/1 RNG 数组连续与恢复逐项相同。

跨 runtime 的 scheduler 状态 key 结构和 RNG tensor 编码存在差异；不把它们强行解释为逐字节相同。报告只确认各 runtime 内部从其 checkpoint 恢复后继续到 step 4 的状态与结果。没有单独导出 dataloader sampler cursor，也没有在这次运行中采集完整逐参数梯度，因此恢复路径有直接数值支持，但不能据此填满更早层级或整个双卡 SFT 面。

## L0–L5

| 层 | 状态 | 证据与缺口 |
| --- | --- | --- |
| L0 | partial | 原生与 strict shim 构造公开训练 CLI 并保存完整模型 checkpoint；缺该运行完整逐参数 device/dtype 和初态映射清单。 |
| L1 | partial | 训练损失与 step 4 轨迹对齐；没有独立同权重、同输入的完整 logits/hidden-state 对拍。 |
| L2 | partial | 更新后全模型权重、loss 和恢复状态可比较；本运行没有全 trainable 参数逐项梯度证据。其他运行的梯度报告不自动继承本运行。 |
| L3 | partial | 连续四步与 step 3→4 恢复的模型、optimizer、scheduler、rank RNG 和 step 4 指标在各自 runtime 内对齐；sampler cursor 未直接比较，且 L0–L2 门槛未完整通过。 |
| L4 | partial | 原生与 strict shim 均通过公开 `swift sft` CLI 完成连续和恢复流程；前序层级未齐，不能将其作为完整端到端通过。 |
| L5 | blocked | 前序层级未通过；本运行不含稳态性能协议。 |

Slurm 13021/13023 是状态检查脚本的早期失败尝试；13024 成功生成上述状态比较。报告文档门禁 Slurm 13181 的完整 Torch 模式结构测试为 1385 passed、6 skipped、1 failed；失败仅指出新页面未列入 toctree。补上目录入口后，Slurm 13187 的布局和 Manifest 检查通过，定向页面可达性测试 1 passed。原始脚本、checkpoint、环境、状态 JSON 和日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-ddp-resume-7a18-v1/`；报告门禁日志在 `20261008-qwen2-ddp-resume-report-gate-v1/` 和 `20261008-qwen2-ddp-resume-report-gate-v2/`；关联三步基准在相邻 `20261008-qwen2-causal-ddp-02498-*` 运行目录。

上游远端本轮未能实时读取：HTTPS fetch 遇到 `SSL_ERROR_ZERO_RETURN`，GitHub SSH 拒绝当前公钥。文中的同步 SHA 是该 Slurm 运行时记录的基线，不表示本次确认了实时远端头。
