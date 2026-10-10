# Qwen2-0.5B 公开双卡全参数 SFT：strict shim 第六步 backward 失败

- 状态：原生 PyTorch 完成固定配置 12 步；strict shim 在第 6 步 backward 失败。初态、输入、前五步逐参数梯度和真实 CUDA/NCCL 已有审计证据，但完整收敛、checkpoint/恢复和性能合同未通过。
- 日期：2026-10-10。
- 基线：Jittor `1a1997ce1e52c415a6274eb8192a12affc153742`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B、全参数、FP32/eager、公开双卡 `swift sft`，固定重复样本，batch 1/GPU、SGD `lr=1e-5`、`max_grad_norm=1`、constant scheduler、12 步。仅代表此配置，不表示所有 SFT 或 ms-swift 兼容。
- 维护者：ms-swift CUDA 适配。
- 复查条件：先将第六步失败缩至可解释的最小调用/图路径并确认根因，再以新运行键完成相同 native/strict CUDA 配置的 12 步梯度、更新、完整状态与恢复对拍；前置层通过后另测稳态性能。

权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，固定八行重复数据 SHA256 为 `71e308ff953f9d89feec800d83c307fcedb5ce4cad60ce70e7359ccb1fe026c5`。两张设备为 RTX 4090，UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea` 与 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。原生独立 oracle 运行键 `20261010-qwen2-causal-ddp-fixedbatch-convergence-v2` / job 15454 完成 12 步；只读 GPU 审计 job 15457 确认每步两 rank 输入相同、每 rank 290 项有限非零 CUDA 梯度且 rank 梯度 hash 一致，loss 首三步均值 `3.89629332` 降至末三步 `3.84225074`，checkpoint-12 含 290 个权重键、243 个张量相对初态改变。

候选严格 shim 运行键 `20261010-qwen2-causal-ddp-fixedbatch-convergence-shim-v2` / job 15515 采用相同公开 CLI、数据和初态。双 rank 分别映射至上述两张 GPU；父子进程均记录 `use_cuda=1`、strict shim、`fallback=0`。初始 290 个参数 hash 与 native 相同，且前六步候选两 rank 的输入相同。训练在第六步 rank 1 的 `loss.backward()` 触发 `tensor/autograd_api.py:241 -> jt.core.grad_optional`，异常为 `nano_vector.h:41: slice overflow: 94160115837290 0 1`；rank 0 随后由 launcher 终止。没有完整 12 步、checkpoint-12 或恢复产物。

只读 GPU artifact audit 运行键 `20261010-qwen2-causal-ddp-fixedbatch-shim-partial-audit-v1` / job 15528 检查已有文件，确认 shim 前五步每步 290 项梯度有限、非零且位于 CUDA，两 rank 梯度 hash 相同。与 native 对比的逐参数梯度范数最大相对差依次为 `8.79e-6`、`2.36e-5`、`1.45e-5`、`6.81e-6`、`8.47e-6`；hash 不同。step 1 的完整 logits 最大绝对差 `0.09867`、loss 最大绝对差 `0.03574`，仅作诊断，不据此宣称数值对齐。step 1–5 shim loss 由 `3.902385` 降至 `3.87809229`，但未达到预先定义的 12 步收敛窗口，不能据此判 L2 通过。

双卡 no-build 预检 job 15511 在真实两卡上完成 strict shim NCCL all-reduce：rank 分别绑定 `cuda:0`/`cuda:1`，结果误差为 0、fallback 为 0，14 项共享库路径与 hash 前后不变。这只证明该预检路径的设备、NCCL 与缓存条件，不替代模型训练证据。所有原始文件均保留在未版本化 state 运行目录；artifact audit JSON 位于 `20261010-qwen2-causal-ddp-fixedbatch-shim-partial-audit-v1/audit-15528.json`。

第六步异常的具体触发图尚未定位。它与已知 DPO 条目 KI-COMPAT-010 的异常文字相同，但该 DPO 断点在首个 backward，本 SFT 断点在五次更新之后；目前没有证据证明两者根因相同，因此不合并归因，也不据此修改运行时。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 290 项初始权重 hash 与两 rank 设备记录已审计；其余 tokenizer/template、完整模型状态映射未完成。 |
| L1 | partial | 同一初态和输入；首步输出形状/dtype/有限性有效，但 logits 与 loss 的绝对差仅为诊断值，完整前向数值合同未验收。 |
| L2 | failed / incomplete | 前五步 290 项梯度均有限非零并有范数比较；第六步 backward 失败，无完整预定窗口与候选末态更新/checkpoint。 |
| L3 | blocked | 没有 shim checkpoint-12，未运行同进程或新进程恢复。 |
| L4 | 入口已运行，功能未通过 | 公开 `swift sft` strict-shim 入口实际执行五步后失败；不能记为 L4 通过。 |
| L5 | blocked | 正确性前置层未通过，未测预热后稳态延迟、吞吐与显存。 |

双卡预检与失败训练不构成全 ms-swift 兼容结论。后续不得复用上述运行键或 job；除非最小复现/证据对根因提供实质新信息，否则不直接重启该故障轮次。
