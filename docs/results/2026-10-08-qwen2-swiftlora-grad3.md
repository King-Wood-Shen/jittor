# Qwen2-0.5B ms-swift 私有 Swift LoRA 三步梯度与更新对拍

- 状态：固定配置下，私有 Swift LoRA 直接 API 的三步梯度、AdamW 状态与 adapter 更新通过；不代表公开 `swift sft`/Trainer、恢复或性能通过。
- 日期：2026-10-08。
- Jittor 基线：集成 HEAD `15ff2f5ce81b7be924e223c2b14b0673ab3fc756`；上游 `2.0-refactor` SSH 查询 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`，已是 HEAD 的祖先。HTTPS fetch 因 GitHub TLS 连接关闭失败。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Swift LoRA tuner、torch shim autograd/AdamW 改动时；公开训练入口、完整 checkpoint 或新进程恢复另立验收。

功能运行键 `20261008-qwen2-swiftlora-grad3-v3`，Slurm 13254 在 cscg-qh15 的 RTX 4090（UUID `GPU-afd56a2e-3deb-3bd7-f122-1075e6de8956`，driver `580.178.04`）先运行独立原生 PyTorch CUDA 进程，再运行严格 shim。使用 `AutoModelForCausalLM`、`swift.tuners.LoRAConfig` 与 `Swift.prepare_model`，Qwen2-0.5B FP32/eager，rank 8、alpha 32、目标 `q_proj/v_proj`、dropout 0、`lorap_lr_ratio=None`。固定两个短提示 tokenizer 输出为同一 CUDA batch，batch shape `[2,7]`；原生侧保存的 96 个 trainable adapter 张量被逐键载入候选。两边各自用 AdamW（lr `1e-3`、weight decay `0.01`）执行 3 步 causal LM loss 训练，未使用 ms-swift Trainer 或公开 CLI。

全部 386 个模型参数、输入和候选梯度均驻留 CUDA；每一步 96/96 个 trainable 参数梯度均存在且有限。每个参数逐步比较梯度、更新后权重、AdamW `exp_avg` 与 `exp_avg_sq`，共 1,152 组张量。三步 loss：原生 `[7.32767105, 4.59883881, 2.90620136]`，shim `[7.32767153, 4.59883976, 2.90620375]`。所有比较张量最大绝对差 `1.20095e-3`，最坏单张量相对 L2 `3.76535e-3`，在 CUDA backward `2e-2` 容差内。候选设置 `use_cuda=1`，整个运行 `fallback_count=0`。

首次脚本运行键 `20261008-qwen2-swiftlora-grad3-v1` / Slurm 13251 在原生模型计算前因探针忘记将 tokenizer 输入放到 CUDA 失败；v2 / Slurm 13253 已进入原生训练，但日志语句参数错误导致保存前退出。它们均未运行 shim，不算功能结果。v3 完成两侧模型运行；汇总作业 13254 首因数组数断言错误退出，13255/13256 的归一门槛将近零张量的比值误当作绝对容差而失败，13257 使用报告的 max-absolute 与 relative-L2 标准完成后处理。模型没有因这些汇总器错误重跑。原始脚本、NPZ、JSON 与日志保存在未版本化 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-swiftlora-grad3-v1/`、`v2/`、`v3/`。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | partial | 原生与 shim 构造 Qwen2 和私有 Swift LoRA；训练参数键/数量一致并驻留 CUDA。未构造 Swift template、dataset、collator、Trainer，也未完整比较 386 个 state-dict 键及所有 dtype。 |
| L1 | not-run（本运行） | 本次训练轨迹记录 loss 与梯度，没有另存 logits/hidden-state；固定状态直接前向另见 [前向报告](2026-10-08-qwen2-swiftlora-forward.md)。 |
| L2 | PASS（固定直接训练） | 96/96 trainable 梯度三步齐全；loss、全部 adapter 更新权重及 AdamW 两组状态比较 1,152 张量，误差在 `2e-2` 门槛内。 |
| L3 | not-run | 未保存/恢复 adapter、optimizer、scheduler、RNG 或 dataloader 状态。 |
| L4 | not-run | 未运行公开 `swift sft`、launcher 或 ms-swift Trainer；直接调用私有 tuner API 不能代表公开入口。 |
| L5 | blocked | 前序 L0/L1/L3/L4 未通过或未运行；没有稳态性能协议。 |
