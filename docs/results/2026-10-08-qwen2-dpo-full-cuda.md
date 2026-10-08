# Qwen2-0.5B 公开 DPO 三步 CUDA 对拍：strict backward 失败

- 状态：当前固定配置下原生公开训练三步完成；严格 CUDA shim 已构造训练器并进入首步反向，但在 `grad_optional` 抛出 Jittor 内部 `nano_vector.h:41: slice overflow`。这是兼容失败证据，根因尚未定位；不代表所有 DPO 或 RLHF 配置。
- 日期：2026-10-08。
- 基线：Jittor `97a66cf5a650461ac08f9cdf609e4eddb25a230b`（本提交仅更新文档）；同步上游缓存 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`，本轮上游 HTTPS fetch 因 `SSL_ERROR_ZERO_RETURN` 未核验实时值；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B、公开 `swift rlhf --rlhf_type dpo`、全参数 FP32、eager attention、本地固定四行偏好数据、batch 1、SGD、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：定位 `grad_optional` 中触发损坏 `NanoVector` 范围的 DPO 图路径，再以最小原生/strict CUDA 复现验证修复；不要复跑长轨迹代替根因定位。

Slurm 13175 在 `cscg-qh04` RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`）完成原生 PyTorch DPO 公开训练三步并写出 `native-output/checkpoint-3`。模型文件 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；四行本地偏好数据 SHA256 为 `57b44eaa07bdf2f2a82d5f407df5b04e7ae81ac4f3be04f65ef3ed07322cde28`。逐步原生 loss 为 `0.69314706`、`0.69314176`、`0.69329756`，最终 `train_loss=0.69319546`。

严格候选 Slurm 13176 使用同一公开 CLI、模型与数据，启动日志显示 494,032,800 参数且全参数可训练；`sitecustomize` 中 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()` 从导入前启用。launcher、CLI 与 RLHF 子进程退出日志均为 `use_cuda=1`、shim 标记存在、`fallback=0`。作业在 `swift/rlhf_trainers/dpo_trainer.py:494` 调用 Transformers `training_step` 后，于 `accelerator.backward(loss)` 进入 `autograd_api.py:241` 的 `jt.core.grad_optional` 时失败：`slice overflow: 94674578185759 0 1`。候选没有完成 step 1、没有保存 checkpoint，也没有可用的数值/梯度对拍；这是 Jittor 内部不变量失败，不能视为数据或参数错误。

为避免误把启动/缓存故障当作 DPO 兼容证据，前序原始日志均保留：13169 原生预检路径未进入 ms-swift checkout；13170/13171 在 CUDA capability 查询子进程持续休眠时取消；13174 取消于脚本准备错误，未进入模型阶段；13175 首次 `jit_utils_core` 冷编译辅助进程停滞，候选模型未启动。13176 复用同驱动/ABI 的已验证串行 Jittor 缓存后进入真实训练并得到上述反向错误。所有运行键均在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-dpo-3step-v*/`；未修改 ms-swift 源码。

| 层 | 状态 | 证据/缺口 |
| --- | --- | --- |
| L0 | partial | 原生与 shim 构造真实 DPO trainer/model；候选 CUDA runtime 与零 fallback 已证，未保存逐参数 device/dtype 清单。 |
| L1 | not-run | 候选在首步反向前未记录同权重同输入 logits/log-prob 对拍。 |
| L2 | failed | 原生三步完成；候选首步反向触发 Jittor 内部范围不变量错误，没有梯度或更新结果。 |
| L3 | blocked | L2 未通过，未做恢复轨迹。 |
| L4 | partial | 公开 CLI 到达 DPO training step，但未完成训练或保存 checkpoint。 |
| L5 | blocked | 前序层未通过，未做性能测试。 |
