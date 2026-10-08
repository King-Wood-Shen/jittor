# Qwen2-0.5B 公开 DPO 三步 CUDA 对拍：strict backward 失败

- 状态：当前固定配置下原生公开训练三步完成；严格 CUDA shim 首步反向仍在 `grad_optional` 抛出 Jittor 内部 `nano_vector.h:41: slice overflow`。补充的首批前向探针发现同一 seed 下两侧公开 CLI 取到不同偏好样本，因此目前没有同输入 DPO 前向数值对拍。兼容失败范围只限该配置，不代表所有 DPO 或 RLHF。
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

### 首批 DPO 前向与批次身份探针（Slurm 13477、13478）

运行键 `20261008-qwen2-dpo-forward-probe-r4` 在 cscg-qh04 RTX 4090 上先运行独立原生公开 CLI，再运行 strict shim 公开 CLI。采集钩子在 `swift.rlhf_trainers.dpo_trainer.DPOTrainer.get_batch_loss_metrics` 记录训练 batch 的 `input_ids`、`attention_mask`、`labels`、DPO loss 与公开指标；随后在 `Accelerator.backward` 入口有意抛出探针哨兵，未执行反向或 optimizer step。Slurm 13477 的作业最终因比较器发现 batch 不同而退出 1，这是预期的不匹配结果，不是模型计算异常。两侧模型首个参数均报告 `cuda:0`；strict shim `use_cuda=1` 且 `fallback_count=0`。

Slurm 13478 使用同一模型 tokenizer 对已保存输入解码，确认原生首批是数据第 4 条（“What color is a clear daytime sky?”，chosen/rejected 为 Blue/Green），shim 首批是第 1 条（“What is 2 plus 2? Answer briefly.”，chosen/rejected 为 4/5）。两边 batch shape 分别为 `[2,30]` 与 `[2,33]`；偏好成对展开后的 `input_ids`、mask 与 labels 不同。模型文件和数据文件 SHA256 与上面的原生 oracle 相同，CLI seed/data_seed 都是 1234。native/shim loss 碰巧都为 `0.6931470633`，但 chosen/rejected logps 与 logits 指标不同；由于输入样本不同，这些数值不能用于模型前向正确性对拍。

| 首批指标 | 原生 | strict shim |
| --- | ---: | ---: |
| `logps/chosen` | `-21.33212471` | `-17.36635590` |
| `logps/rejected` | `-20.04442024` | `-14.73171616` |
| `logits/chosen` | `-2.30225992` | `-2.24321890` |
| `logits/rejected` | `-2.13749123` | `-2.27255225` |

观察与既有 `torch-randomsampler-generator` 报告的同 seed 跨 runtime sampler 顺序差异一致，当前 DPO 首批错位具体由 Trainer sampler 还是其他数据路径引起仍未单独定位。

同一探针的 `r1` 没有捕获指标，因为钩子挂在 TRL 基类，而 ms-swift 子类覆写该方法；`r2` 钩子签名遗漏了 ms-swift 的 `pair_loss_scale` 参数；`r3` 在 Jittor CUDA capability 查询子进程持续休眠、尚未进入模型前向时停止。上述尝试均未运行 backward。原始脚本和日志保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-dpo-forward-probe-r{1,2,3,4}/`。

因此 L1 更新为 **partial**：公开入口确实分别执行了 CUDA 前向并产生 DPO 输出，但首批输入不一致，缺少同输入 logits/log-prob/loss 对拍。L2 原有 backward 断言失败保持不变；本探针刻意没有重走该失败路径。后续需固定相同样本身份后再对拍前向，再按根因定位 backward，不以相同 seed 推断数据顺序相同。
