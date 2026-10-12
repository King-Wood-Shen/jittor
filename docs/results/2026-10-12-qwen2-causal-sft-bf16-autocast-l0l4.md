# Qwen2-0.5B 公开 SFT BF16 autocast 新收敛协议 CUDA 复验

- 状态：已修复 FP32 权重/输入请求 BF16 autocast 实际走 FP16 的路由缺陷；指定公开 SFT profile 的 L0–L4 通过，L5 未运行。
- 基线：Jittor `b599924f1787a2c799b71df1243e1043ddf22fc2`；修复提交 `f62688c53`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：扩展到冻结矩阵中的其他模型/任务、AMP dtype、训练配置或 L5 性能面时，使用新运行键和对应设备合同。

## 验证范围

运行键 `20261012-qwen2-causal-sft-bf16-autocast-resume-anchor-v2` 使用 Qwen2-0.5B、FP32 全参数权重、公开 `swift sft` CLI、`--bf16 true`，在一张 RTX 4090、CUDA 12.2、Python 3.11.15 上执行。模型 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，数据 SHA256 为 `c6d57b79740a2cf80333c265359b617e119c46e42a50a812386d1f9e2701afc2`。Strict shim 运行设置 CUDA scope、`backend_fallback="error"` 和 `forbid_backend_fallbacks()`；记录的 fallback 增量为零。原生 24 步 checkpoint 来自 Slurm 18745，仅只读复用，没有重训。

最终只读分析 Slurm 18820 写出的 `validation.json` 共 96 个具名检查，`overall_pass=true`、`failures=[]`；其 SHA256 为 `0ea6ea77537b65e69fbdf8d32205fc2ebff7bc5d29139dbfb9807bce557cd069`。严格 CUDA dtype trace 显示 FP32 输入与 FP32 Linear 权重在 BF16 autocast 下产生位于 `cuda:0` 的 BF16 输出；shim `amp_reg=66`，BF16 偏好位已启用。

## 冻结层结果

| 层 | 状态 | 证据 |
| --- | --- | --- |
| L0 | pass | 原生与 strict shim 通过公开 CLI 构造 Qwen2-0.5B 全参数 SFT；参数 schema、dtype 与 CUDA 放置已审计。 |
| L1 | pass | 同 profile 的 CUDA 前向有限；FP32 Linear 输入/权重的 BF16 autocast 路由已直接观测为 BF16 CUDA 输出。跨实现 loss/logit 差值只作诊断。 |
| L2 | pass | 两侧分别按预锁定窗口 `[1..4]` 与 `[21..24]` 验收：原生均值 `7.040389657` → `0.000019458`，shim 均值 `6.239583462` → `0`。两侧全部 290 个 trainable 参数梯度均存在且有限，训练期间记录到非空真实更新。跨实现 loss/gradient 数值差异不构成该层失败。 |
| L3 | pass | 两侧从 checkpoint-24 在同进程与新进程继续到 global step 36；模型、optimizer、scheduler、CPU/CUDA/NumPy/Python RNG 与 dataloader 游标均恢复，后续训练继续。 |
| L4 | pass | 使用公开 `swift sft` CLI 端到端完成训练与恢复并保留 checkpoint。shim fresh/resume 的严格 CUDA fallback 增量分别为 0。 |
| L5 | not-run | 未执行预热后至少 10 次稳态性能测量。 |

训练与恢复作业：native anchor 18745（只读输入）、native resume 18757、strict shim train/resume 18791；只读分析 18820。前置严格 CUDA dtype 探针/兼容回归分别在 18650/18654 完成。代码提交前门禁 Slurm 18897：布局通过；结构测试 1384 passed、8 skipped、1019 subtests；core 白名单 native 98 passed / 41 skipped / 1 xfailed，torch 225 passed / 1 skipped；gate exit 0。结构门禁复用了与 `compat`、`tests/structure` 源码树 SHA 相同的持久 CPU 缓存，无重复 core 编译。core 白名单使用独立 native/torch 缓存，完成了首次冷编译，随后两种模式的核心扩展哈希一致。

## 问题结论与边界

1. **已解决：** 新增独立 BF16 偏好位，贯通 AMP 注册、核心 dtype 推断、矩阵运算、数组输出和 Torch autocast；先前 all-FP32 输入走 FP16 的缺陷已在独立分支修复，并由 CUDA 探针及公开入口验证。
2. **已验证（仅本 profile）：** 公开 ms-swift 入口在此 Qwen2-0.5B 全参数 SFT 配置的 L0–L3 不需要再跑新运行键；该运行键也通过 L4。其他冻结矩阵格子不因此自动通过。
3. **独立诊断：** Slurm 18660 通用半精度测试中的 25% 跨实现 FP32 数值阈值仍是该通用测试自身的合同/诊断结果；按当前 Skill，它不单独构成 ms-swift L2 失败。保留该结果，不放宽阈值，也不据它推断 BF16 dtype 路由根因。

2026-10-08 的三步报告仍保留原始旧状态与严格对拍数据，见 [历史报告](2026-10-08-qwen2-causal-sft-bf16-amp-cuda.md)。原始脚本、日志、checkpoint、模型及验证 JSON 保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-causal-sft-bf16-autocast-resume-anchor-v2/`，代码门禁记录保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-bf16-autocast-code-gates-v1/`。
