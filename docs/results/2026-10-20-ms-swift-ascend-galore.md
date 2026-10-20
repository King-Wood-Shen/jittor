# Ascend ms-swift GaLore 公开训练

- **状态：** 单卡真实 Ascend NPU 上 Swift GaLore AdamW 的公开 SFT 与优化器状态定向回归通过。
- **日期：** 2026-10-20
- **基线：** `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`（同步后的 `origin/2.0-refactor`）
- **实现提交：** `5178a34542f4778585911ef2492202f75237d0bb`
- **验证范围：** `swift.cli.main sft`，全参 BF16 Llama，`--optimizer galore --optim adamw_torch`，GaLore q/k/v/o 投影，6 步训练；外部 Torch optimizer defaults、逐参数自定义状态、`addcmul_`、scheduler 与 state_dict 恢复。
- **维护者：** ms-swift / Torch compatibility 维护者
- **复查条件：** Swift GaLore optimizer 实现、Torch optimizer state/defaults 契约或 Jittor ACL 运算路径变化时复查。

## 结论

Swift 的公开 SFT 入口在 Ascend NPU 上完成 6 步 GaLore 训练。回调确认优化器链为 `accelerate.optimizer.AcceleratedOptimizer` → `swift.optimizers.galore.adamw.AdamW`；backend 为 `acl`、设备为 `npu:0`、fallback policy 为 `error`，fallback 计数保持 `0→0`。loss 依次为 4.90625、4.90625、4.8125、4.765625、4.796875、4.859375，loss 与 grad norm 均有限。这个短 fixture 证明公开训练入口和优化器更新可执行，不衡量算法质量或性能。

定向 NPU 回归还覆盖外部 Optimizer 子类传入 defaults 字典、参数组覆盖、`state[param]` 自定义 moment 与 step、scheduler、state_dict/load_state_dict，以及 NPU 上的 `Tensor.addcmul_`。Slurm 5902 为 1 passed、6 deselected。公开训练 Slurm 5904 完成六步；首次兼容探索曾逐步暴露 defaults 初始化、外部 optimizer 状态归属和 `addcmul_` 缺口，修复后由定向回归与最终公开入口复验。

## 复现与证据

在维护者 Ascend Slurm 环境以 strict ACL fallback 运行。外部未版本化目录保留 source/patch receipt、输入哈希、Slurm 日志、设备采样和审计事件：

- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/galore-public-20261020-r1/`
- NPU 定向回归：作业 5902。
- Swift 公开 SFT：作业 5904。

源码 `git diff --check`、目标 Python 文件 `py_compile` 和 `bash tools/check_repo_layout.sh` 均通过。

## 验证边界

仅验证单卡 NPU、BF16、tiny Llama、6 步、指定 GaLore 投影模块和 AdamW 内优化器。未验证 Muon、其他目标层/算法、推理、双卡、吞吐或长程数值质量。未运行 CPU full、smoke 或 CPU 适配门禁。
