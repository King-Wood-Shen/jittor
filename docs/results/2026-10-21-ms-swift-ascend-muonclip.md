# Ascend ms-swift MuonClip 与 QK clipping 公开训练

- **状态：** 单卡真实 Ascend NPU 上 MuonClip 六步公开 BF16 SFT 通过；低阈值 QK clipping 触发并完成六步。
- **日期：** 2026-10-21
- **基线：** `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`（同步后的 `origin/2.0-refactor`）
- **源码：** `af3eef47fde83d29ee9e9f7fa449e883da044222`（本项未修改源码）
- **验证范围：** Swift 4.5.2 `swift.cli.main sft`，tiny Llama 全参 BF16，`--optimizer muonclip`，Muon Newton-Schulz 2 步；QK clipping 关闭和开启两种配置。
- **维护者：** ms-swift / Torch compatibility 维护者
- **复查条件：** Swift MuonClip 参数分组、QK tracker 或 Jittor ACL SDPA 路径变化时复查。

## 结论

Slurm 5908 在单卡 NPU 上完成 MuonClip 公开 SFT 六步，优化器链为 `accelerate.optimizer.AcceleratedOptimizer` → `swift.optimizers.muonclip.MuonClip`。loss 和 grad norm 均有限，`backend=acl`、`backend_fallback=error`、fallback 0→0。该配置使用 `newton_schulz_steps=2,qk_clip_enabled=false`。

Slurm 5915 在相同入口启用 `qk_clip_enabled=true,qk_clip_tau=1e-8`，并由 callback 记录 `_MaxLogitsTracker.consume()` 返回值。六步全部完成，所有 loss/grad norm 有限，fallback 0→0。记录的 logit 上界为约 `0.210478, 1.01457e-8, 2.40780e-7, 3.95480e-8, 1.19464e-7, 5.82658e-8`；每一步均超过阈值，因而触发 QK 权重限幅分支。普通和开启 clipping 的 loss 数值接近，不能据此推导优化算法效果。

尝试以 `qk_clip_tau=1` 验证默认尺度时，训练本身仍完成，但 tracker 上界未超过 1，审计按预设条件拒绝将该次记为 clipping 已触发。随后用显式低阈值与 tracker 记录完成了实际分支触发证据。

## 复现与证据

运行使用维护者的 Ascend Slurm 环境、BF16 与 strict ACL fallback。外部未版本化状态目录保存脚本、输入哈希、源提交回执、设备记录、日志与 JSON 审计：

- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/muonclip-public-20261020-r1/`（job 5908）
- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/muonclip-qkclip-public-20261021-r3/`（job 5915）

## 验证边界

仅覆盖指定 tiny Llama、单卡 BF16、六步和 2 次 Newton-Schulz 迭代。未验证外部 Moonlight Muon、推理、长程收敛、吞吐、其他 clipping 阈值或双卡。未运行 CPU full、smoke 或 CPU 适配门禁。
