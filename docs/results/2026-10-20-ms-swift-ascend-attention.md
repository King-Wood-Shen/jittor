# Ascend ms-swift attention graph 与 packing SFT

- **状态：** 单卡真实 Ascend NPU 上 BSND/TND attention ACL 图正反向定向测试通过；公开 ms-swift BF16 SFT 在 packing 关闭和开启两种配置各 20 步通过。
- **日期：** 2026-10-20
- **基线：** `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`（同步后的 `origin/2.0-refactor`）
- **实现提交：** `f624a575f17f56c72d74b6d266c3566f5ee125a0`
- **验证范围：** `torch_npu.npu_fusion_attention` 的 BSND/TND bool mask ACL SDPA 图；ms-swift `swift.cli.main sft`、BF16、`--attn_impl flash_attn`、packing false/true、strict fallback。
- **维护者：** ms-swift / Torch compatibility 维护者
- **复查条件：** Transformers NPU attention 参数/掩码约定、ms-swift packing/TND 调用约定或 ACL SDPA 图路径变化时复查。

## 结论

之前公开 SFT 选择 `flash_attn` 时，Transformers NPU integration 调用了 facade 中明确拒绝的 `torch_npu.npu_fusion_attention`，训练因此无法开始。现在 facade 对 BSND 和 TND 提供 Jittor ACL `scaled_dot_product_attention` 图实现，并把 API 标记为 approximate。它不调用或声称调用 CANN 融合算子，也不返回融合算子的 workspace；不支持的融合参数会明确报错。

定向 NPU 测试在 NPU 0、BF16 输入上分别检查 BSND 与两个不等长 TND segment 的 causal bool mask 输出、形状、dtype 和有限梯度；Slurm 5863 为 1 passed、7 deselected。首次执行暴露 shim `torch.bool` 名称判别问题，修复后通过；另一次启动尝试由于 Slurm shim job 标记缺失而在测试前退出，更新运行脚本后复验。

两次 ms-swift 公开 SFT 分别以 `--packing false` 和 `--packing true --packing_length 128 --packing_strategy sequential --packing_num_proc 1` 运行 20 步，均选择 `flash_attention_2`，在 NPU 0 上完成。审计事件各记录 80 次 `npu_fusion_attention` 调用、fallback 计数 0→0、20 个有限 loss；首末 loss 都由 4.90625 降至 4.59375。峰值 Jittor 分配约 50.8 MB。短小 fixture 和 20 步运行只证明公开路径可执行，不是性能基准或算法质量评估。

## 复现与证据

训练和测试均通过 Slurm 在 Ascend 计算节点运行，`backend_fallback=error`。维护者外部状态目录保留完整脚本、输入哈希、source/patch receipt、日志、逐步 loss、fallback 与 attention 调用审计、HBM 采样：

- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/flash-attn-graph-20261020-r1/`
- NPU 定向测试：作业 5863；首次失败的 dtype 判别试次为 5851。
- packing 关闭 SFT：作业 5864；packing 开启 SFT：作业 5873。

## 验证边界

`bash tools/check_repo_layout.sh` 通过。没有运行 CPU full、smoke 或 CPU 适配门禁。该实现是通用 attention 图路径，不等同于 Ascend CANN fused attention，也不代表吞吐/内存优势。只验证了单卡、BF16、标准 causal bool mask 和这份小型训练 fixture；FP16、推理、其他优化/算法开关、双卡都不由此报告授级。TND 分段边界由 NPU 定向测试验证，packing-enabled public SFT 验证了 packing 开关下的 attention 路径可完成训练。
