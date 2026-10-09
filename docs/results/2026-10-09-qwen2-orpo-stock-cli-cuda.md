# Qwen2-0.5B stock `swift rlhf --rlhf_type orpo` CUDA 输入身份审计

- 状态：原生与 strict shim 均完成公开 ORPO CLI 三步训练并保存 checkpoint；初始 290 项参数一致，strict shim 父子进程为 CUDA 且 fallback=0。第 1、2 步输入逐张量相同，第 3 步两条偏好记录的批次顺序反转，导致完整轨迹不满足同输入条件。L0 partial、L1/L2 blocked、L3 not-run、L4 入口已运行但功能未通过、L5 blocked。
- 日期：2026-10-09。
- 基线：Jittor `6bd0cf9bc15d29c17f6b18b929b55dfb4d4ea2fc`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，公开单卡 `swift rlhf --rlhf_type orpo`，两条 preference 样本、batch 2、SGD、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：ORPO trainer 接入尊重 `train_dataloader_shuffle` 的数据加载路径后，使用固定批次重新完成输入、前向、梯度、状态与恢复对拍；ms-swift trainer MRO 或 Torch shim sampler/random generator 变化时复查。

运行键 `20261009-qwen2-orpo-stock-cli-v1` 的数据 SHA256 为 `80ace7504a35976e65bda0f8bf7f651b6efbef7257b2f5858290451d9f58450f`，Qwen2 权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0。Slurm 13866 在 `cscg-qh04` RTX 4090（GPU UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`）完成 worker 预检；Slurm 13874 随后依次运行 native 与 strict shim CLI。两侧均训练三步并生成 `checkpoint-3`。候选父子进程均记录 `use_cuda=1`、`fallback=0`。

审计插件先核对全部 290 个初始参数名称、shape、dtype、trainable 标志和 SHA256；两侧逐项相同且参数在 CUDA。输入记录为 `input_ids`、`labels`、`attention_mask`，shape 均为 `[4,33]`。第 1、2 步三种张量逐元素相同。Slurm 13886 对已保存 NPZ 的 GPU worker 只读诊断确认第 3 步 `input_ids` 与 `labels` 的四行发生样本顺序交换：native 的行顺序为 A、B、A、B，shim 为 B、A、B、A；`attention_mask` 也随记录对应。批次内容相同而样本次序不同。

静态检查 ms-swift `ORPOTrainer` 继承 `RLHFTrainerMixin, SwiftMixin, HFORPOTrainer`，未混入单独的 `DataLoaderMixin`。`RLHFTrainerMixin.get_train_dataloader` 只转调父类，随后落到 Transformers Trainer 的 map-style 随机 sampler；因此 CLI 的 `--train_dataloader_shuffle false` 没有在该 ORPO 路径关闭随机采样。native 与 shim 的跨 runtime 随机 sampler 顺序差异与第 3 步观察一致。该归因限定于当前 ORPO MRO 和这组固定数据，不外推其他 Trainer。

首个比较器在第 3 步 `input_ids` 身份断言处停止。虽然两侧均保存了三步梯度及末态 checkpoint，比较器未校验后续步的输入身份，故这些记录不能作为同输入梯度/更新通过证据。训练日志 loss 轨迹接近也不替代批次身份。native 训练约 20.8 秒，shim 约 540.6 秒且包含首次 JIT；不作为 L5 性能结果。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 真实模型、公开 ORPO trainer 与 optimizer 已构造；290 项初始参数逐项相同且在 CUDA。完整模型/buffer、trainer 与 optimizer 状态映射未审计。 |
| L1 | blocked | 第 1、2 步输入相同且首步 forward 已保存；完整三步首个前向对拍未完成，第 3 步批次顺序不同。 |
| L2 | blocked | 三步梯度和末态 checkpoint 已采集，但第 3 步输入不同，不能验收完整固定轨迹梯度与更新。 |
| L3 | not-run | 未测试同进程/新进程恢复、optimizer/scheduler/RNG/dataloader 游标和续训轨迹。 |
| L4 | 入口已运行，功能未通过 | native 与 strict shim stock `swift rlhf --rlhf_type orpo` 均完成三步并保存 checkpoint；输入身份合同未通过。 |
| L5 | blocked | 前级未通过；shim 耗时含冷 JIT，未做预热后至少 10 次稳态性能及显存协议。 |

原始脚本、模型/数据摘要、NPZ、逐步梯度、checkpoint 和日志均未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-orpo-stock-cli-v1/`；输入顺序诊断保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-orpo-input-diagnose-v1/`。本报告仅适用于上述单卡 ORPO 配置，不代表其他偏好算法或整个 ms-swift。

文档门禁 Slurm 13894 在 RTX 4090 worker 上通过：`tools/check_repo_layout.sh` 成功，`JITTOR_TORCH_SHIM=1 pytest -q tests/structure` 为 1386 passed、6 skipped、1027 subtests passed。测试环境说明 2 个跳过项因解释器未安装 Jittor、2 个因未安装 pytest-xdist；其余跳过项由门禁分类记录。
