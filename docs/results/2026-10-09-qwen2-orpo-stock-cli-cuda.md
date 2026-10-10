# Qwen2-0.5B stock `swift rlhf --rlhf_type orpo` CUDA 输入身份审计

- 状态：两条样本的三步公开 ORPO CLI 测试在 step 3 出现批次顺序反转；新的单样本受控三步测试输入、前向、全参数梯度及末态权重对齐。该受限场景 L0/L1/L2/L4 partial，L3 not-run、L5 blocked；ORPO 多样本路径仍受未生效的 `train_dataloader_shuffle=false` 限制，不能标为整体通过。
- 日期：2026-10-10 复核。
- 基线：Jittor `6bd0cf9bc15d29c17f6b18b929b55dfb4d4ea2fc`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，公开单卡 `swift rlhf --rlhf_type orpo`，两条 preference 样本、batch 2、SGD、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：ORPO trainer 接入尊重 `train_dataloader_shuffle` 的数据加载路径后，使用多条固定数据重新完成输入、前向、梯度、状态与恢复对拍；ms-swift trainer MRO 或 Torch shim sampler/random generator 变化时复查。

运行键 `20261009-qwen2-orpo-stock-cli-v1` 的数据 SHA256 为 `80ace7504a35976e65bda0f8bf7f651b6efbef7257b2f5858290451d9f58450f`，Qwen2 权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0。Slurm 13866 在 `cscg-qh04` RTX 4090（GPU UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`）完成 worker 预检；Slurm 13874 随后依次运行 native 与 strict shim CLI。两侧均训练三步并生成 `checkpoint-3`。候选父子进程均记录 `use_cuda=1`、`fallback=0`。

审计插件先核对全部 290 个初始参数名称、shape、dtype、trainable 标志和 SHA256；两侧逐项相同且参数在 CUDA。输入记录为 `input_ids`、`labels`、`attention_mask`，shape 均为 `[4,33]`。第 1、2 步三种张量逐元素相同。Slurm 13886 对已保存 NPZ 的 GPU worker 只读诊断确认第 3 步 `input_ids` 与 `labels` 的四行发生样本顺序交换：native 的行顺序为 A、B、A、B，shim 为 B、A、B、A；`attention_mask` 也随记录对应。批次内容相同而样本次序不同。

静态检查 ms-swift `ORPOTrainer` 继承 `RLHFTrainerMixin, SwiftMixin, HFORPOTrainer`，未混入单独的 `DataLoaderMixin`。`RLHFTrainerMixin.get_train_dataloader` 只转调父类，随后落到 Transformers Trainer 的 map-style 随机 sampler；因此 CLI 的 `--train_dataloader_shuffle false` 没有在该 ORPO 路径关闭随机采样。native 与 shim 的跨 runtime 随机 sampler 顺序差异与第 3 步观察一致。该归因限定于当前 ORPO MRO 和这组固定数据，不外推其他 Trainer。

### 单样本固定批次数值隔离

为独立检查 ORPO 算法数值并移除已证实的随机样本次序变量，运行键 `20261009-qwen2-orpo-fixedbatch-cli-v2` 使用原数据首条偏好记录、batch 1 和三步公开 CLI。预检 Slurm 13904、训练 Slurm 13905 均在 `cscg-qh04` RTX 4090（GPU UUID `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44`）完成；环境版本和模型权重同上，单行数据 SHA256 为 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10`。native 先于 strict shim，候选父子进程均为 `use_cuda=1`、`fallback=0`。

两侧三步的 `input_ids`、`labels`、`attention_mask` 逐张量完全相同；全部 290 个初始参数名称、shape、dtype、trainable 状态和值 SHA256 一致。首步 ORPO forward 的 chosen/rejected logits shape 均为 `[1,28,151936]`，最大绝对误差 `8.16137e-5`、相对 L2 分别 `2.54478e-6` 和 `2.53565e-6`；其余聚合输出的最大绝对误差不超过 `5.96e-6`。三步 loss 最大绝对差 `6.19888e-6`。

每步 290 项全参数梯度、494,032,768 个元素均被逐项比较：最大绝对差依次为 `3.22238e-7`、`3.09199e-7`、`3.76254e-7`，相对 L2 为 `4.89679e-6`、`4.90052e-6`、`5.36509e-6`。step-3 checkpoint 的 290 个权重键对齐，169 个张量发生更新，末态最大绝对差 `7.45058e-9`、相对 L2 `6.60016e-11`。该受限结果支持同输入下的前向、梯度和末态权重数值；当时尚未比较 optimizer state，L2 为 partial，L3 未运行。单样本结果不推断多样本 sampler 行为。

首个比较器在第 3 步 `input_ids` 身份断言处停止。虽然两侧均保存了三步梯度及末态 checkpoint，比较器未校验后续步的输入身份，故这些记录不能作为同输入梯度/更新通过证据。训练日志 loss 轨迹接近也不替代批次身份。native 训练约 20.8 秒，shim 约 540.6 秒且包含首次 JIT；不作为 L5 性能结果。


三步训练 checkpoint 的只读状态审计运行键 `20261010-qwen2-orpo-fixedbatch-l2-audit-v5` / Slurm 15279 在 cscg-qh04 RTX 4090 完成。审计通过 strict Jittor Torch shim `torch.load` 读取 native 与 shim 的 optimizer/scheduler 文件，父进程 CUDA=1、fallback=0。两侧 SGD `state` 均为空，2 个参数组分别含 169/121 项；两组所有选项完全相同（learning rate `1e-5`、momentum 0、weight decay 分别 0.1/0.0）。scheduler 的 `base_lrs`、`last_epoch=3`、`_step_count=4`、`_last_lr` 与其他共同字段相等；native 独有 `verbose=false`，shim 独有 `_is_initial=false`，故状态字典非完全相等。候选 optimizer checkpoint 为 Torch shim 的 portable pickle 格式，已由 shim loader 成功读回；该格式差异本身未导致恢复失败。直接从每步梯度按参数映射重构权重更新仍未完成，L2 保持 partial。审计脚本首三次分别因 NumPy bfloat16 读取限制、named-parameter 与 optimizer 参数顺序不同、native loader 不识别 shim pickle 格式而终止；这些都是审计器假设错误，并非训练故障。第四次因漏配 PYTHONPATH 在 import 前失败；不计兼容结果，所有日志与运行键均留存。

报告门禁运行键 `20261010-qwen2-orpo-fixedbatch-l2-docgate-v1` / Slurm 15280 在 cscg-qh04 完成。`bash tools/check_repo_layout.sh` 通过；以 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python` 运行 Torch-mode `tests/structure`，结果为 1384 passed、8 skipped、1019 subtests passed（516.55 秒）。4 个额外 skip 分别来自未安装 core wheel 与 pytest-xdist；门禁脚本注明另有 4 个声明性 skip。该文档门禁未执行模型验收。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 真实模型、公开 ORPO trainer 与 optimizer 已构造；290 项初始参数逐项相同且在 CUDA。完整模型/buffer、trainer 与 optimizer 状态映射未审计。 |
| L1 | partial | 两样本运行第 3 步输入不同；单样本受控运行三步输入相同并对齐首步 logits/loss，完整 hidden-state 和多样本路径仍未验。 |
| L2 | partial | 单样本受控运行三步覆盖全部梯度及末态权重；补充确认 optimizer 状态及参数组一致，但 scheduler 状态字段不同，且直接逐步更新重构未完成。两样本轨迹第 3 步输入不同。 |
| L3 | not-run | 未测试同进程/新进程恢复、optimizer/scheduler/RNG/dataloader 游标和续训轨迹。 |
| L4 | partial | 两种设置都通过 stock CLI 并保存 checkpoint；单样本受控场景数值对齐，多样本场景受 sampler 顺序影响，不能将公开 ORPO 面标为完整通过。 |
| L5 | blocked | 前级未通过；shim 耗时含冷 JIT，未做预热后至少 10 次稳态性能及显存协议。 |

原始脚本、模型/数据摘要、NPZ、逐步梯度、checkpoint 和日志未版本化，分别保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-orpo-stock-cli-v1/`、`20261009-qwen2-orpo-fixedbatch-cli-v2/` 和 `20261009-qwen2-orpo-input-diagnose-v1/`。本报告仅适用于上述单卡 ORPO 配置，不代表其他偏好算法或整个 ms-swift。

文档门禁 Slurm 13894 在 RTX 4090 worker 上通过：`tools/check_repo_layout.sh` 成功，`JITTOR_TORCH_SHIM=1 pytest -q tests/structure` 为 1386 passed、6 skipped、1027 subtests passed。单样本证据更新后的第二次文档门禁 Slurm 13922 同样通过：布局成功，结构测试 1386 passed、6 skipped、1027 subtests passed。两次测试日志均说明 2 个跳过项因解释器未安装 Jittor、2 个因未安装 pytest-xdist；其余跳过项由门禁分类记录。
