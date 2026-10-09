# Qwen2-0.5B RewardTrainer Python API CUDA 对拍

- 状态：固定初始化与六组预 tokenized preference 数据下，构造状态、三步前向、trainable head 梯度和更新通过数值对拍；L0 通过，L1/L2/L4 partial，L3 未运行，L5 blocked。该结果不代表公开 CLI、预训练 reward head 或整体 ms-swift reward modeling 兼容。
- 日期：2026-10-09。
- 基线：Jittor `2b30a5a2b68bb192918c56aef36d1f23e1ce0dd5`，上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：`swift.rlhf_trainers.RewardTrainer` Python API、Qwen2-0.5B sequence-classification backbone、新建单值 `score.weight`、只训练该 head、单张 RTX 4090、FP32/eager、三步 SGD。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补 optimizer 状态与 checkpoint 恢复、输出原始 dtype 断言和 stock `swift rlhf --rlhf_type rm` CLI；更换 ms-swift trainer、Qwen2 CUDA 执行或 RewardTrainer 损失实现时复验。

Slurm 13696 在 `cscg-qh04` RTX 4090（UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`，驱动 `580.178.04`）按先原生、后 shim 的顺序运行。环境为 Python 3.11.15、原生 PyTorch `2.6.0+cu124`、shim Jittor `2.0.0`、Transformers `4.57.6`、ms-swift 固定 SHA 如上。Qwen2-0.5B 权重文件 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。候选在独立 `JITTOR_HOME` 冷编译，运行使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`，最终 `fallback_count=0`。

两侧各构造 291 个 state-dict 项；名称、shape、dtype、设备和参数 `requires_grad` 元数据完全相同，所有模型参数均在 CUDA。唯一 trainable 参数为 `score.weight`，两侧都用同一个确定性公式初始化。数据由六个固定 preference pair 构成：第 i 对 chosen token IDs 为 `[1+i,2,3,4,5,6,7,8]`，rejected 为 `[9+i,10,11,12,13,14,15,16]`，attention mask 全一；batch 2、无梯度累积、seed/data_seed 42、`max_length=32`、SGD 学习率 `1e-5`、无 AMP、无梯度裁剪、三步。为保证跨 runtime 的逐步输入相同，采样器在测试子类中固定为顺序采样；loss 包装器只采集 Trainer 实际 loss。

GPU worker 上比较 18 项逐步张量：每步输入 IDs 与 mask 完全相同；logits 最大绝对差 `4.11272e-6`、最大相对 L2 `6.09540e-6`；loss 最大绝对差 `2.50340e-6`；`score.weight` 梯度最大绝对差 `7.24793e-5`、最大相对 L2 `2.92147e-6`；更新后 head 最大绝对差 `2.79397e-9`。独立公式复核确认两侧每步 loss 均等于 `mean(-logsigmoid(chosen_score - rejected_score))`，最大绝对误差分别为原生 `7.10163e-8`、shim `4.53779e-8`。输入为离散 token IDs，不适用输入梯度。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | pass | 真实 Qwen2 模型、processor/template、RewardTrainer、数据和 SGD 已构造；291 个 state key/shape/dtype/device、参数名及 trainable 标记逐项一致，参数全在 CUDA。相同 checkpoint 文件 SHA 与确定性 head 初始化保证初态来源一致。 |
| L1 | partial | 三步输入完全相同；logits/loss 形状、有限值与数值已比较并通过容差，loss 公式独立复核。采集器在存盘前统一转为 FP32，没有保留原始输出 dtype 元数据。 |
| L2 | partial | 唯一 trainable 参数的三步梯度与更新已比较；无动量 SGD 的 optimizer state 未显式导出对照，且 L1 dtype 记录未完整。整数 token IDs 不适用输入梯度。 |
| L3 | not-run | 没有同进程或新进程恢复与后续轨迹。 |
| L4 | partial | 实际调用 ms-swift 导出的 `RewardTrainer` Python 类并完成三步；使用测试子类固定顺序采样及采集 loss，没有验证 stock RLHF CLI。 |
| L5 | blocked | 前序层未完整通过；冷 JIT 包含在训练日志计时中，未运行稳态性能协议。 |

仓库门禁：`bash tools/check_repo_layout.sh` 和 manifest 生成/校验通过。Slurm 13720 的 `tests/structure` 为 1383 passed、8 skipped、1 failed；唯一失败是 CPU matmul 测试需要构建 oneDNN v3。加载 `cmake/4.3.2` 后在 Slurm 13738 完整复跑仍失败，因为 worker 下载 `https://codeload.github.com/uxlfoundation/oneDNN/tar.gz/refs/tags/v3.9.1` 时连接被拒绝。该结构门禁未通过，未把部分通过计为通过。

运行键 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-rewardtrainer-api-2b30-v1/` 保存脚本、日志、状态元数据、张量与比较 JSON，均未版本化。候选冷 JIT 与训练总计约 10 分 51 秒；该时间不作为性能指标。本报告仅覆盖上述固定 API 配置。
