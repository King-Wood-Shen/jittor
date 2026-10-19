# ms-swift Ascend 单卡 RLHF 算法训练对照

- 状态：DPO、KTO、CPO、纯 SimPO、ORPO 在限定三步场景通过；GRPO 的公开入口和设备执行通过，采样轨迹未达到数值对照准入。
- 日期：2026-10-19
- 代码基线：集成提交 `0b872e5ddf94304ebd9281f50a1cec874092a4dd`；修复提交 `7f396c615d017afde175d337399d516f28253005`。
- 下游来源：ms-swift 4.5.2；native 与 shim 环境中的 `swift/cli/rlhf.py` SHA256 均为 `f68c8b93ff4bfd08374fa2bca6efadcfe23e7beec13db162953275138fa56d65`。PEFT LoRA forward、PEFT model 与 TRL KTO trainer 源文件在两环境逐文件 SHA256 相同。
- 维护者：Torch 兼容层与 Ascend 训练维护者。
- 复查条件：Jittor Torch shim 的张量存储/延迟执行、ms-swift/TRL/PEFT 版本、ACL 算子或本表公开入口参数变化。

## 固定场景与结论

每个算法都通过 `swift.cli.rlhf` 公开入口，在 Ascend 910B3 单 NPU 上训练三步。native 控制使用 Torch 2.10.0+cpu、torch_npu 2.10.0；candidate 使用同一硬件上的 Jittor ACL shim，fallback 策略为 `error`，并在训练结束断言 `fallback_count=0`。夹具是 0.116M 参数 Llama，LoRA 可训练参数约 0.0174M；每次对照固定数据、模型文件和启动种子。模型 `model.safetensors` SHA256 为 `0c27276ae7a985335cf509008d6df93c4474faa78a2ef7c73326bd2090c386c6`。

| 算法 | native 与 candidate 结果 | 结论 |
| --- | --- | --- |
| DPO | 三步；batch 与初始权重一致；33 个 NPZ、601 个字段，最大绝对误差 `2.15e-7`；candidate fallback 0（作业 5511/5536，对照 5539） | 通过 |
| KTO | 三步；干净公开入口的 loss、grad norm、奖励与 log-prob 指标对齐；独立 trace 对照 39 个文件、616 个字段，最大绝对误差 `8.94e-8`；candidate fallback 0（干净作业 5615/5616；trace 作业 5611/5612，对照 5614） | 通过 |
| CPO | 三步；30 个文件、598 个字段，最大绝对误差 `8.94e-8`；candidate fallback 0（作业 5548/5559，对照 5560） | 通过 |
| SimPO | 显式 `cpo_alpha=0` 的纯 SimPO；native 与 candidate 无观察器运行三步，loss、梯度范数及指标对齐（作业 5583/5584）；candidate fallback 0 | 通过 |
| ORPO | 三步；30 个文件、598 个字段，最大绝对误差 `8.94e-8`；candidate fallback 0（作业 5564/5565，对照 5566） | 通过 |
| GRPO | 两端都完成三步，初始 LoRA 参数一致，candidate NPU forward 已观察且 fallback 0；采样后第一次 logits 最大绝对误差 `0.7639`（作业 5605/5606，对照 5617） | 部分通过；不声称数值等价 |

GRPO 使用 `num_generations=2` 和真实奖励插件。native 与 candidate 的生成随机流产生不同 completion，随后 rollout、奖励和优化轨迹分叉；这次结果证明完整训练路径执行，不证明相同采样结果下的 loss/更新等价。将 GRPO 提升为数值通过前，需建立两端可复现且相同的生成序列协议。

## 缺陷修复与诊断

DPO 首轮复现发现惰性整型张量执行 `add_` 时，compat 尝试读取尚未分配的存储。修复位于 `torch.Tensor` in-place storage 路径：`location() == "none"` 时不复制 detached storage，让原地操作正常分配；新增 CPU 结构回归覆盖 lazy integer tensor。该改动经 NPU 定向回归真实执行通过，随后 DPO 三步对照通过。

KTO 曾有带观察器的重跑把 candidate reference logits 记录为 policy logits。对该路径追加 PEFT 状态与 LoRA 层输出观测后，确认所有 14 个 LoRA 层的 disabled 状态都正确切换，禁用时被探测层输出与 base-layer 输出一致。带中间层物化的 retry4 数值对照通过；随后移除观察器，以同一公开入口重新运行 native 和 candidate，三步日志指标仍对齐。因此没有据观察器下的单次偏差修改算法源码。历史失败 trace 保存在外部实验目录，不能作为最终通过证据。

纯 SimPO 的初次观察器运行遇到进程堆损坏；无观察器复跑通过，故堆故障记录为观察器运行限制，不归因于算法本身。早期 GRPO 失败分别由缺少 chat template、观察器批输入假设和 callback 注册名错误造成；这些均在状态目录的 retry 日志中保留，最终可执行场景使用具备 chat template 的固定 Llama 夹具。

## 回归与限制

新增长期回归 `test_add_into_lazy_integer_tensor` 在 Slurm NPU 上通过：1 passed、13 deselected、0 skipped，并要求实际 NPU 执行。`bash tools/check_repo_layout.sh` 通过。Torch 模式完整 `tests/structure` 有 1498 passed、8 skipped、14 failed；失败归因于 ASV 可选依赖缺失、一个冷 CPU 子进程超时、ACL BMM float64 不支持，以及 pooling fused reindex/void 注册缺口，与本次算法和 lazy in-place 修复无关。原始日志、逐步 trace、数据、环境和设备证据保存在 `/home/xinshen/_state/ms-swift-ascend/runs/algorithms-world1-0b872e5-20261019/`，没有纳入仓库。
