# ms-swift Ascend 单卡 RLHF 算法训练对照

- 状态：DPO、KTO、CPO、纯 SimPO、ORPO 在限定三步场景通过；GRPO 的公开入口和设备执行通过，采样轨迹未达到数值对照准入；GKD 本地 teacher 尝试在首步出现零 loss/零梯度，未通过。
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

GRPO 使用 `num_generations=2` 和真实奖励插件。两端生成的 completion 与采样轨迹未对齐，后续 rollout、奖励和优化轨迹分叉；但保存的首个差异已经出现在 `step0` 第一次 forward 的 logits（见续接审计），且该 observer 没有保存这次 forward 的输入，所以当前证据不能把首个分歧归因于随机流。尝试向公开入口传递 TRL `generation_kwargs` 来控制 beam sampling 时，ms-swift 参数解析拒绝该字段；没有修改安装包或外部源码。将 GRPO 提升为数值通过前，需先用同输入、同初始权重定位并消除首次 logits 分歧，再建立两端可复现且相同的生成序列协议。

## 缺陷修复与诊断

DPO 首轮复现发现惰性整型张量执行 `add_` 时，compat 尝试读取尚未分配的存储。修复位于 `torch.Tensor` in-place storage 路径：`location() == "none"` 时不复制 detached storage，让原地操作正常分配；新增 CPU 结构回归覆盖 lazy integer tensor。该改动经 NPU 定向回归真实执行通过，随后 DPO 三步对照通过。

KTO 曾有带观察器的重跑把 candidate reference logits 记录为 policy logits。对该路径追加 PEFT 状态与 LoRA 层输出观测后，确认所有 14 个 LoRA 层的 disabled 状态都正确切换，禁用时被探测层输出与 base-layer 输出一致。带中间层物化的 retry4 数值对照通过；随后移除观察器，以同一公开入口重新运行 native 和 candidate，三步日志指标仍对齐。因此没有据观察器下的单次偏差修改算法源码。历史失败 trace 保存在外部实验目录，不能作为最终通过证据。

纯 SimPO 的初次观察器运行遇到进程堆损坏；无观察器复跑通过，故堆故障记录为观察器运行限制，不归因于算法本身。早期 GRPO 失败分别由缺少 chat template、观察器批输入假设和 callback 注册名错误造成；这些均在状态目录的 retry 日志中保留，最终可执行场景使用具备 chat template 的固定 Llama 夹具。

## 回归与限制

新增长期回归 `test_add_into_lazy_integer_tensor` 在 Slurm NPU 上通过：1 passed、13 deselected、0 skipped，并要求实际 NPU 执行。`bash tools/check_repo_layout.sh` 通过。Torch 模式完整 `tests/structure` 有 1498 passed、8 skipped、14 failed；失败归因于 ASV 可选依赖缺失、一个冷 CPU 子进程超时、ACL BMM float64 不支持，以及 pooling fused reindex/void 注册缺口，与本次算法和 lazy in-place 修复无关。原始日志、逐步 trace、数据、环境和设备证据保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/algorithms-world1-0b872e5-20261019/`，没有纳入仓库。

文档索引与报告链接已用标准库检查；`test_docs_structure.py` 的追加定向运行未能收集，因为登录环境和隔离 oracle 环境都没有安装 pytest。没有为了单条文档测试改动共享环境。

## 基本训练算法矩阵与证据边界

本矩阵合并本报告、已封存的 Swift FULL SFT 报告与历史 Swift LoRA 报告；每格只代表表中所列的受限场景。`NPU 训练通过`要求公开训练入口、有限且整体可解释的 loss 趋势、零意外 fallback 和设备峰值显存证据，单纯完成三步不够。

| 算法 | 当前证据 | NPU 结论 |
| --- | --- | --- |
| FULL SFT | 注册 `SwiftSft.main()`/`Seq2SeqTrainer` 三步在 CPU、CUDA 通过；另有两 rank FULL SFT 新进程续训作业在排队 | 未闭合；不把 CPU/CUDA 或排队作业算作 NPU 通过 |
| LoRA | ms-swift 自带 tuner 的 CPU/CUDA 三步 oracle；本报告 RLHF 场景用固定 LoRA 夹具 | 未验证独立公开 SFT CLI 的 NPU 收敛与峰值显存 |
| DPO | 本报告中公开 RLHF CLI 三步数值对照通过 | 既有通过；缺本轮 loss 趋势和设备峰值显存归档 |
| KTO | 本报告中公开 RLHF CLI 三步数值对照通过 | 既有通过；缺本轮 loss 趋势和设备峰值显存归档 |
| CPO | 本报告中公开 RLHF CLI 三步数值对照通过 | 既有通过；缺本轮 loss 趋势和设备峰值显存归档 |
| SimPO | `cpo_alpha=0` 的纯 SimPO 公开 RLHF CLI 三步通过 | 既有通过；缺本轮 loss 趋势和设备峰值显存归档 |
| ORPO | 本报告中公开 RLHF CLI 三步数值对照通过 | 既有通过；缺本轮 loss 趋势和设备峰值显存归档 |
| GRPO | 公开 RLHF CLI 在两端完成三步，candidate NPU forward 及零 fallback 已证实，但轨迹未对齐 | 部分通过；不作为数值通过 |
| GKD | 2026-10-20 本地 teacher 的公开 CLI 在 NPU 实际训练一步，记录 loss `0.0`、grad norm `0.0` 后停止 | 阻塞；未证明有效梯度或收敛，详见本报告续接 |

现有 FULL SFT CPU/CUDA 证据见 [`2026-10-15-swift-trainer-installed-source.md`](2026-10-15-swift-trainer-installed-source.md)；Swift LoRA CPU/CUDA 手动三步证据见 [`2026-10-14-downstream-oracle-scorecards.md`](2026-10-14-downstream-oracle-scorecards.md)。这些报告没有提供本轮所需的真实 NPU loss 趋势与设备峰值显存，因此不重复宣称满足该项验收。RL 目标按有限的奖励/损失/梯度和可解释的整体趋势判断，不要求逐步单调。

## 2026-10-20 续接审计

- 同步基线：官方 `origin/2.0-refactor`，`d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`；工作树 `fix/msw-ascend-algorithms-sync` 的 HEAD 为 `1d9f66d9b`，以该同步基线为共同祖先，领先 21 个提交，开工时工作树干净。
- 旧 fork 分支仍保留 `d17f8e024f65103467cf86d06e60803b52d12eac`；不要改写或非快进推送该共享分支。`git cherry` 未找到这 21 个提交的 patch-id 等价项。官方主线已另行摘取 #18036/#18039 的 `torch.npu` 设备发现/选择及 RMSNorm、Rotary、SwiGLU 原语，见上游提交 `4233b3f97`；它不包含当前分支新增的 RNG 状态、lazy in-place、ACL 算子语义等其他工作，需按文件/语义分别审阅，不能把整组旧提交标为已合入。
- 当前 Slurm 查询显示作业 `5592`、`5590`、`5553` 均为 `sft-fsdp2` 且 `PENDING`（`DependencyNeverSatisfied`）；没有取消或重复提交同类实验。
- GRPO 续查确认保存的最差字段是 `forward-values-step0-call0.npz:model_output.logits`，绝对差 `0.7639221847`，形状 `(2, 5, 128)`；candidate 观测到 NPU forward、fallback 为 0。现有 observer 没有保存该次 forward 的输入，因此无法排除首批输入/上下文已不同，也不能将最先分歧归因于采样随机流。记录见未版本化目录 `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/algorithms-world1-0b872e5-20261019/compare-grpo-retry4.json`；需要先补齐同输入、同初始权重的 NPU forward 根因复现，再决定是否属于 JTorch/ACL 缺陷。

- 文档验证（2026-10-20）：`bash tools/check_repo_layout.sh` 通过（298 个活动 Markdown 文件）；`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python3 -m pytest -q tests/structure` 未能启动，系统 Python 3 无 pytest（`No module named pytest`）。本次仅改文档，未运行 core 或真实设备作业。


## 2026-10-20 GKD 本地 teacher 补验

- 状态：未通过，按五次不同尝试上限停止并转向 reward-model 路径；不把 CLI 完成一步计作训练通过。
- 同步基线：官方 `origin/2.0-refactor` SHA `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`；任务源码 SHA `ac97bef528678a7a04e4c1aa71108abcbc26c18b`。
- 入口与依赖：ms-swift `4.5.2` 的公开 `swift/cli/rlhf.py --rlhf_type gkd`，TRL `0.29.1`、Transformers `4.57.6`、PEFT `0.17.1`、Jittor `1.3.11.0`，Python `3.11`。native/shim 安装中 `swift/cli/rlhf.py` SHA256 均为 `f68c8b93ff4bfd08374fa2bca6efadcfe23e7beec13db162953275138fa56d65`；GKD trainer、loss、TRL GKD trainer SHA256 分别为 `3ed60c38920ccce6a86529504fc2d7d02def1dd2e22ee9c33c85fe62d0e020a9`、`5d02ce7d1dfd162cb37dba054eff1e23e29d41dd1f2122352a8651f6e7ca0fec`、`888437a4c1f166faef6c2719542ebaea48251bddacd891393936cbd2af3796f9`。
- 场景：Ascend 910B3 单 NPU；0.0986M 可训练参数的 Llama full-tuner；本地 frozen teacher；12 步上限、batch 1、4 个短 prompt、completion 长度 4。teacher `model.safetensors` SHA256 为 `0c27276ae7a985335cf509008d6df93c4474faa78a2ef7c73326bd2090c386c6`；student 仅将 `lm_head.weight` 乘以 `0.9`，SHA256 为 `a1bf309aee5533d4a35b0e9712247baecfc5738d04415994cc5402c26df9d20e`；数据 SHA256 为 `1f76963ab799e90108a7250312a80244613ed04bae22cf79642f7af2134b7ce0`。Checkpoint 中的自定义 adapter 键被标准 `LlamaForCausalLM` 加载器忽略；实际训练模型报告 `0.0986M` 参数、100% 可训练。
- 尝试记录：作业 `5823` 因 host wrapper job-marker 不在 compute node 可见路径退出；`5825` 因缺 `CCACHE_BASEDIR` 在 Jittor 导入时退出；`5826` 因 Swift 4.5.2 不识别 `--train_type full` 而停在 CLI 解析；`5832` 因长 `TMPDIR` 令 datasets manager 的 AF_UNIX 路径超限而退出；`5835` 修复上述启动问题后进入 NPU 训练，但首步即报告 loss `0.0`、grad norm `0.0`，随后取消，避免继续执行无更新步骤。以上日志与独立缓存未版本化保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/gkd-world1-0b872e5-20261020-r1/`。
- 设备证据：作业 `5835` 使用 `backend_fallback=error`，执行到一步未触发 fallback 异常；因提前停止，没有最终 `fallback_count` 断言，故不登记为零 fallback 通过。作业期间以 `npu-smi info -t proc-mem` 每秒采样实际设备进程内存，峰值 `192 MB`；Swift 的 `memory(GiB)` 字段未用于峰值结论。
- 根因边界：Swift GKD trainer 在 `num_valid == 0` 时明确将 loss 置零，但本次没有保存 student/teacher logits 和有效标签数，故无法确认该分支是否触发，也不能排除其他零梯度原因。修复前需添加能保留有效响应 token 数和两侧 logits 形状的 NPU 诊断；本轮到达尝试上限后转向 RM，不再给 GKD 数值通过结论。
