# Qwen2-0.5B stock `swift rlhf --rlhf_type orpo` CUDA 输入身份审计

- 状态：修复前两条样本的三步公开 ORPO CLI 在 step 3 出现批次顺序反转；2026-10-12 已修正 generator-backed CPU `randperm` 并在 strict CUDA loader-only CLI 中复核三轮输入身份与冻结 native 输入一致。后续两行12步 native 与 strict-shim control 输入身份、梯度/更新和各自loss窗口均通过；L3 strict-shim恢复仍未成套验收，L4受前置恢复层级限制。恢复 smoke 中，两类完整 RNG 观察 hook 分别单独运行及仅含轻量日志的双 wrapper 组合都通过；但原失败运行还同时启用了 loss、batch、gradient、model 与 optimizer 观察器，因此目前只将 NanoVector 故障限定为复杂观察器栈相关，不能归因于 RNG 比较或某一具体 hook。单固定样本12步 profile 的 L0–L4 通过（L3 恢复于2026-10-11补验）；原生 L5 完成十次稳态测量，strict shim 的测量轨迹在 step 16 backward 失败，故 L5 未通过。不能将 ORPO 或整个 ms-swift 标为整体通过。
- 日期：2026-10-12 复核。
- 基线：恢复补验使用 Jittor `049bcc8548a63dfad654a605564e6f1949a1bbda`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。原12步训练证据仍绑定其报告中记录的历史基线。
- 范围：Qwen2-0.5B causal LM，全参数 FP32/eager，公开单卡 `swift rlhf --rlhf_type orpo`，两条 preference 样本、batch 2、SGD、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：ms-swift trainer MRO、Accelerate `SeedableRandomSampler` 或 Torch shim 的 generator-backed `randperm` 语义变化时复查；在多样本 ORPO 的 L1输出结构、完整 strict-shim L3恢复和后续层级验收前，不提升其完整 L0–L4。

运行键 `20261009-qwen2-orpo-stock-cli-v1` 的数据 SHA256 为 `80ace7504a35976e65bda0f8bf7f651b6efbef7257b2f5858290451d9f58450f`，Qwen2 权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0。Slurm 13866 在 `cscg-qh04` RTX 4090（GPU UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`）完成 worker 预检；Slurm 13874 随后依次运行 native 与 strict shim CLI。两侧均训练三步并生成 `checkpoint-3`。候选父子进程均记录 `use_cuda=1`、`fallback=0`。

审计插件先核对全部 290 个初始参数名称、shape、dtype、trainable 标志和 SHA256；两侧逐项相同且参数在 CUDA。输入记录为 `input_ids`、`labels`、`attention_mask`，shape 均为 `[4,33]`。第 1、2 步三种张量逐元素相同。Slurm 13886 对已保存 NPZ 的 GPU worker 只读诊断确认第 3 步 `input_ids` 与 `labels` 的四行发生样本顺序交换：native 的行顺序为 A、B、A、B，shim 为 B、A、B、A；`attention_mask` 也随记录对应。批次内容相同而样本次序不同。

### Sampler 运行时因果复核（2026-10-12）

2026-10-11 对源码 MRO 的静态推断有误，现按运行时证据更正；step-3 输入反序本身仍有效。ms-swift `88d727951203256baa564c643c651b6f8d90fd7e` 的 `ORPOTrainer` MRO 为 `RLHFTrainerMixin, SwiftMixin, HFORPOTrainer`，且该类未混入 `DataLoaderMixin`。运行时 `RLHFTrainerMixin.get_train_dataloader` 经 `super()` 到达 TRL/Transformers trainer；strict shim 实际构造 `accelerate.data_loader.DataLoaderShard`，其中 batch sampler 是 compat `_BatchSampler`，其 sampler 为 Accelerate `SeedableRandomSampler`。尽管已解析配置记录 `dataset_shuffle=false`、`train_dataloader_shuffle=false`、`group_by_length=false`，该 ORPO loader 仍走该随机 sampler 路径。

strict shim loader-only 运行键 `20261012-qwen2-orpo-multisample-loader-probe-v8` / job17212 在 cscg-qh04 RTX 4090（GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）捕获三轮 sampler 索引 `[1,0]`、`[1,0]`、`[0,1]`。三轮 CUDA loader 输入分别与旧 shim `shim-input-step1..3.npz` 逐张量相等，运行时 `use_cuda=1`、fallback=0；没有模型前向、优化器更新或 checkpoint 写入。`loader-diagnostic.json` 内嵌的 run_key 误写为 v7；实际执行键由 v8 的 worker 日志、脚本与目录确认，报告以 Slurm job 和产物目录为身份依据。

独立的 native sampler-only 复核键 `20261012-qwen2-orpo-seedable-sampler-native-probe-v2` / job17215 使用 PyTorch 2.6.0+cu124 与同版本 Accelerate `SeedableRandomSampler(data_seed=1234)`，不导入 Jittor、不加载模型或重跑训练。三轮索引均为 `[1,0]`，并逐轮映射到未修改的历史 native `native-input-step1..3.npz`。原生与 shim 历史输入文件在该只读作业前后 SHA 不变。因此 step-3 差异发生在 seeded sampler 产生的索引处，和旧训练观察的第 1、2 步一致、第 3 步 A/B 交换相符。

以上段落记录的是修复前的根因判断与证据；下节修正了 generator 序列问题，但不撤销当时的输入观察或把 loader 复核升级为训练结论。原始脚本与证据保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-orpo-multisample-loader-probe-v8/` 和 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-orpo-seedable-sampler-native-probe-v2/`。

### Generator-backed CPU `randperm` 与多样本 loader 复核（2026-10-12）

compat `torch.Generator` 的新建 MT19937 stream 现按 Torch 2.6 CPU `randperm` 的 word 消耗和排列方式实现：小尺寸分支使用 uint32 modulo 的 forward Fisher–Yates，大尺寸分支使用两词高位优先组成 64-bit 值的 inside-out shuffle。旧版本 checkpoint 中标记为 PCG64 的生成器状态仍走原来的排列与恢复路径。修复位于 `jittor.compat.torch`；没有改 ms-swift 源码或加入下游 adapter。独立 PyTorch 2.6.0+cu124 sampler oracle job17215 未重跑。

strict CUDA 定向测试运行键 `20261012-torch-generator-randperm-mt19937-v3` / job17273 在 cscg-qh04 RTX 4090（GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）执行；Jittor HEAD 为 `d233ea654c00b3143861004351e522bbd2ba9e12`，测试期间使用已验证的串行 CUDA Jittor 缓存。`test_generator_streams.py`、`test_torch_sampler_rng.py` 与 `test_default_generator_snapshot.py` 共 20 passed、5 subtests passed；strict worker 记录 `use_cuda=1`、fallback 从 0 到 0。大尺寸分支另以受控 MT 原始词序测试，覆盖两词合成次序。较早的有效定向运行 job17267 为19 tests、5 subtests通过；无效 harness 尝试保留在各自 state manifest，不作为测试结果。

公开入口的 loader-only 复核运行键 `20261012-qwen2-orpo-multisample-loader-fidelity-v3` / job17272，在 cscg-qh04 RTX 4090（UUID `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44`）对当前兼容补丁运行固定 Qwen2-0.5B ORPO CLI 配置。独立 native sampler oracle job17215 与只读保存的原生 `input_ids`、`attention_mask`、`labels` 未重跑或改写。strict shim 三个 epoch 都采得索引 `[1,0]`，每轮 loader 输入与冻结 native 对应步骤逐张量相同；worker/CLI 报告 CUDA 可用、`use_cuda=1`、fallback=0。该诊断构造了模型和公开 CLI Trainer，但拦截了 `Trainer.train`，没有执行模型前向、反向、optimizer 更新或 checkpoint 保存；因此只证明该固定数据/seed/loader 路径的 sampler 与输入身份，不通过多样本 ORPO 的 L1–L4。

源码修复已在提交 `d233ea654c00b3143861004351e522bbd2ba9e12` 之上验证；测试时暂存源码树为 `ddd28eaa1218770047da66298fde3b3b02d10270`。job17276 的核心门禁中 Torch-mode compat session 为226 passed、246 subtests passed；另一 native CUDA session 127 passed、10 skipped、2 xfailed，但一个无关的 `tests/ops/test_binary_op.py::TestBinaryOpCudaFp16::test_mod_negtive` 因一项 FP16 modulo 差值超其 `1e-5` 容差而失败（`-0.11669921875` 对 `-0.116943359375`）。故 `--tier core --backend cuda` 整体状态为失败，不能报告全 core 门禁通过。布局门禁通过；Torch-mode `tests/structure` 为1384 passed、8 skipped、1019 subtests passed。job17276 的自定义 gate wrapper 又把空缓存首次构建后 core `.so` 改变误判为结构测试失败；测试本身通过，no-build bootstrap 1.5秒通过。正式门禁 job17304 移除额外执行阈值后通过：布局与 Torch-mode tests/structure 的 1384 passed、8 skipped、1019 subtests 均符合仓库门禁；复用同一专用 CPU 缓存时，前后 JITTOR_NO_BUILD=1 检查通过、核心库 SHA 不变且无 core 冷编译。该门禁 has_cuda=0，仅证明结构测试缓存复用，不证明 CUDA 计算。运行键、日志和原始工件均未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-torch-generator-randperm-mt19937-v1/`、`20261012-torch-generator-randperm-mt19937-v2/`、`20261012-torch-generator-randperm-mt19937-v3/`、`20261012-qwen2-orpo-multisample-loader-fidelity-v1/`、`20261012-qwen2-orpo-multisample-loader-fidelity-v2/`、`20261012-qwen2-orpo-multisample-loader-fidelity-v3/` 与 `20261012-randperm-fix-core-structure-gates/`。

### 单样本固定批次：旧三步协议

为独立检查 ORPO 算法数值并移除已证实的随机样本次序变量，运行键 `20261009-qwen2-orpo-fixedbatch-cli-v2` 使用原数据首条偏好记录、batch 1 和三步公开 CLI。预检 Slurm 13904、训练 Slurm 13905 均在 `cscg-qh04` RTX 4090（GPU UUID `GPU-381c130e-e915-d4b7-0a6f-dce556f02e44`）完成；环境版本和模型权重同上，单行数据 SHA256 为 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10`。native 先于 strict shim，候选父子进程均为 `use_cuda=1`、`fallback=0`。

两侧三步的 `input_ids`、`labels`、`attention_mask` 逐张量完全相同；全部 290 个初始参数名称、shape、dtype、trainable 状态和值 SHA256 一致。首步 ORPO forward 的 chosen/rejected logits shape 均为 `[1,28,151936]`，最大绝对误差 `8.16137e-5`、相对 L2 分别 `2.54478e-6` 和 `2.53565e-6`；其余聚合输出的最大绝对误差不超过 `5.96e-6`。三步 loss 最大绝对差 `6.19888e-6`。

每步 290 项全参数梯度、494,032,768 个元素均被逐项比较：最大绝对差依次为 `3.22238e-7`、`3.09199e-7`、`3.76254e-7`，相对 L2 为 `4.89679e-6`、`4.90052e-6`、`5.36509e-6`。step-3 checkpoint 的 290 个权重键对齐，169 个张量发生更新，末态最大绝对差 `7.45058e-9`、相对 L2 `6.60016e-11`。该受限结果支持同输入下的前向、梯度和末态权重数值；当时尚未比较 optimizer state，L2 为 partial，L3 未运行。单样本结果不推断多样本 sampler 行为。

首个比较器在第 3 步 `input_ids` 身份断言处停止。虽然两侧均保存了三步梯度及末态 checkpoint，比较器未校验后续步的输入身份，故这些记录不能作为同输入梯度/更新通过证据。训练日志 loss 轨迹接近也不替代批次身份。native 训练约 20.8 秒，shim 约 540.6 秒且包含首次 JIT；不作为 L5 性能结果。


三步训练 checkpoint 的只读状态审计运行键 `20261010-qwen2-orpo-fixedbatch-l2-audit-v5` / Slurm 15279 在 cscg-qh04 RTX 4090 完成。审计通过 strict Jittor Torch shim `torch.load` 读取 native 与 shim 的 optimizer/scheduler 文件，父进程 CUDA=1、fallback=0。两侧 SGD `state` 均为空，2 个参数组分别含 169/121 项；两组所有选项完全相同（learning rate `1e-5`、momentum 0、weight decay 分别 0.1/0.0）。scheduler 的 `base_lrs`、`last_epoch=3`、`_step_count=4`、`_last_lr` 与其他共同字段相等；native 独有 `verbose=false`，shim 独有 `_is_initial=false`，故状态字典非完全相等。候选 optimizer checkpoint 为 Torch shim 的 portable pickle 格式，已由 shim loader 成功读回；该格式差异本身未导致恢复失败。直接从每步梯度按参数映射重构权重更新仍未完成，L2 保持 partial。审计脚本首三次分别因 NumPy bfloat16 读取限制、named-parameter 与 optimizer 参数顺序不同、native loader 不识别 shim pickle 格式而终止；这些都是审计器假设错误，并非训练故障。第四次因漏配 PYTHONPATH 在 import 前失败；不计兼容结果，所有日志与运行键均留存。

报告门禁运行键 `20261010-qwen2-orpo-fixedbatch-l2-docgate-v1` / Slurm 15280 在 cscg-qh04 完成。`bash tools/check_repo_layout.sh` 通过；以 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python` 运行 Torch-mode `tests/structure`，结果为 1384 passed、8 skipped、1019 subtests passed（516.55 秒）。4 个额外 skip 分别来自未安装 core wheel 与 pytest-xdist；门禁脚本注明另有 4 个声明性 skip。该文档门禁未执行模型验收。

### 单固定样本：12 步收敛协议

按更新后的收敛门槛，新运行键 `20261010-qwen2-orpo-fixedbatch-convergence-v2` 保存原生独立 oracle；v2 作业 15420/15426 分别完成隔离 CUDA 缓存构建与 `JITTOR_NO_BUILD=1` 复用检查，11 个共享库路径及 SHA256 完全一致。原生 12 步 CLI 于 job15427 完成。v2 shim 产物因 runtime scope 路径配置错误排除；新键 `20261010-qwen2-orpo-fixedbatch-convergence-v3` 仅重跑严格 shim 侧，job15430 预检成功，job15431 完成 12 步 CLI 并保存 checkpoint-12。两侧均为 Qwen2-0.5B、FP32/eager、全参数、batch 1、固定首条偏好记录、SGD lr `1e-5`、constant scheduler、`max_grad_norm=1`，使用相同单行数据 SHA256 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10` 与模型 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945872fecdf77cad`。GPU worker 为 cscg-qh04 RTX 4090，UUID `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea`；shim 训练父子进程均记录 CUDA=1、shim=True、fallback=0。

GPU-worker 只读 artifact audit 新键 `20261010-qwen2-orpo-fixedbatch-convergence-audit-v1` / job15441 完成，未重跑训练。全部 12 步输入逐张量完全相同；首步 forward shape/dtype/有限值合法，chosen/rejected logits shape `[1,28,151936]`，跨 runtime 最大绝对差 `8.16137e-5`、相对 L2 约 `2.54e-6`。两侧 loss 的 step1–3 均值到 step10–12 均值分别由 `3.98413571` 降至 `3.92896231`（native）、由 `3.98413301` 降至 `3.92895770`（shim）。每一步两侧 290 个 trainable 参数均有有限且非零 CUDA 梯度，更新非空；SGD 两参数组分别为 169 项 decay `0.1`、121 项 decay `0`，lr `1e-5`、momentum 0，state 结构有效。scheduler 每步的重复 hook 记录只有在相邻记录逐字段完全相同时才折叠；12 个有效状态步的 `last_epoch`、双组 lr、`_step_count` 均正确。checkpoint 两侧均为 290 键、各 244 个张量相对初态改变，global step 12；末态跨 runtime 最大绝对差 `7.45058e-9`、相对 L2 `1.39669e-10`。优化器/scheduler 文件格式的 runtime 私有字段差异不要求数值相同。

该 profile 的层级结论仅限单固定样本公开 ORPO CLI：

| 层 | 状态 | 证据与边界 |
|---|---|---|
| L0 | pass | 公开 CLI 构造同一真实模型、数据、ORPO trainer 与 SGD optimizer；290 个初始参数名称、shape、dtype、trainable 与 CUDA 状态一致。 |
| L1 | pass | 12 步同输入；首步 forward 结构、dtype、有限值及可反向 loss 合法。跨 runtime 数值差只作诊断。 |
| L2 | pass | 两侧各自 12 步 loss 窗口下降；逐步 290 参数梯度完整、有限、非零，真实更新非空，optimizer/scheduler/checkpoint 结构有效。 |
| L3 | pass（固定单行数据的恢复延伸） | 2026-10-11 补验同 Trainer 实例与新 CLI 进程恢复；完整状态、固定行游标、继续更新与 loss 窗口见下节。多行 sampler 顺序仍未覆盖。 |
| L4 | pass | 两侧真正通过公开 `swift rlhf --rlhf_type orpo` CLI 并保存 checkpoint；仅此固定单样本配置。 |
| L5 | blocked（strict shim 测量轨迹 step 16 失败） | 原生十次稳态测量完成；strict shim 在要求的测量 hook 下触发 `NanoVector` slice overflow，没有候选性能结论。诊断边界与运行键见下节。 |

该新 profile 不改变前述多样本 sampler 限制，也不覆盖其他 ORPO 数据、并行模式或 ms-swift 其他功能面。job15418 的 `/usr/bin/time` 缺失、job15427 的 marker 路径问题和 job15431 的比较器分组假设均为 harness 失败，已保留原日志；最终结论仅基于原生 15427 与严格 shim 15431 的现有产物及成功的只读审计 15441。

### 单固定样本：L5 稳态测量（2026-10-11）

L5 新键 `20261011-qwen2-orpo-fixedbatch-l5-steady-v4` / Slurm job16970 绑定 Jittor HEAD `dd7e091d4242affbb2d2b53573a2382a1f3a16ad`、ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`，在 cscg-qh04 RTX 4090（GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）执行。原生公开 CLI 从 checkpoint-12 训练至 step24，step13–14 预热，step15–24 同步计时十步：median `0.08785684 s`、P95 `0.08876179 s`、`11.35135` preference pairs/s、`658.37828` tokens/s；PyTorch allocator 峰值 allocated/reserved 为 `5,956,803,584 / 6,215,958,528` bytes。该 allocator 统计不能与 Jittor 内存池数值直接互换。原生记录 SHA256=`5c6b2081480a014de8796245af3021805eef9c3d592124c986b2041bb4ce28a3`。

同一 job 的 strict shim 恢复与公开 CLI 参数相同，父/子进程均为 CUDA、fallback delta=0，完成 step13–15；step16 backward 在 `autograd_api.py::_backward → jt.core.grad_optional` 触发 `nano_vector.h:41 slice overflow`，没有 shim 稳态 JSON 或 L5 结论。最多五轮的最小化诊断保留如下：16971 去除外部测量插件后从 checkpoint-12 连续完成至 step16 并保存 checkpoint-16；16973 的同步探针因自身错误访问 Accelerate optimizer wrapper 的 `state.global_step`、在首个更新后退出，不作兼容证据；16974 只在每步前和 SGD 后同步、无张量审计仍在 step16 复现 overflow；16975 只保留步前 `torch.cuda.synchronize()` 可完成至 step16；16976 只保留优化器后的 `torch.cuda.synchronize()` 也可完成至 step16。结果将失败边界缩小到双侧同步/观测组合在该连续轨迹中的交互，尚未定位其更深层 C++ 根因；不据此断言普通未插桩训练失败。达到本轮故障诊断上限后停止该 L5 重试，保留 strict L5 为未通过。日志、脚本、清单和产物均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-orpo-fixedbatch-l5-steady-v4/` 与相应 `l5-uninstrumented-v1`、`l5-sync-only-probe-v1/v2`、`l5-presync-only-probe-v1`、`l5-postsync-only-probe-v1` 目录。

### 单固定样本：L3 checkpoint 恢复补验

恢复验证在当前 Jittor HEAD `049bcc8548a63dfad654a605564e6f1949a1bbda` 上完成，ms-swift checkout 为 `88d727951203256baa564c643c651b6f8d90fd7e`。只读输入是 Slurm job15427 的原生 PyTorch checkpoint-12 和 job15431 的 strict-shim checkpoint-12；未重跑或修改这两个旧训练。冻结运行键 `20261011-qwen2-orpo-fixedbatch-l3-restore-v16` / job16941 在 cscg-qh04 的 RTX 4090（GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）上依次执行原生 oracle 和 strict shim；只读审计键 `20261011-qwen2-orpo-fixedbatch-l3-restore-audit-v2` / job16946 校验已完成产物。模型为 Qwen2-0.5B，单卡、FP32/eager、单固定偏好行、全参数 SGD，seed/data_seed 1234。原生与候选逐步 `input_ids`、`attention_mask`、`labels` 的身份 SHA 均为 `71157c71cea191871b2e56fb8b08e8f67e2bc911a00b47b691ee7a951536aed8`。

两侧均完成两条恢复路径：同一个公开 CLI Trainer 从 checkpoint-12 继续到 step14，在同一 Trainer 对象载入新写出的 checkpoint-14 并继续到 step16；新进程路径由 CLI A 从 checkpoint-12 写到 step14，再由全新 CLI B 恢复 checkpoint-14 并到 step16。每次恢复时 290 个模型权重逐值匹配该 checkpoint，`lm_head.weight` 与 embedding 的 tied alias 共享 CUDA storage；optimizer 参数组（169/121）、optimizer state、scheduler、Python/NumPy/Torch CPU/CUDA RNG 均与各自保存状态精确相符。恢复段 global_step/epoch 正确推进。checkpoint-14 和 checkpoint-16 均确认含非空 `model.safetensors`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`，其中 TrainerState global_step 分别为14和16；checkpoint-14 还被后续同进程或新进程实际重新加载。

steps 13–16 每一步的 290 个 trainable 参数梯度都在 CUDA 上、有限且非零，并执行了非空 SGD 参数更新。两侧各自 steps15–16 平均 loss 都低于 steps13–14：native `3.91392481 → 3.90198851`，strict shim `3.91392910 → 3.90199387`（同进程）；strict shim 新进程为 `3.91392684 → 3.90199125`。四条轨迹都重复同一固定行，因此 cursor 检查覆盖 global_step/epoch 恢复和单行数据续取；它不覆盖多行 sampler 的非平凡顺序。strict shim 的 worker 与公开 CLI 子进程均记录 `use_cuda=1`、fallback delta=0。跨 runtime 的逐步 loss 差最大约 `1.22e-5`，只作诊断。

job16941 在三个轨迹都已训练并保存后，于 worker 内比较器因把原生文件名错误拼成 `native/native-same.jsonl` 而以 exit 1 结束；没有发现 CUDA 或恢复断言失败。job16946 修正路径分组后只读审计四条轨迹成功。两次日志、逐步梯度与更新摘要、checkpoint manifest 和比较 JSON 均未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-orpo-fixedbatch-l3-restore-v16/` 与 `20261011-qwen2-orpo-fixedbatch-l3-restore-audit-v2/`。本节仅补齐固定单行 profile 的 L3；多行 sampler 问题仍未解决，L5 稳态性能未运行。

新结论文档门禁 run key `20261010-qwen2-orpo-fixedbatch-convergence-docgate-v1` / Slurm 15445 在 cscg-qh04 完成：布局检查成功；Torch-mode `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed。worker 未配置 nvcc，该门禁只验证仓库布局和结构契约，不构成 CUDA 兼容性证据。

本次 L3 报告更新的门禁 run key `20261011-qwen2-orpo-l3-report-docgate-v3` / Slurm 16952 在 cscg-qh04 完成：`tools/check_repo_layout.sh` 通过，文档治理检查238个活动 Markdown；Torch-mode `tests/structure` 为1384 passed、8 skipped、1019 subtests passed（235.05秒）。复用已验证的结构专用持久 `JITTOR_HOME` 原绝对路径；Python/SOABI、g++ 11.5、Linux 6.1.188 主版本、source tree SHA、CPU arch/config一致，nvcc缺失且明确走CPU结构路径。`JITTOR_NO_BUILD=1` bootstrap检查1.7秒通过，门禁前后 `jittor_core` / `jit_utils_core` SHA相同，日志无core冷编译。v1/v2在工具链指纹的过严字符串断言处退出，未运行测试；日志保留在state，不影响v3结果。该门禁不是CUDA计算证据。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | pass（单固定样本12步 profile）；pass（两行12步 control） | 固定样本与两行 control 的公开 CLI 均构造真实模型、trainer、optimizer 和 CUDA 参数；loader-only 分支另有三轮 sampler/输入身份复核。 |
| L1 | pass（单固定样本12步 profile）；partial（两行12步 control） | 两行 native 与 strict shim control 均使用相同步批次身份并完成有限loss的真实 forward；该多样本报告尚无成套 logits/hidden shape、dtype 和逐值诊断输出，故不提升为 L1 pass。 |
| L2 | pass（单固定样本12步 profile）；partial（两行12步 control） | 两行 native 与 strict shim control 的各步 290 项梯度有限/非零且有真实更新，各自末段 loss 窗口下降；按层级顺序，L1 输出结构仍不完整，故多样本 L2 暂不验收为 pass。 |
| L3 | pass（单固定样本恢复延伸）；partial（多样本） | 固定单行 profile 的同/新进程恢复通过；多样本原生 new-process 子路径通过。strict shim 的 12 步 control 已运行，实际 Trainer RNG restore 的独立状态比较通过，最小同进程 step1→2 通过；完整多样本同/新进程恢复的状态、游标、逐步梯度/更新及 loss 窗口证据仍未成套通过。SHA 类型标记误报已纠正；无 hook 基线、分别启用单个完整 RNG hook、以及两个 RNG hook 的轻量日志组合均未触发 NanoVector。原失败运行还包含 loss/batch/gradient/model/optimizer observers，具体触发 hook 或交互未定位。 |
| L4 | pass（单固定样本12步 profile）；blocked（两行恢复未通过） | 单固定样本公开 CLI 端到端通过；多样本公开 CLI 的 loader-only 与 12 步 control 均已实际运行并保存 checkpoint，但完整恢复未通过，故多样本 L4 受 L1/L3 前置缺口阻塞。 |
| L5 | not-run（多样本）；strict shim fixed-sample L5 failed | 多样本 L4 未通过，未执行其性能协议；固定单样本 strict shim 的 step16 backward 故障仍保留为未通过。 |

原始脚本、模型/数据摘要、NPZ、逐步梯度、checkpoint 和日志未版本化，分别保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-orpo-stock-cli-v1/`、`20261009-qwen2-orpo-fixedbatch-cli-v2/` 和 `20261009-qwen2-orpo-input-diagnose-v1/`。本报告仅适用于上述单卡 ORPO 配置，不代表其他偏好算法或整个 ms-swift。

文档门禁 Slurm 13894 在 RTX 4090 worker 上通过：`tools/check_repo_layout.sh` 成功，`JITTOR_TORCH_SHIM=1 pytest -q tests/structure` 为 1386 passed、6 skipped、1027 subtests passed。单样本证据更新后的第二次文档门禁 Slurm 13922 同样通过：布局成功，结构测试 1386 passed、6 skipped、1027 subtests passed。两次测试日志均说明 2 个跳过项因解释器未安装 Jittor、2 个因未安装 pytest-xdist；其余跳过项由门禁分类记录。

### 多样本 L3：原生新进程恢复子路径（strict shim 未运行）

多行 ORPO L3 的 v5 运行键 `20261011-qwen2-orpo-multisample-l3-resume-v5` / job16993 曾在原生 new-process B 的 step4 突然退出，日志没有 traceback、信号或 OOM 证据；strict shim 与比较阶段没有启动。新诊断键 `20261011-qwen2-orpo-multisample-l3-resume-v6` / job17008 只从 v5 不可变的 native checkpoint-1 执行原生公开 CLI 新进程 B，没有重跑 control、same-process、new-process A 或任何 strict-shim 阶段。该次恢复完成到 step12，CLI 返回0，checkpoint-1 关键文件前后 SHA 不变；v5 的 step4 退出未重现，原始原因仍未知。

只读审计确认新进程恢复了 290 个模型张量及 optimizer、scheduler、Python/NumPy/Torch CPU/CUDA RNG；续训 step2–12 的批次身份与 v5 原生连续 control 相同，每步 290 个参数梯度均有限且非零并发生真实更新，loss 窗口均值从 step1–3 的 `3.46475267` 降至 step10–12 的 `3.42912006`，并保存 checkpoint-12。该证据只通过原生 new-process 子路径；strict shim 未运行，因此多样本 L3 整体仍未验收，也不改变已观察到的跨 runtime sampler 顺序差异或给出兼容结论。产物未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-qwen2-orpo-multisample-l3-resume-v5/` 与 `20261011-qwen2-orpo-multisample-l3-resume-v6/`。

### 多样本 L3：strict shim 恢复诊断（2026-10-12）

strict-shim 运行键 `20261012-qwen2-orpo-multisample-l3-strict-shim-v2` / job17315 完成了两行 ORPO control 的 12 步并写出 checkpoint-12；same-process 场景生成了 step-1 checkpoint，但观测器随后以 CUDA RNG SHA 不同退出，new-process 场景及完整比较器没有启动。该断言不是兼容性失败：同一只读 checkpoint 的新诊断键 `20261012-qwen2-orpo-rng-state-digest-audit-v1` / job17333 在 cscg-qh04 RTX 4090（UUID `GPU-4b797e8c-3982-9b01-dc17-43d77960643c`）通过 strict CUDA scope 真实调用 Transformers 4.57.6 `Trainer._load_rng_state`。恢复后的 Jittor Var 与保存的 Torch Tensor 均为 28 个 `uint8` 值、shape `[28]`，原始字节 SHA256 相同（`06d2929f63149d025449ef25ad0534e63e454d07b274772292f2dd473a8ec884`），`torch.equal`、NumPy 元素比较和原始字节比较均为 true，fallback 0。旧观测器给两种容器分别加 `J`/`T` 类型前缀，所以即使值完全相同，复合 digest 仍不同；真实 mismatch 断言是比较器的类型标记误报。

对 v2 已完成的独立 uninterrupted control 事件作只读复核后，native v5 control 与 strict-shim v2 control 的 12 个逐步输入身份 SHA 全部一致；两侧各 12 步都有 290 项 finite/nonzero gradients 与非空更新。loss step1–3 到 step10–12 窗口均值分别为 native `3.4647526741 → 3.4291200638`、strict shim `3.4647494952 → 3.4291232427`。strict-shim父/子进程均为 `use_cuda=1`、fallback delta=0。它补足两行独立 control 的训练轨迹证据，但不替代完整前向结构输出或恢复轨迹，因此不能将多样本 L1–L3 全部标为通过。原生事件来自 v5/job16993，strict shim 事件来自 v2/job17315；均为只读复核，没有重跑 oracle 或改写其产物。

另一个最小公开 CLI 键 `20261012-qwen2-orpo-rng-restore-same-process-minrepro-v3` / job17323 带有 RNG、loss、batch、gradient、model 与 optimizer 观测 hook；它完成 step1/checkpoint-1 后，同一 Trainer 恢复到 step2 的 backward 在 `autograd_api.py::_backward → jt.core.grad_optional` 触发 `nano_vector.h:41 slice overflow`。新键 `20261012-qwen2-orpo-rng-restore-same-process-minrepro-v4` / job17330 去掉这些状态及张量观测 hook，仅保留同 Trainer step1→2 调用，严格 CUDA 下完成并保存 checkpoint-2，父子进程 `use_cuda=1`、fallback 0。两次最小运行表明该断点与额外观测配置相关，但具体触发 hook 或更深层原因尚未隔离；不能将 v4 的两步烟测推广为多样本 L3 通过。17317、17318、17321、17322 为在模型/恢复前退出的 harness 诊断，证据保留且不作兼容结论。以上键与日志位于各自 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-orpo-*/` 运行目录。

为隔离上述 observer-correlated 故障，新增 load-RNG-only 键 `20261012-qwen2-orpo-rng-restore-loadrng-hook-minrepro-v2` / job17343、set-RNG-only 键 `20261012-qwen2-orpo-rng-restore-setrng-hook-minrepro-v1` / job17345，以及两层 wrapper 仅写事件、不读取张量值的组合键 `20261012-qwen2-orpo-rng-restore-combined-light-hooks-minrepro-v1` / job17346。三者均在真实 RTX 4090 strict CUDA、fallback 0 下完成同 Trainer step1→2 并保存 checkpoint-2；job17343 的 `_load_rng_state` 后 CUDA RNG bytes 精确相同，job17345 的 setter 后 bytes 精确相同。对照基线 job17330 无观察 hook，也通过step1→2。失败的 job17323 还同时启用 loss、batch、gradient、model 与 optimizer 观察器，故上述隔离只能排除单个 RNG hook 或轻量双 wrapper 会稳定触发该故障，不能将其归因于两组完整 RNG 张量读取/比较。经五次该根因观察后，保留边界为“与复杂状态观察器栈相关、具体 hook/交互未定位”；17343/17345/17346只是两步诊断，没有跨 runtime 梯度或完整恢复合同，不提升多样本 L3，也不归因为模型恢复实现缺陷。此前v1登录节点缺 `python`，在 sbatch 前停止，保留于其 run 目录且没有 worker。
