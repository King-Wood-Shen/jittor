# CUDA DPO 反向传播中的越界切片归一化

- 状态：修复后在真实 CUDA 设备上完成单步训练；CPU 定向回归通过。
- 日期：2026-10-13。
- 同步基线：`4896af611650d1436302c06b0653ac61726fd10c`（`origin/2.0-refactor`）。
- 范围：Jittor 共享 `Slice::fill`、单卡 CUDA、公开 ms-swift DPO 单步。
- 维护者：Jittor core maintainers。
- 复查条件：切片归一化、getitem/setitem 梯度、PySlice 转换或 CUDA 执行路径变更。

## 现象与定位

原始 Slurm 作业 `21607` 的 DPO 反向传播在 `Slice::fill` 断言：`start=93983792218421`、`stop=0`、`step=1`、`size=2`。工作树原有的一行诊断改动给断言增加 `size`，确认失败发生在长度为 2 的维度；该改动保留在修复中。`PySlice_Unpack` 只提取 Python 切片边界，`GetitemOp::infer_index_slices` 首次调用 `Slice::fill` 时才按维度大小归一化。旧实现只限制了正向 `stop`，没有限制过大的正向 `start`；反向 `start` 也有同类缺口。因此合法的 Python 大起点切片会在首次归一化时变为无效的内部切片。原始 DPO 大起点的具体 Python 构造调用点未由旧日志记录，不能由其数值推断。

最小复现是长度 2 的张量 `x[10**14:0]`。Slurm GPU 作业 `21661` 在修复前按预期以相同断言失败，说明问题位于 Jittor 的共享切片实现。`tiny.py` 作业 `21525` 仅遇到导入顺序的 harness 错误，不是计算复现；它未重跑。此前 `dpo2shape` 作业 `21594` 完成了两步，说明原 DPO 失败不能仅凭运行步数稳定复现；本次没有把该旧结果当成修复验证。

## 修改与验证

`Slice::fill` 现在按切片方向将起止边界收在合法范围，允许正向空切片的起点等于维度大小，并保留诊断信息中的 `size`。`tests/ops/test_slice.py` 覆盖大正向起点的空切片和大反向起点的实际 CUDA 取值。

| Slurm 作业 | 范围 | 结果 |
| --- | --- | --- |
| `21661` | CUDA 最小回归，修复前 | 预期失败：`slice overflow: 100000000000000 0 1 size: 2` |
| `21662` | CUDA 最小回归，修复后 | `1 passed`；实际 CUDA 运算；fallback 增量为 0 |
| `21663` | 公开 ms-swift DPO 单步，修复后 | `global_step/max_steps=1/1`；Slurm `COMPLETED`、退出码 0；两个 shim 进程退出时 `use_cuda=1`、fallback 增量均为 0 |

同一最小回归在 CPU 上以 `JITTOR_TEST_DEVICES=cpu use_mkl=0` 运行，`1 passed`。仓库布局门禁通过。首次 core 门禁未设置 `use_mkl=0`，13 项失败均由当前解释器缺少 `cmake`、无法构建 oneDNN 引起；该环境失败保留在未版本化的 `core-gate.log` 中。设置 `use_mkl=0` 后 core 的 native/Torch 两个会话合计 `322 passed, 42 skipped, 1 xfailed`，退出码 0。

结构门禁首次运行 `1381 passed, 8 skipped, 3 failed`。3 项失败分别是同一 oneDNN 环境问题、新增报告后尚未再生的 `MANIFEST.in`，以及 ROCm 结构测试错误地在绝对源码路径中查找 `cuda`（当前工作树目录名包含 `cuda`）。清单已再生；前两项在 `use_mkl=0` 下定向复验 `2 passed`。ROCm 路径误报不涉及本次切片代码，未修改该测试。

原始日志、Slurm 脚本及运行产物未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261013-nanovector-minrepro/`。DPO 的第一次修复后运行耗时约 7 分 37 秒，包含首次 CUDA 算子编译，不作为性能比较。未修改外部 ms-swift checkout；ROCm 与 NPU 未验。


## 2026-10-14 最新主线集成

在独立分支 `fix/ms-swift-cuda-dpo-integration` 合入 `origin/2.0-refactor` 的 `bdcbe8de5`，保留 CUDA 分支已有的 `randperm` 兼容实现，再接入本报告的 Slice 修复。主线把 CPU RNG 引擎迁入 `runtime_rng_state()` 后，旧快照接口仍引用已不存在的 `eng/current_seed`；现统一读取和恢复该运行时对象，且不触发 CUDA 重播种。主线撤掉 ACL Dropout 类后，结构清单仍登记该类；已移除失效登记并按生成器重建 `MANIFEST.in`。合并后的 `cat` 快速路径仍经过输入梯度状态传播，ACL 同类型快捷路径同样保留该语义。

验证：Slurm `24433` 在真实 RTX 4090 上通过切片 1 项、Torch 兼容 51 项与 37 个子测试；`24573` 的完整结构门禁为 1405 通过、6 跳过、2 失败，失败分别对应上述过期登记和清单顺序；修复后 `24578` 定向复验 2 项通过，core 门禁合计 323 通过、42 跳过、1 个预期失败。初次混合 native/Torch 测试入口和随后错误的顶层包路径都属于运行脚本问题，已改为分进程、包内路径复验，不计为产品缺陷。

本集成分支尚未重跑公开 DPO 三步及 checkpoint，因此前文 `21695` 的通过结论仍只属于 `9ffe7cbd9` 基线。2026-10-14 预检显示 `/home/xinshen` 挂载仅剩约 2.2 GiB，而既有三步 DPO 输出约 1.6 GiB；按本项目存储余量规则，新的保存作业暂不提交。Slurm `24592` 直接运行 ms-swift 的 `BatchSamplerShard(4, 1, shuffle=True, data_seed=1234)`：独立原生 PyTorch 与严格 shim 进程均得到索引 `[3, 1, 2, 0]`，两侧 `torch.randperm` 也一致。此结果与旧基线缺少 `randperm` 修复的解释一致，并证明本集成分支的该 sampler 路径一致；它仍不能代替完整 DPO 训练中逐步 `input_ids` 与 loss 的对拍。解决容量后须在本集成分支以相同模型、数据和公开入口复验，并记录逐步样本 ID。
