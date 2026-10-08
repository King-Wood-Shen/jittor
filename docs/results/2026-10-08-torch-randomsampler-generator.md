# Torch `RandomSampler` 显式 Generator CUDA 运行时对拍

- 状态：固定长度八样本 API 探针完成；仅 L0 通过，L3 partial，L1/L2 不适用，L4 partial，L5 not-run。该探针不覆盖模型计算或 ms-swift 公开 CLI。
- 日期：2026-10-08。
- 基线：Jittor `d59afc84713309a8f6f94d370339df94ecf74557`；Slurm worker 同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`。
- 范围：原生 PyTorch 与 `jittor.compat.torch` 的 `torch.Generator(device='cpu')`、`manual_seed`、`get_state`/`set_state` 和 `torch.utils.data.RandomSampler`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch shim generator、sampler 或 Swift 数据加载器变化时复验；将结论扩展到模型训练前，需用公开入口额外捕获逐批样本 ID。

Slurm 13189 在 `cscg-qh04` RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`）先跑无 shim 的原生 oracle，再跑 strict shim。原生为 PyTorch `2.6.0+cu124`；候选 Jittor/Torch shim 为 `2.0.0`。候选进程从启动到退出受 `runtime.scope(use_cuda=1, backend_fallback='error')` 和 `forbid_backend_fallbacks()` 约束，`use_cuda=1`、`fallback=0`。Sampler 仅产出 CPU 整数索引；本运行没有 CUDA tensor 或模型计算，故设备信息仅证明 strict CUDA runtime 已启用，不声称 sampler kernel 在 GPU 执行。

在数据集长度 8、seed 1234 下，原生首轮索引为 `[7,5,2,3,4,6,1,0]`，候选为 `[5,6,7,0,4,1,3,2]`；第二轮顺序也不同。两侧分别保存 generator 初态和完成一轮后的状态，再用 `set_state` 恢复。原生和候选各自恢复首轮及下一轮的顺序均逐项相同，证明这组 generator 状态能在各自 runtime 内重放。两边状态哈希不同，符合各自随机数流不同的观察。

此结果说明相同 seed 不能作为跨 runtime 样本顺序相同的证据；这是由本次索引差异得出的操作性结论，不据此认定 PyTorch RNG bitwise parity 是项目承诺，也未在 ms-swift Trainer 中复现。训练对拍应记录或固定实际 batch 的样本身份；已验证的 `dataset_shuffle=false` 可用于固定数据顺序场景，但仍应检查实际 batch。

| 层 | 状态 | 证据与边界 |
| --- | --- | --- |
| L0 | pass | 两侧均构造 CPU Generator 与 `RandomSampler` 并生成八项 permutation；原生未导入 shim。 |
| L1 | not-applicable | sampler API 不执行模型前向。 |
| L2 | not-applicable | sampler API 无反向或优化器更新。 |
| L3 | partial | 两侧各自在本 runtime 内恢复 generator state 后重放首轮与下一轮；未测试 DataLoader iterator/sampler cursor 的 checkpoint 恢复。 |
| L4 | partial | 仅直接调用 Torch sampler API；未进入 ms-swift 公开 Python API 或 CLI。 |
| L5 | not-run | 没有模型或 sampler 性能协议。 |

仓库布局、manifest 及 Torch 模式页面可达性门禁在 Slurm 13190 通过，页面可达性测试 1 passed。原始脚本、JSON、环境及日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-randomsampler-generator-v1/`。此前 Slurm 13063 的旧基线 probe 只作定位线索；本结论绑定本报告记录的当前同步基线与 Slurm 13189。
