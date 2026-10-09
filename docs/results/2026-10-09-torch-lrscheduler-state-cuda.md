# JTorch 学习率调度器状态字段 CUDA 对拍

- 状态：修复 JTorch scheduler `state_dict()` 与 PyTorch 2.6 的状态字段差异；三类调度器的 CUDA 状态、学习率轨迹和参数值与原生一致，严格 shim fallback 为 0。标准 core CUDA tier 未通过，失败项与 scheduler 修改无关或受 worker 环境依赖影响。
- 日期：2026-10-09。
- 基线：修复提交 `596f17a2aae75bfae04ddeb8cfcc41d848eed5b6`，父提交 `bef17f588171e757feec77b1e915c1175c50d527`。
- 验证范围：PyTorch 2.6.0+cu124 与 JTorch `jittor.compat.torch` 的 `LambdaLR`、`MultiplicativeLR`、`StepLR`，单 RTX 4090 CUDA 参数，严格 fallback 策略。
- 维护者：ms-swift CUDA 适配。
- 复查条件：scheduler 初始化、`state_dict`/`load_state_dict`、训练恢复、PyTorch scheduler API 或 JTorch optimizer 调度集成发生变化时。

## 根因与修复

Qwen2 SimPO 双卡 checkpoint 的状态检查发现，原生 PyTorch 保存 `verbose=False`，JTorch shim 却保存初始化期间使用的瞬态 `_is_initial=False`。这来自 compat scheduler 构造后没有保留 `verbose`，且把 `_initial_step()` 的临时标记保留在实例字典中。

提交 `596f17a2` 为 scheduler 实例保存 `verbose`，并在初始 step 的 `finally` 中删除瞬态 `_is_initial`；普通和 lambda scheduler 的 `state_dict()` 都排除该瞬态字段。对应回归测试覆盖 verbose 值与标志生命周期。改动位于 compat 公共 API，不修改 ms-swift 源码。

## 验证

Slurm 14086 在 `cscg-qh04` 的 RTX 4090（UUID `GPU-4b797e8c-3982-9b01-dc17-43d77960643c`）上先跑独立原生 PyTorch CUDA，再跑 strict shim CUDA。`LambdaLR`、`MultiplicativeLR`、`StepLR` 的 `state_dict` 键和值、逐步学习率与参数更新完全相同；状态中包含 `verbose` 且没有 `_is_initial`。候选参数位于 `cuda:0`，`fallback_delta=0`。运行键为 `20261009-torch-lrscheduler-state-compat-v1`。

Slurm 14090 在同一节点/设备执行 source-tree 定向测试，11 项 scheduler metadata 与 continuation tests 全部通过（0.95 s）。测试路径为 `python/jittor/compat/tests/...`，从仓库 `python/` 目录收集，避免将仓库物理 `compat/` 目录误当作顶层包。

标准 `tools/run_test_suite.py --tier core --backend cuda` 作业 14091 因 16 分钟时限结束；条件续跑 14099 完成测试后返回非零：native session 124 passed、4 failed、10 skipped、2 xfailed；Torch session 220 passed、6 failed。六项 Torch 失败栈都在 oneDNN 初始化，worker 环境缺少 `cmake`；native 的四项失败为随机图梯度断言、CPU/CUDA FP16 mod 和 Conv1d specialized-reference。它们没有触及 scheduler，故该 core tier 结果记为未通过，不能据此声称全树通过。

Torch-mode `tests/structure` 的首次作业 14123 发现验证环境中残留的 `jittor-torch` editable finder 会让隔离子进程把 `jittor` 解析为 namespace package。该 editable 安装由前序测试作业添加，不属于仓库改动。Slurm 14136 在 worker 卸载这项临时安装后重跑，结果为 1386 passed、6 skipped、1027 subtests passed；其中两个 core-package 子项因该 venv 未安装 Jittor core wheel而按设计 skip，另两个因缺少 pytest-xdist skip。`bash tools/check_repo_layout.sh` 与 `git diff --check` 通过。

原始脚本、状态 JSON、测试输出和日志未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-torch-lrscheduler-state-compat-v1/`、`20261009-torch-lrscheduler-state-compat-v4/`、`20261009-torch-lrscheduler-core-cuda-v1/`、`20261009-torch-lrscheduler-core-cuda-v2/` 与 `20261009-torch-lrscheduler-structure-v2/`。

这项 compat API 修复证明所列三类 scheduler 的字段和 CUDA 更新行为；它不重写先前 SimPO 作业 14059 保存的 shim checkpoint，也不自动升级该报告的历史 L0–L5 结论。修复后已另用固定 Qwen2-0.5B 全参数 FP32/SGD 公开 SFT 样例验证 step-3→4 新进程续训：scheduler 七字段和值、step-4 权重与 loss 和原生对齐，严格 fallback=0。此项受限证据和 L3 未完成项见 [Qwen2 SFT 报告](2026-10-07-qwen2-causal-sft-ffeb.md)；optimizer、RNG 实值、数据游标、同进程恢复仍未验证，不构成通用恢复合同。
