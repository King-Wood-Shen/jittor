# Ascend ms-swift FP16 公开 SFT 与恢复

- **状态：** 单卡真实 Ascend NPU 上的公开入口 FP16 SFT、RNG checkpoint 与恢复通过；不代表所有模型、算法或双卡支持。
- **日期：** 2026-10-20
- **基线：** `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`（同步后的 `origin/2.0-refactor`）
- **实现提交：** `520880cae`（修复 Ascend FP16 训练种子与 autocast）
- **验证范围：** ms-swift `swift.cli.main sft`，Ascend ACL，单卡 NPU，FP32 master weights + FP16 autocast；checkpoint 续训；NPU seed/state 与 autocast 定向测试。
- **维护者：** ms-swift / Torch compatibility 维护者
- **复查条件：** ACL RNG API、`torch.npu.amp`/GradScaler、mapped matmul mixed dtype 或 ms-swift AMP/恢复调用链变化时复查。

## 结论

实际 public SFT 在 `fp16=true --torch_dtype float32` 下运行 6 步，loss 与 grad norm 有限，ACL fallback 计数为 0。模型权重保持 FP32，由 FP16 autocast 执行运算；将整模型权重预先设为 FP16 会使 GradScaler 检查到 half 梯度并拒绝更新，不是此处验证的标准 AMP 路径。

候选端与原生 Ascend Torch 的 checkpoint 对比包含 21 个张量，无 shape、dtype 或非有限值差异；最大绝对差 `1.9924528896808624e-4`，相对 RMS `2.4562588092989323e-4`。从候选端 step 3 checkpoint 恢复到 step 6，与不间断候选运行的 21 个张量最大绝对差 `4.656612873077393e-10`、相对 RMS `2.572491137582835e-10`，无 shape、dtype 或非有限值差异。

新增的 seed/state 恢复及 FP16 autocast 定向用例在真实 NPU 上通过（2 passed）。实现同时修复冷 ACL C++ 编译时 `VarPtr` 完整类型缺失、ACL RNG 未走原生播种/状态 API，以及 mapped matmul 混合精度反向输入 dtype 不匹配。

## 复现与证据

所有训练和测试均在 Slurm Ascend 节点运行。验证入口是公开的 `swift.cli.main sft`，不使用 ms-swift 私有补丁；训练采用固定小型 fixture、6 steps、full tune、AdamW、eager attention。候选和原生 Torch 对照使用相同输入与配置。源输入哈希、完整命令、运行日志、事件和比较 JSON 均保存在维护者外部状态目录：

- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/msw-fp16-public-sft-20261020-r20/`：完整候选/原生训练、最终 checkpoint 与对比。
- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/msw-fp16-public-sft-20261020-r21/`：step 3 恢复到 step 6 与不间断训练对比。
- `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/msw-fp16-public-sft-20261020-r22/`：更新后的矩阵优先级定向复验。

Slurm 作业 5814 完成候选和原生对照；5815 完成恢复对比；5813 和 5820 的 NPU 定向测试分别为 2 passed。所有运行设置 `backend_fallback=error`，候选训练记录 fallback start/end 均为 0。

## 验证边界

`bash tools/check_repo_layout.sh` 通过。Torch-mode `tests/structure` 全量曾因宿主 shim 缺少 `asv`，在首项失败（47 passed，1 failed）。排除该项后的结构重跑停在约 18%，卡在 `test_process_mode_contract.py` 的长时间子进程收集；为避免长期占用 NPU，终止了自己提交的 Slurm 5827。`tests/structure` 完整结果和 core tier 尚未取得。系统 Python 本身没有 pytest，未运行 CPU full 或 smoke；也没有以 CPU 作为适配交付或数值验收设备。

双卡、推理、其他算法、BF16 与其他公开选项不由本报告授予覆盖。它们需按各自矩阵入口在真实 Ascend 上继续验证。
