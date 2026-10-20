# ms-swift DPO training through JTorch on CUDA

**状态：** 单卡 CUDA 公开 DPO 训练对拍通过；Qwen2 causal-mask 捕获中的未分配 tensor 原地写入已修复。
**日期：** 2026-10-20
**基线：** `origin/2.0-refactor` / `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`
**维护者：** Jittor 维护者（本次验证执行）
**范围：** ms-swift `swift rlhf --rlhf_type dpo`、Qwen2-0.5B、单卡 RTX 4090、JTorch CUDA shim 与原生 PyTorch CUDA 对照。
**复查条件：** 换模型家族、精度或注意力实现时，按矩阵新增对应入口与 CUDA 证据。

使用固定偏好样本、FP32、eager attention、SGD、三步更新。Native 与 shim 都经公开
`swift rlhf` 入口运行。Native losses 为
`0.69314718 → 0.69251090 → 0.69187433`；shim 为
`0.69314826 → 0.69251192 → 0.69187522`，最大逐步绝对差 `1.08e-6`。两侧 loss
都整体下降；shim runtime sentinel 为 `cuda=1`、`fallback=0`。同一 Slurm allocation
上的 NVML 设备峰值为 native `8348 MiB`、shim `14897 MiB`（100 ms 采样）。这是整卡
峰值记录，不外推为其他模型或序列长度的显存比率。

第一次 shim 训练在首个 loss 后，于 Transformers Qwen2 causal mask 的
`kv_arange += kv_offset` 崩溃。原因是 Torch shim 为带 detached alias 的原地操作调用
`_copy_into`，但 graph replay 中刚创建的 arange 只有逻辑形状、尚无 backing allocation；
`_copy_into` 要求目标已经分配。修复后，`_copy_detached_storage` 对
`_mem_ptr_now == 0` 的目标返回未处理，让 `_ip` 走 Var 图赋值路径；已分配的 detached
storage 仍保留原地拷贝路径。CUDA 定向回归
`TestCompileReplayCuda.test_capture_materializes_a_fresh_inplace_target` 覆盖同一新建
arange、原地加法和 graph replay，三组 offset 均通过。自动 graph replay 也新增训练态
模块保护；DPO 内部的 causal-mask 路径仍可能运行在 eval/no-grad 模式，因此 tensor
未分配回退才是本次 DPO 故障的直接修复。

成功作业 110372；定向 CUDA 回归作业 110371。完整命令、训练日志、NVML 样本、runtime
sentinel 与 manifest 保存在实验区
`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-dpo-cuda-fix2/`；回归日志在
`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-dpo-inplace-cuda-test/`。首轮失败栈保留于
`20261020-dpo-cuda/logs/shim.log`。本次未运行 CPU 全量或 smoke 门禁。
