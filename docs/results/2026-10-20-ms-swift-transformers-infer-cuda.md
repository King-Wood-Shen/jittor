# ms-swift Transformers inference through JTorch on CUDA

**状态：** 单卡 CUDA greedy inference 对拍通过。
**日期：** 2026-10-20
**基线：** `origin/2.0-refactor` / `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`
**维护者：** Jittor 维护者（本次验证执行）
**范围：** ms-swift `infer`、Transformers 引擎、Qwen2-0.5B、单卡 RTX 4090。
**复查条件：** 更换模型家族、精度或 attention backend 时分别补充对应矩阵证据。

验证使用公开 `swift infer` 命令和 `--infer_backend transformers`，以一条固定 `val_dataset` 样本进行 greedy generation（`temperature=0`、FP32、eager attention、`max_new_tokens=8`）。原生 PyTorch CUDA 与 JTorch CUDA shim 在同一 GPU allocation 中顺序运行。两侧生成内容完全相同：`The answer to the question "What is`（按 8-token 上限截断）。Shim runtime sentinel 记录 `cuda=1`、`fallback=0`。

NVML 按 100 ms 采样：原生路径设备峰值 2438 MiB（88 个样本），shim 路径 2883 MiB（137 个样本）。这是同卡顺序采样的整卡占用峰值，包含进程之外的基础占用；只作为该配置运行记录，不外推为其他模型或长度的显存比率。原生推理用时 10.4 s，shim 16.6 s；计时不是本轮门槛。

环境：ms-swift commit `88d727951203256baa564c643c651b6f8d90fd7e`，Jittor worktree HEAD `c58648563db78945df21f4a3b7910b33d3460661`，候选 dirty diff SHA256 `97f921258532fffd2eb6a498197f5c81b3fbd35cb026ac8a97fd5cc4c0ebebfe`。成功 Slurm 作业 110302。完整命令、JSONL 输出、runtime sentinel 和 NVML 样本保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-qwen2-infer-cuda-v2/`，原始产物不入仓库。
