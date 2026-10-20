# ms-swift SimPO gradient-checkpointing CUDA switch

**状态：** CUDA 公开入口通过；gradient checkpointing 仅验证调用路由，JTorch 当前不实现激活重算。
**日期：** 2026-10-20
**基线：** `origin/2.0-refactor` / `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`
**维护者：** Jittor 维护者（本次验证执行）
**范围：** Qwen2-0.5B、ms-swift SimPO、JTorch CUDA shim，与独立原生 PyTorch CUDA 对照；单卡 RTX 4090。
**复查条件：** checkpoint 兼容层实现激活重算后，重跑峰值显存对照并确认开关 on 确实节省显存。

公开命令走 `swift rlhf --rlhf_type simpo`，固定一个样本、FP32、eager attention、三步更新，仅切换 `--gradient_checkpointing true|false`。Shim 用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 和 `forbid_backend_fallbacks()`；两阶段运行记录均为 CUDA 且 fallback 计数为零。

| 路径 | 开关 | 3 步 loss | checkpoint 路径证据 | NVML 峰值 |
| --- | --- | --- | --- | ---: |
| 原生 PyTorch CUDA | off | `1.84327781 → 1.83792770 → 1.83255649` | 模型未启用 | 6404 MiB |
| 原生 PyTorch CUDA | on | `1.84327781 → 1.83792770 → 1.83255649` | Qwen2 模型 `is_gradient_checkpointing=true` | 6412 MiB |
| JTorch CUDA shim | off | `1.84328425 → 1.83792293 → 1.83257091` | checkpoint 调用警告未出现 | 12909 MiB |
| JTorch CUDA shim | on | `1.84328425 → 1.83792412 → 1.83256209` | 观察到 checkpoint 兼容层调用警告 | 12909 MiB |

Shim 与原生路径的最大逐步绝对 loss 差在 off/on 分别为 `1.442e-5` / `6.44e-6`。两侧训练 loss 都下降。两个 shim 阶段记录 `cuda=1`、`fallback=0`。显存数字来自两次独立 Slurm 分配的 NVML 设备序列；不是同一物理卡的配对测量，因此不能据此比较 shim 与原生实现的显存开销。shim on/off 峰值相同也符合当前兼容层行为：`torch.utils.checkpoint.checkpoint` 被直接执行并给出“不会节省激活显存”的警告。

验证使用 ms-swift commit `88d727951203256baa564c643c651b6f8d90fd7e`、本地 Qwen2-0.5B 权重 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`、固定数据 SHA256 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10`。成功作业为 110196；完整 manifest、命令、trace、NVML 采样及训练日志保存在实验区 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-gradckpt-simpo-cuda-v22/evidence/logs/`，不入仓库。

此次公开入口暴露了同步前旧候选补丁与远端不兼容的重叠：它删除了 native selector 的 `group_norm`、`layer_norm_inference`、`depthwise_conv2d` 规则实现，缩短了 cuBLAS 模块调用构造器签名，并删除 `_torch_view` 实现但留下其导入。按照同步规则，候选工作树恢复了这些文件的 `d0fd6f5` 上游实现后，作业 110196（v22）通过。async runtime flags 在远端基线的生命周期分类保持完整；候选补丁移除了其中若干项，所以最终运行同样使用远端版本。没有把这些临时绕过提交成新行为。

**结构验证：** `bash tools/check_repo_layout.sh` 与 `tools/build/generate_manifest.py --check --project .` 通过；新增 async runtime scope 定向回归 5 项通过；attention admission、设备选择契约和生成 manifest 定向结构回归 8 项通过；ASV 结构项在隔离 venv 安装项目声明的 `asv==0.6.6` 后通过。完整 `tests/structure` 首次运行结果为 1502 passed、8 skipped、10 failed；修正 CUDA attention 参数遗漏、设备选择读取方式和新增文档 manifest 后，相关定向项已通过。剩余一项 ROCm 路径断言把包含 `ms-swift-cuda` 的绝对 checkout 路径误当成源文件自身含 CUDA，未在本 CUDA 任务中修改；因此本次不宣称完整结构套件通过。没有运行 CPU full/smoke 门禁。

交叉熵设备放置另以 job 110241 在 RTX 4090 上验证：CPU 侧只准备 target/weight，`logits` 与 `cross_entropy_loss` 在 CUDA 执行；`dispatch_context(logits).backend == "cuda"`，输出 `[0.16984606, 0.68415165]` 有限，backend fallback 为零。
