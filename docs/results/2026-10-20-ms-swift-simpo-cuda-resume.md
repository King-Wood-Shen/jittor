# ms-swift SimPO CUDA 新进程恢复

- 状态：固定单卡 profile 的公开 CLI 训练、checkpoint 新进程恢复、loss 下降、strict CUDA 与设备峰值通过；不外推到其他算法或配置。
- 日期：2026-10-20。
- 同步基线：`origin/2.0-refactor` `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`；测量源码 `c2542352eb1b236b63b95c8c04af672ac29d487f`。
- ms-swift 源码：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor 源码、ms-swift 依赖、训练参数、模型/数据、checkpoint 保存恢复路径或存储合同变化。

## 验收 profile

Slurm job `108965` 以 1 张 NVIDIA GeForce RTX 4090 完成，显存容量 `24564 MiB`。模型为本地 Qwen2-0.5B，权重文件 SHA-256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；数据为单条固定偏好 JSONL，SHA-256 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10`。

两阶段都由公开入口 `python -m swift.cli.main rlhf` 执行 `--rlhf_type simpo --tuner_type full`。首个 Python 进程训练至 step 1 并保存 checkpoint；在同一 Slurm job、同一 worker 节点内，第二个全新 Python 进程以公开 `--resume_from_checkpoint <simpo-step1>` 从 step 1 恢复并继续至全局 step 3。batch size 为 1，gradient accumulation 为 1，SGD，FP32，eager attention，固定单样本，`learning_rate=1e-5`，`beta=2.0`，`simpo_gamma=1.0`。

两份完整 Trainer checkpoint 按 `msw-storage-v2` 原子发布并完成 receipt/SHA 校验：恢复起点 `simpo-step1`、终点 `simpo-step3`。可复现命令参数数组、checkpoint 路径和损失轨迹见外置 `logs/commands-and-result.json`；恢复入口和起点完整性也记录于外置 `storage-manifest.json`。

| 全局步 | 进程阶段 | SimPO loss |
| --- | --- | ---: |
| 1 | 新训练进程 | 1.84328341 |
| 2 | 从 step 1 恢复 | 1.83791804 |
| 3 | 从 step 1 恢复 | 1.83256972 |

有限且整体下降。8 条 strict-runtime 记录均为 `cuda=1 shim=True fallback=0`。worker 起止时没有其他 compute process。通过 NVML-backed `nvidia-smi` 每 50 ms 对已分配 GPU 采样 19,453 次，实际设备级 `memory.used` 峰值为 **15,255 MiB**（基线 `1 MiB`）；峰值时间为 `2026-10-20 14:33:14.195`。这是整张设备的使用量，原始样本与汇总保存在外置 `logs/cuda-device-memory.csv` 和 `logs/cuda-peak-summary.json`；Swift `memory(GiB)` 字段未用作设备峰值。

## 运行记录与边界

未版本化证据根：`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-qwen2-simpo-resume-c2542352e-r2/evidence/`。入口参数、源码与输入身份在 `logs/run-manifest.json` 和 `logs/commands-and-result.json`；两进程日志为 `logs/simpo-initial.log`、`logs/simpo-resume.log`，fallback 计数为 `logs/shim-runtime.log`；采样文件和汇总如上。

恢复日志报告缺少 `lm_head.weight`。原模型配置的 `tie_word_embeddings=true`，与 safetensors 省略重复 tied weight 的保存方式一致；训练完整执行。另有 `eval_steps` 与 `save_steps` 的 Trainer 状态差异，分别来自续训阶段的评估/保存频率配置，不影响恢复至 step 3。未发现 Jittor core、torch compat 或 CUDA 算子缺陷。

第一次启动因 prep 脚本的全局 `sitecustomize` 在纯 `query_cuda_cc` 子进程中重复导入 Jittor，使它等待父进程持有的 `jittor.lock`；job `108947` 在训练前取消。修正 harness，使 CUDA 架构探测跳过 shim 导入后，job `108965` 完成。该 harness 调整仅保存在外置 prep 目录，未改 Jittor 源码。

本结论只覆盖上述 Qwen2-0.5B、单样本、单卡 FP32 SimPO profile 的新进程恢复与显存测量；不能推出 CPO、其他偏好/RL 算法、模型规模、数据游标、精度、注意力、分布式或全库通过。

文档验证：`bash tools/check_repo_layout.sh` 通过；`tools/build/generate_manifest.py` 更新了根目录 `MANIFEST.in`，`TestPackagingStructure.test_manifest_covers_runtime_trees_without_cache_payloads` 定向复验 `1 passed`。完整 Torch 模式 `tests/structure` 为 `1513 passed, 6 skipped, 1 failed`；唯一失败 `test_source_config_is_executable_without_nox_path_rewriting` 是验证环境没有安装 `asv`（`No module named asv`），与本次文档内容无关。
