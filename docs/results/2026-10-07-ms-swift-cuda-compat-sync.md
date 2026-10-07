# ms-swift CUDA 适配：基线同步与长上下文流式断点

- 状态：历史验证保留；新基线仅完成所述诊断；长上下文 Python 流式默认路径 L4/L5 未通过。
- 日期：2026-10-07。
- 基线：历史集成 `fb23d894ef3eb85d947691c7a85debd6615429f1`；同步上游 `ffeb7bd80447ca78735e7cb96628ab88c1e9360b`；ms-swift `88d7279`。
- 验证范围：真实缓存 Qwen2-0.5B、原生 PyTorch CUDA 与 Jittor torch shim；本报告新增实验覆盖约 1.3k-token 的公开 Python API 流式生成。
- 维护者：ms-swift CUDA 适配。
- 复查条件：修复 Core 自动 flush 相关存活管理并在当前基线完成原生/候选逐事件对拍与两次预热加 10 次同步稳态。

上游已删除 `refactor-wip/`。本分支先前 42 份报告保留在 Git 提交 `fb23d894e` 的 `refactor-wip/results/`，原始日志与模型仍在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/`。例如 `git show fb23d894e:refactor-wip/results/2026-10-07-qwen2-public-infer-1k-context.md` 可查公开 CLI 千 token 非流式的原生对拍；其他报告涵盖 Qwen2 基座、embedding、分类、tuner、双卡训练、恢复和公开推理。历史 PASS 均只属于各自原 SHA；同步后这些功能面记为 **not-run**，需逐面复验。

## 长上下文流式结果

运行键 `20261007-qwen2-python-stream-1k-l5`，真实缓存 checkpoint，单张 RTX 4090，FP32/eager，greedy 最多 32 新 token。原生作业 10886 在独立进程完成两次预热与 10 次同步稳态；候选作业 10886 第一轮预热后段错误，没有候选稳态结果。

在旧基线，最小两请求作业 10909 在首条流式迭代中段错误。符号化堆栈到达 `Node::release_both_liveness()`，调用链含 `BatchState` 析构及 `Executor::run_sync/auto_flush/submit_pending`。10918 关闭异步执行后仍复现；10920 禁用 `auto_flush_ops` 与 `async_flush_ops` 后两次请求完成、正常退出、fallback=0。证据将故障范围收窄到自动 flush 相关路径，但未证明首先破坏存活计数的算子，也未证明禁用 flush 时的性能。

同步上游后的新 JIT 缓存作业 10931，在默认配置下两次请求均完成且 fallback=0；这只是短诊断。随后的完整性能作业 10951 在同一 GPU 上第一轮预热得到 1396 prompt tokens、17 completion tokens 后段错误，未留下候选结果或正常退出标记。新基线仍不能给长上下文流式 L4/L5 记 PASS，也不能报告原生/候选性能比；候选只有启动时的严格 CUDA、零 fallback 证据，崩溃前缺少结束审计。

作业 10930 的仓库布局检查通过；全量结构测试先得 1384 passed、6 skipped、2 failed，失败仅因本报告当时未列入文档索引及生成清单；修正后作业 10950 定向复验 6 passed。未跑新基线的 core/smoke 或模型训练数值门禁，均记 **not-run**。本问题已停止重试；详见 `KI-EXEC-011`。独立入口继续推进，不以禁用自动 flush 的诊断结果冒充默认适配。

未版本化证据：`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-python-stream-1k-l5/` 中的 `native-result.json`、`shim.log`、`repro.log`、`sync.log`、`no-flush.log`、`merged.log`、`shim-merged.log`、`merged-perf-bootstrap.log`，以及 `20261007-merge-ffeb-verify/` 的结构测试日志。
