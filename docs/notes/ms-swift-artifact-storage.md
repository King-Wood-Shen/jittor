# ms-swift 新实验存储合同

状态：单节点 Slurm 存储链路；完整 CUDA 训练仍按 profile 单独验收。存储策略版本为 `msw-storage-v1`。多节点在 rank 汇集实现并验证前拒绝初始化。

新运行在 worker 调用 `tools/ms_swift_storage_worker.sh driver.py`。调用前设置绝对路径 `MSW_PYTHON`、全新 `MSW_EVIDENCE_ROOT`、`MSW_RETENTION_PLAN`，以及 `MSW_SCRATCH_PEAK_BYTES`、`MSW_DURABLE_PEAK_BYTES`、`MSW_ESTIMATED_FILES`、`MSW_RAM_RESERVE_BYTES`（训练驻留内存峰值，不含 scratch）。运行键不得复用。HOME 配额须按 skill 核实；工具的容量检查使用保守挂载余量，不能代替配额检查。

| 写入 | 路径 | 保留合同 |
| --- | --- | --- |
| 完整梯度、逐步更新、大 logits/hidden | `$MSW_WORK_ROOT/raw/<backend>/<rank>/<step>/...` | 节点临时；比较器通过 manifest 定位并重新核验 SHA-256 |
| 临时模型、Trainer 所有工作 checkpoint | `$MSW_MODEL_ROOT`、`$MSW_TRAINER_ROOT` | 节点临时；CLI `--output_dir` 必须显式使用这里 |
| 命令、版本、输入身份、轻量统计、日志、比较结论 | `$MSW_EVIDENCE_ROOT` | 持久；不能把包含原始张量的 NPZ 当作轻量证据 |
| 必要 checkpoint | `$MSW_EVIDENCE_ROOT/checkpoints/<name>` | 阶段完成立即校验、发布；不等待整个作业 EXIT |
| 已验证 JIT 缓存、已有模型/oracle | 原固定路径 | 保持路径、指纹和串行锁约定；输入只读 |

`TMPDIR` 是 `/tmp/msw-<uid>-<job>/t` 实目录。初始化在真实节点核对 findmnt、独立 filesystem、空间、inode 和 fsync；拒绝指回 HOME 的别名和重复运行键。临时目录受站点的 10 天老化清理规则约束，这不是保证保留十天。原始数据失效应记 `raw_evidence_expired`，不能宣称仍可逐值复核。

节点可能使用内存支撑的 rootfs/tmpfs/ramfs。此时同时取 MemAvailable、Slurm 作业内存请求及可见 cgroup 内存剩余的最小值，再扣除训练驻留内存，检查 scratch 峰值和余量。容量不足直接拒绝；应调整 Slurm 内存或使用经过验证的节点本地磁盘。每阶段前调用 `Storage.refresh()` 或 CLI `check` 刷新预算。

Python driver 使用 `importlib.util.spec_from_file_location` 加载 `MSW_STORAGE_TOOL`，然后 `Storage(MSW_EVIDENCE_ROOT)`。`raw_path(relative)` 提供写入位置；关闭写入后 `record(relative, kind, shape, dtype)` 保存完整 SHA-256 和定位；逐值比较前 `resolve(relative)` 校验并返回文件。仍要执行真实逐值比较，哈希不能代替数值验收。小输入和统计自行用 `atomic_json` 持久写入。

必要 checkpoint 必须预先声明理由、角色和完整文件契约，例如：

```json
{
  "native-restore": {
    "role": "recovery",
    "reason": "原生新进程恢复起点",
    "required": ["model.safetensors", "config.json", "tokenizer_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]
  }
}
```

该列表是示例，实际合同按全模型/adapter、保存格式及拓扑展开。分片权重必须逐个列出，不能只检查 index；双卡逐个列出 `rng_state_0.pth`、`rng_state_1.pth`。需要恢复的 optimizer/scheduler/RNG、buffer、训练游标和 tokenizer 必须完整。预先列出独立 oracle、恢复起点、被复用中间点和各场景验证终点；Trainer rotation 不得删除待恢复或待发布来源。没有列入计划的 checkpoint 留在临时目录。

所有 rank 完成保存、writer 关闭并经过 barrier 后，由 rank 0 调用 `publish(name, source)`。工具拒绝缺项、链接、特殊文件和源目录变化，复制到持久 staging，核验每个文件 SHA-256、fsync，再原子发布。发布目录携带 receipt，因此进程在发布后、更新总索引前被杀，也可通过 `verify_checkpoint(name)` 核验。发布过程中被杀的 `.publish-*` 仅是未完成证据，不能当作 checkpoint。工具不删除任何模型来源。

恢复阶段通过 `verify_checkpoint(name)` 获得持久目录，再执行实际的同进程及新进程训练恢复验收。哈希链路通过仅证明文件完整，不能证明优化器或数据游标语义正确。原生阶段完成就发布其必要终点，后续 shim 失败不影响已经持久保存的 oracle。

默认原生、shim、逐值比较在同一节点同一作业完成。跨作业传递原始张量必须另行声明 producer node 和显式 stage-in，并核验哈希；本版本直接拒绝其他 producer job/node。不要建立 HOME 到节点 `/tmp` 的永久软链接。

启用范围是本规则生效后生成的全新实验。历史脚本、已提交队列、运行中作业和已有结果保持原协议；不得对它们替换输出根、修改 plugin/comparator 或补跑旧运行键。
