# ms-swift checkpoint 生命周期与 L3 容量

- 状态：`msw-storage-v2` 与恢复路径回归通过；单个已测 checkpoint 尺寸的存储准备就绪。完整 L3 profile 的并存产物峰值仍须由启动作业按实际计划声明；本报告不授予 L3 数值覆盖。
- 日期：2026-10-15。
- 同步基线：`origin/2.0-refactor` `0751e425b0f197b8e7b388f218613378c71f701a`；合并提交 `3ca897bfb16ea35349f5223622d9fd0454f3e239`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：变更 checkpoint 保存格式、保留阶段、NFS 配额或 `msw-storage-v2` 的预算公式时。

## 占用审计与安全释放

`20261015-msw-checkpoint-capacity-v1` 的 Slurm job 26245 已完成（exit 0）。它把 GRPO 键
`20261014-qwen2-grpo-trl026-full-v2/native-output/checkpoint-3` 复制到
`full-checkpoint-copy/`，专门测量完整 checkpoint 的复制成本。两目录 `diff -qr` 完全一致，
均无符号链接；源目录的 16 个文件逐项 SHA-256 校验通过。源 checkpoint 被保留，删除的只有容量探针副本。

删除前副本逻辑大小 `1,992,113,448` bytes，NFS 已分配大小 `1,651,510,272` bytes（`1.54 GiB`）。
逐文件大小、SHA-256、作业状态和恢复方法保存在外置 state：
`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261015-msw-checkpoint-capacity-v1/`，其中
`redundant-copy.inventory.tsv`、`redundant-copy.sha256`、`redundant-copy.jobs.txt`、
`redundant-copy.recovery.txt` 组成清单。源 checkpoint 的 SHA 清单复验通过。

清理前一次观测的 home 可用量为 `54,000,615,424` bytes；NFS 更新后为
`55,647,928,320` bytes，净增加 `1,647,312,896` bytes（`1.53 GiB`）。观测窗口内其他
作业仍在写 home，因此净增量低于副本块占用。随后 worker/cache 作业继续使用该卷；本次最终快照可用量为
`47,493,152,768` bytes（`44.23 GiB`）。未把登录节点
或 worker `/tmp` 当作持久备份。

该 probe 的 checkpoint 完整逻辑大小约 `1.86 GiB`。一次持久发布需一份 staging 副本加
固定 `2 GiB` 保留，即约 `3.86 GiB`；最终快照满足这一已测尺寸，扣除该容量后约余
`40.37 GiB`。因此一个同尺寸 checkpoint 的保存与新进程恢复可以开跑。完整 L3 profile 的
实际需求须按同时保留的 checkpoint、日志/原始张量、训练输出和 inode 声明峰值重新预检；当前
没有该 profile 的完整峰值测量，故不能据此批准多份大 checkpoint 并行写入或宣称完整 L3
存储预算已验收。

## 产物生命周期改动

旧预算把声明峰值乘二再加 `2 GiB`，但持久 checkpoint 的源在节点 scratch，home 实际只需
容纳一个 staging 副本和保留量。`msw-storage-v2` 改为检查声明峰值加 `2 GiB`，调用者需把
同一时刻会共存的所有新 checkpoint 与 staging 计入峰值。持久卷中已经通过 receipt/SHA 验证的
相同内容以硬链接复用；receipt 记录每个复用文件和实际新增字节。发布仍不可变、逐文件验哈希、
fsync 后原子落地；不自动删除任何来源或失败证据。

## 验证与同步回归

- Slurm job `26458`（RTX 4090）：storage lifecycle 单测 `10/10` 通过；仓库布局检查通过；全量
  structure `1501 passed, 6 skipped, 4 failed`。其中 docs 可达性和 manifest 覆盖在 job `26489`
  定向复验 `2/2` 通过；剩余失败为缺少可选 `asv` 依赖和既有跨文件重复 helper 检查。
- Slurm job `26499`（RTX 4090）：CPU RNG snapshot 定向回归 `2/2` 通过。
- Slurm job `26544`（RTX 4090，HEAD 为合并基线 `3ca897b`，工作树含本次两项运行时修复）：按规定 core tier 完成，native `98 passed,
  41 skipped, 1 xfailed`，Torch `225 passed, 1 skipped`；合计 `323 passed, 42 skipped, 1 xfailed`。
  它同时复验了合并时被远端实现覆盖的 `Tensor.backward` create-graph 上下文：在保留远端
  autograd 实现基础上补回最小上下文传播后，原 4 个 Torch core 失败消失。
- 结构 suite 第一次揭示的 `docs/results/index.md` 历史页面入口和生成 manifest 已修复，且已由
  targeted worker gate 覆盖；未因这两个目录/文档失败修改测试规避行为。

结果目录入口与 ms-swift Skill 已更新。CUDA L3 覆盖矩阵保持原值：已有固定 profile 的结果仍由其
各自报告负责；本次只提高存储准备度，不将 storage probe 或 core tier 计作 L3。
