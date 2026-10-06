# 真实 Qwen2 公开 Embedding SFT 双卡入口：序列化修复与未通过的前向

基线：Jittor 集成工作树 `4c96ab38330c4aa590f1a1c50d083696625a54fd` 起改，ms-swift `88d7279`，Python 3.11.15；真实缓存 Qwen2-0.5B，4 条离线不同组样本，float32 权重加载/eager、CLI fp16=True（非纯 FP32 训练），仅 `model.norm.weight` 可训练，SGD 0.01、三步、每卡 batch 2、NCCL 双 RTX 4090。原始命令、日志、输入/embedding NPZ、比较 JSON 和 checkpoint 均在 `_state/ms-swift-cuda/20261006-qwen2-public-embedding-ddp-cli`。Slurm 节点 cscg-qh15，两卡 UUID 为 `GPU-afd56a2e-3deb-3bd7-f122-1075e6de8956` 和 `GPU-6005c074-c2a4-a62b-e0b0-8b090cec31c9`；无 Ascend 或登录节点计算。

## 可确认结果

- 原生 10285 `COMPLETED 0`；候选首次 10288/10297 因 Jittor JIT 核心要求进程重启而未进入训练，10289 在 NVIDIA worker 串行预热四个角色。10301 暴露 shim 没有 `torch.distributed.run`，因此完整 shim launcher 入口**未通过**。
- 为隔离训练子进程，后续实验只让原生 `torchrun` 负责控制面，两名子进程仍经 Jittor shim、严格 CUDA 和独立 `JITTOR_HOME`。10310 首轮编译后，发现 Jittor NCCL Store 与 torchrun 监听端口 29500 冲突；实验 bootstrap 给 Jittor rank 另设相同的作业内端口。这个控制面替换与端口设置属于实验条件，不能称 shim 自身 `torch.distributed.run` 已兼容。
- 10315 在同一公开 `python -m swift.cli.main sft` 入口完成三步，但 rank 1 保存 `rng_state_1.pth` 时 `torch.save` 的异步 `jt.fetch` 回调在 `jt.sync_all(True)` 后仍未完成，随后出现内存错误。10321 用无需模型的真实双卡 NCCL 探针复现：rank 0 保存成功，rank 1 报 `checkpoint tensor fetch did not complete` 并 SIGABRT。10322 仅在实验中把抓取改为 `jt.fetch_sync`，两 rank 连续三次保存/加载和 fallback0 均通过。
- 产品修复在 `compat/torch/serialization/portable.py` 对活跃张量的克隆用同步 `jt.fetch_sync` 完成快照，不搬动活跃张量。最初直接同步抓取源张量虽解决双卡回调，但 10352 广义序列化回归发现其把源张量迁到 CPU，已改为抓取克隆；10357 同一 CUDA owner 回归 10/10 PASS、fallback0，10358 双卡探针两 rank 各三次保存/加载 PASS、fallback0；10361 克隆版定向回归 2/2 PASS 且双 rank 各三次保存/加载 PASS，fallback0。10326：新 CUDA 混合 RNG/张量回归和既有字典保存回归 2/2 PASS，随后原实现的双 rank 探针三次保存/加载全部 PASS，零 fallback。10327 再跑完整公开训练，`COMPLETED 0`、三步、两 rank strict CUDA 与 shim 标记真、起止 fallback0，`checkpoint-3` 有模型、optimizer、scheduler、trainer state 与双 rank RNG 文件。首次 JIT 不计性能。
- 数值验收仍失败。10329 独立 worker 比较：原生三步 loss 为 `2.03965187/2.03964806/2.03964424`，候选为 `2.03161693/2.03161240/2.03160906`，最大绝对差 `0.00803566`，超过固定前向阈值 `0.005`；grad norm 最大相对差 `0.01857`，最终 896 维 `model.norm.weight` 最大差 `5.722e-6`，两侧均发生非零更新、290 个模型状态键一致、global step 均为 3。不能用最终参数接近掩盖前向失败。
- 五轮定位后停止该数值问题：10335/10338 证明四条预处理样本全局集合相同，但两侧 rank 分配整组对调；10340 仅用于诊断的行序对照让 rank 分配一致，loss 仍为 `2.03161669`；10341/10343 保存并独立 FP64 复核实际 embedding，标签完全相同，InfoNCE 公式分别复现两侧 loss，而模型 embedding 相对 L2 差为 rank 0 `0.008377`、rank 1 `0.008871`；10345/10348 核对同 rank 的 `input_ids`、`attention_mask`、`labels` 形状、dtype 与每个数值完全一致。首个已证实的分歧位于相同模型输入之后、InfoNCE 公式之前的 Qwen 输出 embedding；权重加载状态与具体首个模型层尚未分层核验，不能臆称单一算子根因。诊断行序对照不作为正式同数据运行的通过证据。

## 验收边界与后续

通用 Torch 保存/加载的上述同步修复在真实双卡和严格 CUDA 下通过。公开双卡 ms-swift 入口的 L0 构造与训练/保存可执行性已证实，但 L1 前向未过固定阈值，故 L2–L5 均标记 `blocked`，不能因 10327 退出 0 宣称完整双卡 L4；shim 原生 launcher 模块本身也未实现。双卡 checkpoint 的存在不等于同进程或新进程恢复对齐；该公开 CLI 恢复为 `not-run`。本轮没有 L5 稳态性能结论。

解除数值阻塞时应先在同一公开 CLI 入口、相同 collator 输出和相同初始模型状态下逐层比较 Qwen hidden states 与关键权重，定位第一层误差；这是后续独立任务，当前数值项已到五轮上限。原始脏工作树的 30 个修改与 2 个未跟踪文件未动，独立集成工作树仅改上述 Torch 序列化文件、定向回归及本报告。完整 ms-swift 功能面矩阵仍未完成。
