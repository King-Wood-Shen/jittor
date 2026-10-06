# 真实 Qwen2-0.5B 双卡 EmbeddingTrainer checkpoint L3

集成基线 `1822468ed05e0f4cd80395471cb58e2d19cb42bc`；ms-swift `88d7279`。全部导入、JIT、训练、数值比较与 checkpoint 内容计算都在 Slurm NVIDIA worker。工作树为独立集成树；原有 30 个修改文件和 2 个未跟踪文件未触碰。

## 合同

缓存的真实 Qwen2-0.5B 通过 ms-swift 公开 `get_model_processor(task_type="embedding")` 加载，使用真实 Qwen 模板与私有 `EmbeddingTrainer`，单机双 RTX 4090、NCCL/DDP、FP32/eager、均匀两负例 InfoNCE。仅 `model.norm.weight` 可训练，先测无动量 SGD，再测 momentum=0.9 的有状态 SGD；每种均比较连续 3 步与 2 步保存 checkpoint 后**新进程**恢复第 3 步。固定数据、初态和 seed，逐 rank 捕获输入、标签、真实模型 embedding、loss、完整可训练梯度及更新。候选两个 rank 均强制 `use_cuda=1`、`backend_fallback="error"` 和 `forbid_backend_fallbacks()`；所有阶段 fallback=0。

## 无动量基线

- 首版原生 10133 的三个训练阶段完成，但验收脚本因 `save_total_limit=1` 在恢复保存 checkpoint-3 时删掉 checkpoint-2，最终文件存在性检查失败。这是实验保留策略，不是模型/恢复失败；修为 2 后重新建独立状态目录。
- v2 原生 10135、候选 10137 均 COMPLETED 0:0。原生连续/拆分/恢复两 rank 的 36+24+12 字段逐值完全一致。10168 在 NVIDIA worker 独立比较：跨后端 72 字段、同后端恢复轨迹 72 字段、24 项由真实 embedding 重建的 FP64 全局 InfoNCE、各步双 rank 参数同步均 PASS；六份候选阶段 JSON fallback=0。连续与恢复候选最大绝对差异 2.98e-8；原生/候选各字段最大绝对差异 1.15e-6。checkpoint-2 有 290 个模型张量及 optimizer/scheduler、两 rank RNG、trainer_state，模型 `norm.weight` 与第 2 步更新一致。
- 原始脚本、两侧每 rank NPZ/JSON、日志、comparison.json 与 Slurm 日志见 `_state/ms-swift-cuda/20261006-qwen2-embedding-ddp-resume-v2`。

## 有状态优化器恢复

- 在完全相同的真实模型/数据/两 rank 合同上仅把 SGD 改为 `momentum=0.9`。原生 10172 COMPLETED 0:0，连续、拆分与新进程恢复三个阶段通过。
- 候选首轮 10173 OUT_OF_MEMORY：提交时遗漏 `--cpus-per-task=8 --mem=64G`，Slurm 默认仅给 8 GiB 主机内存，保存 1.9 GiB 模型 checkpoint 时 cgroup OOM；`MaxRSS=8183832K`，不是 CUDA 显存溢出或产品源码故障。失败日志和部分 checkpoint 移入状态目录 `failed-10173` 保存。改为 64 GiB 后 10174 COMPLETED 0:0，`MaxRSS=22357004K`，三阶段均通过；没有改产品源码或数值门槛。
- 10176 独立 worker 比较：跨后端 72 字段按前向/参数 5e-3、梯度 2e-2 scaled-max 门槛全部 PASS；两侧各自连续 vs 拆分/恢复的 72 字段最大绝对误差 ≤1e-6；24 项独立 FP64 InfoNCE 误差 <1e-5，双 rank 参数同步且六份 fallback=0。checkpoint-2 两侧均有 290 个模型张量、scheduler、trainer_state、两 rank RNG，`global_step=2`，并成功恢复到第 3 步。
- 10187 在 NVIDIA worker 审计真正的优化器状态：两侧 checkpoint 各有一个非零 896 维动量缓冲，动量系数均为 0.9；原生 L2 范数 0.13740958、候选 0.13740964，最大绝对差 8.94e-8，scaled-max 1.41e-6。10180/10181 的探索性审计分别因用原生 `torch.load` 读取候选自有 pickle 格式、及把 Jittor Var 序列化字典当 ndarray 失败；定位格式后用可信自生成 checkpoint 的 pickle 解码，并在 10187 验证数值。两次失败不计产品适配通过证据，日志保留。
- 原始脚本、两侧 NPZ/JSON、checkpoint、comparison.json、optimizer_audit.json、Slurm 日志见 `_state/ms-swift-cuda/20261006-qwen2-embedding-ddp-momentum-resume`。两个 rank 独立 JIT 缓存，10174 复用已结束的 v2 rank 缓存，未并发访问同一缓存。

## 判定与边界

该**限定路径**的真实双卡有状态训练恢复达到 L3：原生基准、严格 NVIDIA CUDA、零 fallback、前向/损失/梯度/参数、两 rank 同步、优化器动量、数据游标与第 3 步恢复轨迹均有证据。它不是整个 ms-swift 的 L3 完成：公开训练 CLI/launcher、全参数或 LoRA、BF16、其他模型/优化器、跨节点、非均匀负例未在本合同验证；原生 torch ZIP 与候选自有 pickle checkpoint 的交叉加载亦未验证。此双卡合同的 L4 公开训练入口与 L5 稳态性能均为 not-run；此前按五轮跳过的独立问题保持跳过。
