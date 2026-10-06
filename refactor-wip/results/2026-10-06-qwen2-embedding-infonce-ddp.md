# 真实 Qwen2-0.5B 双卡 EmbeddingTrainer InfoNCE

运行键：20261006-qwen2-embedding-infonce-ddp。基线 Jittor 032e1e97f17bbcf4284e439ffad1b4e8d4f86c70；ms-swift 88d7279。所有导入、JIT、计算与测试在 Slurm NVIDIA worker；原始脚本、原生/候选每 rank 的 NPZ/JSON、日志和 comparison.json 位于 _state/ms-swift-cuda/20261006-qwen2-embedding-infonce-ddp。独立集成工作树；原有 30 个修改文件和 2 个未跟踪文件保持原样。

## 结果与证据

- 原生基准 10101，cscg-qh06，COMPLETED 0:0：缓存的真实 Qwen2-0.5B 经 ms-swift 公开 get_model_processor(task_type=embedding) 加载；真实 template 和私有 EmbeddingTrainer，FP32/eager、两张 RTX 4090、NCCL/DDP、每组 anchor+positive+2 negatives、仅 model.norm.weight 训练、SGD、3 步。两 rank 各保存 18 字段。
- 候选首轮 10105，cscg-qh06，FAILED 1:0：严格 CUDA 已初始化 NCCL，但 Accelerate 即使 comm_hook=NO 也无条件导入 torch.distributed.algorithms.ddp_comm_hooks；shim 缺少该模块，两个 rank 均在 DDP 包装时失败，未进入训练。历史 5852 的短用例未使用实际 DDP wrapper，因此不覆盖此断点。
- Torch 兼容层现发布 ddp_comm_hooks/default_hooks/powerSGD_hook 导入图。默认无 hook 路径保持现有 Jittor DDP 广播与梯度平均；请求 FP16/BF16 压缩或 PowerSGD 时显式 NotImplementedError，避免静默改变训练语义。
- 候选第二轮 10118，cscg-qh15，两个 rank 的真实 Trainer 均完成 3 步且 fallback=0；作业退出 1:0 是首版验收脚本把每步全局 batch 固定为 4 组，遇到动态尾批后公式形状断言失败。只修改状态目录中的验收脚本，产品代码和训练产物未变。
- 独立复核 10123 首次仍因统一按 2 组计算而失败；检查四份原始 NPZ 后确认每 rank 每步句向量数为 8、4、8，全局组数为 4、2、4。复核 10126 在 cscg-qh15 COMPLETED 0:0：按每步实际组数构造全局 Q-D 温度 0.1 交叉熵，原生与候选 12 项 FP64 公式误差均 <1e-5；逐 rank 输入、标签、句向量、损失、全部可训练梯度与更新共 36 字段按前向 5e-3、梯度 2e-2 容差全部通过；两 rank 参数更新一致，fallback 均为 0。比较结果记录 failures=[]。
- 定向回归 10127 的非默认通信 hook 导入与显式拒绝检查已通过；其余 pytest 收集失败是从仓库根目录收集 compat/tests 时把 jittor.compat 错误识别为顶层 compat，未产生该项测试结果。改为独立结构回归 10128，cscg-qh15 COMPLETED 0:0：模块导入与 fail-fast 再次通过，结构/API 覆盖测试 38 passed、40 subtests passed。10127 不能记为 pytest 通过。

## 验收边界

此真实双卡合同达到 L0 模型/Trainer/DDP 构造、L1 前向和 L2 三步反向与更新；原生与候选均在 NVIDIA CUDA，候选严格 use_cuda=1、backend_fallback=error，两个 rank fallback=0。仅覆盖 FP32/eager、均匀两负例、默认无通信 hook、单机双卡、只训练 model.norm.weight。完整状态恢复 L3、公开 swift 训练 CLI/launcher L4、预热后稳态性能 L5，以及全参数/LoRA、BF16、非均匀负例和非默认通信 hook 均为 not-run 或既有独立失败范围。10118 的约 393 秒 train_runtime 含首次真实图 JIT，不能作为 L5 对比。旧非均匀负例完整 Trainer 五轮失败、packing 五轮失败、TinyLlama streaming logprob、GKD BF16 和 RewardTrainer BF16 跳过状态未改变。
