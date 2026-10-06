# ms-swift Qwen2 InfoNCE QQ+DD CUDA 适配证据（2026-10-06）

基线：`integration/ms-swift-cuda-upstream-20261006`，起点 `2576e6e90`；ms-swift 使用 `/home/xinshen/projects/ms-swift-cuda/ms-swift`。本轮仅在干净集成 worktree 修改 Jittor；原 `ms-swift-cuda-3008-d99` 的 30 个已修改文件和 2 个未跟踪文件未动。未 push。

问题与改动：ms-swift 的 InfoNCE QQ 分支在 `qq_matrix.fill_diagonal_(-inf)` 因缺少 Tensor API 中断。新增支持 2D、高维等长、矩形 `wrap`、原对象返回和梯度传播的 `fill_diagonal_`。干净基线加载真实 checkpoint 时另在 `torch.empty(device="meta")` 中断，故仅迁入元数据占位所需的三处 `meta` 设备标识处理；未搬运其他脏补丁，也未改 ms-swift 源码。

验证：所有导入、JIT、测试和数值计算均在 Slurm `s1` 的 NVIDIA RTX 4090 worker 执行，`JITTOR_HOME` 独占且作业串行。
- 9452 原生 PyTorch CUDA 基准确认形状、`wrap`、返回对象与无效维度错误；9478 又确认空列矩阵 `wrap=True`。
- 9464 和 9478：`jittor.compat.tests.torch.test_fill_diagonal` 3 项通过，严格 CUDA、`forbid_backend_fallbacks`、fallback 计数 0；覆盖值、身份、错误与非叶梯度。
- 9469：真实缓存 Qwen2-0.5B 经 ms-swift 公共 `get_model_processor` 加载，私有 `EmbeddingTrainer` 的 `INFONCE_INCLUDE_QQ=True`、`INFONCE_INCLUDE_DD=True`，FP32/eager、两组均匀双负例、只训练 `model.norm.weight`，三步 SGD。原生先跑、严格 CUDA 候选后跑；输入、标签、真实 embedding、loss、完整可训练梯度及参数共 18 字段对齐。独立 FP64 从 embedding 重建 QD+QQ+DD 交叉熵，双方六项通过，最大公式误差 1.60e-6；`global_step=3`，候选 fallback 0。Slurm 9469 COMPLETED 0:0。
- 9478：布局检查通过；结构测试 1388 passed、6 skipped、1031 subtests passed；`git diff --check` 通过。6 个 skipped 按原测试分类，不能算通过。

原始脚本、日志、JSON、NPZ 与 Slurm 产物：`/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261006-fill-diagonal-qq`。首次真实基准 9467 暴露干净分支缺失 `meta` 占位，原生完成、候选在 checkpoint 元数据加载前失败；保留于 `qwen/`。成功轮在 `qwen-v2/`。

范围：仅证明上述固定 FP32 Qwen2 embedding QQ+DD 三步路径。无全参数/LoRA、BF16、checkpoint 恢复、分布式或十次稳态性能结论。默认 fused AdamW 的独立 FP64 二阶矩合同仍未通过；Reward BF16、TinyLlama streaming logprob、GKD BF16 累积和 packing 多进程仍维持此前失败/跳过状态。其他模型权重、多模态和服务面仍未覆盖，不能宣称整个 ms-swift 已兼容。
