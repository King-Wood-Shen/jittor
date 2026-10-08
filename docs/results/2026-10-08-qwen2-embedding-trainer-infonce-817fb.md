# Qwen2-0.5B ms-swift EmbeddingTrainer InfoNCE CUDA 基线复验

- 状态：固定 Qwen2-0.5B/FP32/eager 配置的 EmbeddingTrainer 直接 Python API 在新 Jittor 基线上 L0-L2、L4 通过；不代表 `swift sft` CLI、其他 embedding 模型或整个 ms-swift。
- 日期：2026-10-08。
- Jittor 基线：集成 HEAD `817fb1e6e0ebe91e709b19586cbdc6bbaa1da819`；上游 `2.0-refactor` SSH 查询 SHA `7a18abf295668d9b19da5fa1657f5606e84b65a0`，为 HEAD 祖先。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：EmbeddingTrainer/InfoNCE、Qwen2 embedding pooling 或 torch shim CUDA/autograd 改动时；需扩大到 CLI、checkpoint 恢复或稳态性能时另立验收。

运行键 `20261008-qwen2-embedding-trainer-infonce-817fb-v1`。Slurm 13276 在 cscg-qh13 RTX 4090（UUID `GPU-5331cb2a-7ec6-ad9a-5890-cb43fd723398`，driver `580.178.04`）先跑独立原生 PyTorch CUDA，再跑严格 shim。模型由当前 ms-swift `get_model_processor(..., task_type='embedding')` 从本地 checkpoint 构造，FP32/eager；使用 Qwen template、max length 32、固定 6 行合成的离线 anchor/positive/negative token 数据，batch 2、InfoNCE、温度 0.1，seed 42，3 步。仅 `model.norm.weight` 一个参数可训练，SGD lr `1e-5`，无梯度裁剪。两侧 290 个参数的名称、shape、dtype、device 与 requires-grad 状态一致；初始 norm 权重也逐值一致。样本 ID、attention mask、labels 全部从两侧训练步捕获并逐值相等。

每一步比较了输入、mask、embedding `last_hidden_state`、labels、loss、trainable 参数梯度和更新后参数，共 22 项（含初始 trainable 权重）。loss 轨迹 native `[1.71876144, 2.05013180, 1.80068803]`、shim `[1.71876359, 2.05013156, 1.80068922]`；全部比较张量最大绝对差 `2.14577e-6`，最坏相对 L2 `3.00453e-6`，零项超出前向 `5e-3`/反向 `2e-2` 门槛。另用保存的 embedding 独立计算 InfoNCE cross entropy，native 与 shim 每步误差均小于 `5e-7`。候选所有 290 个参数、输入、embedding、loss 与梯度都在 CUDA；`use_cuda=1` 且 `fallback_count=0`。

原生 trainer 报告训练 3 步用时 `0.663 s`，shim 报告 `171.707 s`；该差异包含本轮 JIT 首次编译，且没有两次预热加 10 次同步稳态，因此不构成 L5 性能结论。运行时脚本、22 项 NPZ、比较 JSON 与日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-embedding-trainer-infonce-817fb-v1/`。2026-10-04 同配置的旧运行来自旧基线，仅作为脚本/测试历史，不继承为本次通过证据。

文档布局检查通过；Slurm 13280 的包装结构定向门禁 `tests/structure/test_packaging_structure.py` 为 8 passed、8 subtests passed。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS（固定 trainer 构造） | 两侧构造真实 Qwen2 embedding 模型、processor、template、离线 dataset、EmbeddingTrainer 与 SGD；290 参数 metadata 一致且全在 CUDA。 |
| L1 | PASS | 3 步输入/mask、embedding 输出、InfoNCE loss 和独立交叉熵公式对齐，均有限。 |
| L2 | PASS（完整可训练参数集合） | 本配置唯一 trainable 参数的三步梯度与更新权重逐值对拍；token ID 输入不求梯度。 |
| L3 | not-run | 未保存/恢复模型、optimizer、scheduler、RNG 或 dataloader 游标。 |
| L4 | PASS（Python Trainer API） | 端到端调用 ms-swift `EmbeddingTrainer.train()` 完成固定数据三步；不是 `swift sft` CLI/launcher。 |
| L5 | not-run | 无两次预热和 10 次稳态同步计时；首次 JIT 编译影响运行时间，不能报性能比。 |
