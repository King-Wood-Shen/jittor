# 真实 Qwen2 公开 seq_cls SFT CLI：固定分类头的纯 FP32 数值对齐

基线 Jittor `abb06adf23e1a753f33e48565484ee18f233cd70`、ms-swift `88d7279`。全部导入、JIT、checkpoint 处理、训练和数值比较在 Slurm NVIDIA worker。入口为 `python -m swift.cli.main sft`，真实缓存 Qwen2-0.5B 基座、四条离线三分类消息整批、`--task_type seq_cls --num_labels 3`、显式 `--torch_dtype float32 --fp16 false --bf16 false`、eager、full tuner 冻结基座只训练 `score.weight`，SGD 0.01 三步。原始脚本、日志、checkpoint、比较与 fixture 构造位于 `_state/ms-swift-cuda/20261007-qwen2-public-seqcls-cli-fp32`。

首轮原生10471、候选10472均完成公共入口，但新分类头 `score.weight` 不在原始因果模型 checkpoint 中，两侧独立随机初始化。首步 loss 分别6.5576/2.2469，初值不可比较；10485比较失败并保存原始差异。源码核查表明 `--init_strategy zero` 仅对 NaN/Inf 或极端数值的参数生效，不会清零有限的随机头，不能用该选项宣称固定初值。

为隔离模型算子，10489在worker建立分片测试checkpoint：290个原始基座权重仍从真实缓存 `model.safetensors` 软链读取，仅新增明确的 3×896 零分类头 `score.safetensors` 与索引。10490发现单文件优先于索引，因此仍沿用随机头；改名基座分片后10493原生首步 loss=ln(3)，证实固定头生效。10494候选被测试分片缺少 `format=pt` 元数据阻断；10496补元数据时重建索引导致10497引用缺失旧文件，修正索引后10498候选成功。以上均为测试产物/装载协议调整，未改产品源码或原缓存。

固定头原生10493、严格 CUDA候选10498与独立比较10499均 `COMPLETED 0`。三步loss最大绝对差1.1921e-7，grad norm最大相对差4.5104e-7，acc三步精确相同；最终 `score.weight` 最大绝对差1.1176e-8，L2分别0.0177102462/0.0177102458，证明非零更新。两侧291个模型键、global_step3及optimizer/scheduler/RNG checkpoint文件存在；候选CLI父子进程bootstrap起止均strict CUDA、shim marker真、fallback0。

判定：真实 Qwen2 基座加显式确定性分类头的公开单卡seq_cls SFT纯FP32三步限定L4数值合同PASS。它不证明随机头默认初始化一致、模型分类质量、全参数/LoRA、混合精度、其他模型、双卡、状态恢复或L5；这些保持not-run/未通过，不宣称完整适配完成。
