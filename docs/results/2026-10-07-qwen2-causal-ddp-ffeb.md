# Qwen2-0.5B 公开因果 SFT：双卡全参数新基线复验

- 状态：固定 FP32/SGD 配置的公开 CLI 双卡三步数值诊断完成；严格 L0–L5 尚未逐级验收。
- 日期：2026-10-07。
- 基线：Jittor `37591b485`（代码包含上游 `ffeb7bd80`）、ms-swift `88d727951`。Jittor 工作树干净；ms-swift 有既存未跟踪 `model_arch.jsonl`，本实验未使用或修改。
- 范围：真实 Qwen2-0.5B、公开 `python -m swift.cli.main sft`、两张 RTX 4090、`NPROC_PER_NODE=2`、全参数 FP32、eager attention、SGD 无动量、每卡 batch 2、三步、固定四条数据、最大长度 64、学习率 `1e-5`。候选采用原生 `torch.distributed.run` 控制平面启动 Jittor 双 rank，不能据此宣称纯 Jittor launcher 兼容。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch shim、NCCL/Jittor 分布式、ms-swift 公开训练入口或 CUDA executor 变更；补逐参数梯度、双卡恢复、纯 Jittor launcher 与 L5 时。

模型 `model.safetensors` SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；数据 JSONL SHA256 为 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。隔离 Python 3.11.15、原生 PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、Accelerate 1.10.1。实际 NCCL include 为 2.21.5，`libnccl.so` 指向隔离环境的库文件；两个 rank 分别使用独立 `JITTOR_HOME`。模型、数据、缓存、命令、日志和比较 JSON 均未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-ddp-ffeb/`。

Slurm 11114 的独立原生 PyTorch CUDA oracle 与 11116 的严格候选依次在同一 worker 和同两张 GPU 完成公开 CLI 三步，均保存 checkpoint-3、optimizer、scheduler 及 `rng_state_0.pth`/`rng_state_1.pth`。rank 0/1 的 GPU UUID 分别为 `GPU-802213e4-8931-05eb-d8d4-071c591407d8`、`GPU-deaf0c66-ef84-f611-00d3-dd81224c380d`。候选主进程和两个 rank 启动、退出均记录 `use_cuda=1`、shim 标记真、fallback 0；rank 上的 `RANK/WORLD_SIZE/LOCAL_RANK` 与 `JT_NCCL_RANK/JT_NCCL_WORLD_SIZE` 为 `0或1/2/0或1` 与 `0或1/2`。11129 在 worker 上独立比较：三步 loss 每步绝对差 `2.38419e-7`，梯度范数最大相对差 `1.71095e-6`，token accuracy 全相同；最终 290 个模型键最大绝对差 `7.45058e-9`、整体差 L2 `5.50777e-8`，两侧各有 246 个张量发生 FP32 可见更新，更新 L2 分别为 `0.001200761541` 与 `0.001200761587`。

| 层 | 本配置状态 | 缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型、tokenizer、数据、trainer 与双 rank 构造成功；初始状态键、逐项 dtype/设备未直接审计。 |
| L1 | not-run | loss 与 token accuracy 来自训练日志；同权重同输入的 logits/hidden 逐张量前向对拍未运行。 |
| L2 | blocked | 三步 loss、梯度范数和最终权重贴近，但全部 trainable 参数的逐项梯度与每步 optimizer 状态/更新未直接比较。输入为离散 token ID，不适用输入梯度。 |
| L3 | blocked | 双 rank checkpoint 与 RNG 文件存在；同进程及新进程恢复后的模型、optimizer、scheduler、RNG、数据游标和后续轨迹未运行。 |
| L4 | blocked | 两侧确已通过公开 CLI 和原生 torchrun 控制平面端到端完成；前级验收尚未全部通过。纯 Jittor launcher 未运行。 |
| L5 | blocked | 前级未通过；未执行两次预热及十次同步稳态双卡性能协议。 |

最初在另一节点运行的原生双卡尝试 11107/11108 于模型构造前失败；独立设备探针 11110 证明该节点的 PyTorch `device_count()` 报 2，但查询第二张卡属性触发 `CUDAContext.cpp` 的 `device=1, num_gpus=1` 断言。11113 在最终 worker 的两卡探针通过后才运行 11114/11116。候选与原生数据预处理均打印 NFS 临时文件清理警告，未阻止训练。首次 JIT 编译与原生训练耗时不构成 L5 性能数据；历史基线的双卡 PASS 不自动继承到此配置之外。
