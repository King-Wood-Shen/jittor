# Qwen2 公开因果 SFT CLI 全参数 FP32 AdamW：数值对齐与间歇反向故障

Jittor基线`a0016efc469bdd8cae10683a2a3f710a913a0904`、ms-swift`88d7279`。仅Slurm NVIDIA worker执行导入、JIT、训练和比较。固定真实缓存Qwen2-0.5B、四条离线对话整批、公开`python -m swift.cli.main sft`、eager、显式`float32`且关闭fp16/bf16、`--tuner_type full`共494.0328M可训练参数、`adamw_torch` lr1e-5/weight_decay0/constant scheduler、三步。脚本、日志、checkpoint、全权重/optimizer比较及调查笔记位于`_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-adamw-fp32`。

原生10574在cscg-qh09完成三步与checkpoint-3；Slurm仅因实验脚本误查checkpoint-1而FAILED1，已改正验收路径且未重复运行原生。候选10577在cscg-qh17 GPU-b0072d68完成前两步、第三步`jt.core.grad_optional`触发`nano_vector.h:41 slice overflow: 94270235702803 0 1`，未保存checkpoint。该数值形似地址，但没有C++调用栈或最小复现，不能断定产生无效Slice的具体操作。候选10581在cscg-qh09、10583与10598在同一cscg-qh17顺序运行均三步`COMPLETED0`、保存checkpoint；三次父子进程严格CUDA、shim marker真、fallback0。同节点成功表明故障并非该节点必现。

10582/10584对10581的完成运行比较：三步loss最大差4.292e-6、grad norm最大相对差9.324e-6，最终290权重最大差2.058e-6、全体差L2=2.726e-5，双方相对真实基座更新L2约0.365981、290张量均有FP32可见变化；290 optimizer state、494032768元素，step3，一阶矩最大差3.201e-7/相对L2=1.679e-5，二阶矩最大差3.469e-8/相对L2=2.367e-5。10597对10583、10610对10598的独立复核仍同级别：最终权重最大差1.996e-6/1.912e-6，一阶矩相对L2=1.990e-5/1.803e-5，二阶矩相对L2=3.192e-5/2.419e-5。

判定：三次完整候选运行支持该固定真实模型全参数FP32 AdamW配置的**条件性数值对齐**；四次运行中一次真实Jittor core反向故障，公开CLI端到端稳定性不判PASS，不能外推L4或L5。根因定位止于core autograd中的非法Slice断言，具体生成位置仍未证明；此项停止重复试探，后续需带C++栈的最小复现并复测真实基准。AdamW新进程恢复、BF16、双卡、其他模型/数据、稳态性能均not-run，完整适配矩阵未完成。
