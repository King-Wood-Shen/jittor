# Qwen2 公开因果 SFT 全参数变长数据恢复：采样协议错误与 Jittor Core 阻断

## 合同及证据

- 真实缓存 Qwen2-0.5B、公开 `python -m swift.cli.main sft`、494.0328M 全参数、纯 FP32/eager、单卡、SGD 1e-5、微批 2、梯度累积 2，12 条不同且 token 长度 31–37 的样本。目标是比较连续三步与从 checkpoint-2 新进程恢复的第三步，并验证通用数据游标。
- Slurm 原生 10668 COMPLETED0，连续 loss 为 1.02839017、3.26030397、2.68251991；原生恢复第三步 loss 和 grad norm 与连续完全一致。
- 严格 CUDA 候选 10669 FAILED1：前两步 loss 为 1.37361312、2.98017597，第三步在 `jt.core.grad_optional` 中触发 `nano_vector.h:41 slice overflow: 94648546578480 0 1`。候选只保存到 checkpoint-2，恢复阶段未运行。父/子进程 bootstrap 记录 strict CUDA、shim marker 真、fallback0。依赖候选完成的比较作业 10670 已取消；无比较数值结论。

## 根因边界

- 本实验给了 `--dataset_shuffle false`，但原生和候选日志均明确显示 `train_dataloader_shuffle=True`。ms-swift 的前者是数据预处理参数；训练 batch sampler 单独采用 `train_dataloader_shuffle`，默认值为 True（`swift/trainers/arguments.py` 与 `swift/trainers/mixin.py`）。因此本次并未建立顺序采样合同，也未记录实际逐批样本 ID。前两步 loss 分歧不能直接归因于模型数值计算或恢复游标。
- 第三步错误是 Jittor Core 反向传播的内部非法 Slice 断言；生成它的上游操作尚未定位。同类断言曾在全参数 AdamW 第三步出现，本次在 SGD 和变长数据上再次出现，说明不能限定为 AdamW 独有。没有证据支持修复或稳定通过。
- 原生 checkpoint-2 恢复成功仅证明原生基准；候选恢复、跨后端对齐及通用数据游标 L3 均 **blocked/not-run**。此前四条固定样本的 SGD 恢复仍只成立于其原限定合同，不能外推。
- 不以重复提交碰运气规避 Core 断言；该问题已有五轮历史上限。继续独立的 RNG 状态审计与其他功能面。

## 位置

原始 worker 脚本、12 条数据、训练日志、bootstrap、Slurm 输出和 checkpoint：`/home/xinshen/projects/ms-swift-cuda/jittor-lab/_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-cursor-resume-fp32`。提交只收录报告，运行产物留在远端 _state。
