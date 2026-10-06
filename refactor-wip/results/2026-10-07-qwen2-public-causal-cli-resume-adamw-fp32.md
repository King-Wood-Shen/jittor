# Qwen2 公开因果 SFT CLI：纯 FP32 AdamW 限定恢复数值合同

基线 Jittor `4108e955d1c1485dcc366cb7489bd4403f6a3b34`、ms-swift `88d7279`。仅通过 Slurm NVIDIA worker 做导入、JIT、训练和比较。入口 `python -m swift.cli.main sft`，真实缓存 Qwen2-0.5B，四条固定对话整批，显式 `--torch_dtype float32 --fp16 false --bf16 false`、eager attention、full tuner 冻结基座仅训练 `model.norm.weight`，AdamW 1e-4、weight_decay=0、constant scheduler，连续三步并在第二步保存；另起 CLI 进程从 checkpoint-2 恢复第三步。原始脚本、日志、checkpoint、比较器和结果均在 `_state/ms-swift-cuda/20261007-qwen2-public-causal-cli-resume-adamw-fp32`。

原生 10420、候选 10422 均 `COMPLETED 0`，连续与新进程恢复各自得到 checkpoint-3；候选 10421 在 `jit_utils` 重建阶段退出、未进入训练。候选连续及恢复的 CLI 父子进程 bootstrap 起止均 `use_cuda=1`、shim marker 真、fallback0。两个后端各自连续/恢复最终 896 维训练权重完全相同，AdamW 一阶/二阶矩最大差：原生均 0，候选分别 2.515e-8/1.164e-9；optimizer step 都为 3、矩非零，scheduler 的各自状态字典完全一致，global_step3，290 模型键及 optimizer/scheduler/RNG checkpoint 存在。候选恢复与连续的第 3 步 loss 差 4.768e-6、grad norm 差 3.129e-7；原生这两项差 0。

跨后端连续三步的第 3 步 loss 差 1.907e-6、grad norm 相对差 6.426e-6、训练权重最大差 0，AdamW 一阶/二阶矩最大差 3.241e-7/1.700e-8。10430 原定同后端 loss 1e-6 门槛被候选微小波动触发；10431 独立诊断保留全部差异，10432 改以 1e-5 损失容差重核限定数值合同通过。10433 增加跨后端完整 scheduler 字典相等要求后失败；第五轮 10434 定位：原生多 `verbose=False`，候选多 `_is_initial=False`，两侧 `base_lrs`、`_last_lr`、`last_epoch=3`、`_step_count=4` 等影响当前学习率的字段一致。这是内部 schema 差异，不能宣称 checkpoint 字节或全字段等价。

判定：真实公开因果 CLI 的单归一化参数、纯 FP32 AdamW、固定三步的恢复训练**数值合同限定通过**；后验放宽过日志 loss 容差，完整调度器结构等价未通过，不能外推所有训练状态序列化或通用 L3。全参数、LoRA、混合精度、双卡、新模型和 L5 仍 not-run；完整适配矩阵未完成。五轮诊断到此停止，未修改产品源码或历史脏工作树。
