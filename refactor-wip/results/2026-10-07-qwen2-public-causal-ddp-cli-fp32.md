# 真实 Qwen2 公开因果 SFT CLI：双卡显式 FP32 限定数值对齐

基线 Jittor `d224badc1b3e7302a3ac048e1aea004e3dda4081`、ms-swift `88d7279`。全部导入、JIT、训练与数值比较位于 Slurm NVIDIA worker。调用 `python -m swift.cli.main sft`、`NPROC_PER_NODE=2`，真实缓存 Qwen2-0.5B、四条固定离线对话、双卡每卡batch2、三步、`--torch_dtype float32 --fp16 false --bf16 false`、eager、full tuner冻结基座仅训练`model.norm.weight`、SGD 0.01。候选双卡使用原生`torch.distributed.run`作控制平面启动Jittor rank子进程，此限定边界与原双卡Embedding实验一致；不能据此宣称纯Jittor torchrun兼容。原始脚本、日志、checkpoint与比较JSON在`_state/ms-swift-cuda/20261007-qwen2-public-causal-ddp-cli-fp32`。

原生10506、候选10507均`COMPLETED 0`，各有rank0/rank1 RNG文件、optimizer/scheduler、290模型键和global_step3。候选主进程及两个rank启动与结束均`use_cuda=1`、shim marker真、fallback0。独立比较10510`COMPLETED 0`：三步loss最大绝对差2.620e-6，grad norm最大相对差4.373e-6；最终896维`model.norm.weight`最大差4.768e-7、相对L2差2.124e-9，模型键一致；两侧相对原始基座的训练权重更新L2分别0.0048693617/0.0048693964，确为非零更新。token_acc三步均0.4375。首次JIT与不同GPU工作节点的运行耗时不作L5性能比较。

判定：固定真实Qwen、公共CLI、双卡混合控制平面、显式FP32单归一化参数训练三步的限定L4数值合同PASS。原生控制平面、模型全参数/LoRA、混合精度、其他数据与模型、双卡CLI恢复及十次预热稳态L5均未覆盖；历史双卡Embedding公开CLI首个Qwen输出分歧仍为失败，不被本因果任务覆盖。完整适配矩阵未完成。

## 同一限定路径的双卡新进程恢复

保持上述真实基座、公开CLI、两rank、显式纯FP32、仅norm、SGD与固定四条数据。原生10512、候选10513分别连续三步并保留checkpoint-2/3；另起原生10516与候选10518公开CLI进程从各自checkpoint-2恢复第三步，10520独立worker比较。五个Slurm作业均`COMPLETED 0`；候选连续与恢复的主进程和两rank起止均strict CUDA、shim marker真、fallback0。两rank RNG文件、optimizer/scheduler、290模型键、global_step3均核对。

同后端连续/恢复第3步896维训练权重最大差均0；原生loss/grad norm差均0，候选loss差4.768e-7、grad norm差1.490e-8，token_acc一致；各自optimizer字典与scheduler字典完全相同。跨后端连续第3步loss差2.384e-6、grad norm相对差3.948e-6、norm最大差4.768e-7。候选恢复日志提示`lm_head.weight`缺失；原始Qwen2配置`tie_word_embeddings=true`，在该限定三步前向/更新合同中观察到恢复一致，但仍需把该加载警告保留，不外推非共享输出头。该SGD无动量，optimizer state为空；不能推断有动量/AdamW或全参数分布式状态的恢复。

判定：上述双卡混合控制平面公开CLI的纯FP32单归一化参数、无动量SGD新进程恢复限定L3数值合同PASS。纯Jittor启动器、其它优化器和训练参数、双卡Embedding公开CLI失败及L5不受此结果覆盖；完整矩阵仍未完成。
