# Ascend ms-swift 单机双卡公开 BF16 SFT

- **状态：** 真实 Ascend NPU 单机双卡公开 SFT 通过；两个 rank 完成两步 BF16 训练。
- **日期：** 2026-10-21
- **同步基线：** `b2fa61449531341caabf418e639fd41e5e4ff0c7`（`origin/2.0-refactor`）
- **源码提交：** `c3de64d7f85fe6bf9b670827312bf2965baa155f`
- **验证工作树：** 运行回执记录提交 `f42120bc902b51c7644eb28024ba118871d823f5` 与未提交差异 SHA-256 `2614e42f68c55a2940bac5f81827826fc118d890f6d4fc5fc2c909750163c485`；两者合并后即源码提交 `c3de64d7`。
- **验证范围：** Swift 4.5.2 公开 SFT，tiny Llama 全参 BF16，单机 world size 2，HCCL，ACL strict fallback，2 步。
- **维护者：** ms-swift / Torch compatibility 维护者
- **复查条件：** HCCL dtype 映射、distributed launcher 的设备隔离、Torch DDP/Accelerate 初始化路径变化时复查。

## 结果

Slurm job 5925 在物理设备 3、5 上运行，两个 rank 均初始化 HCCL 并完成 Swift 训练。两 rank 的 loss 为 `4.90625`、`4.875`，grad norm 为 `6.0929651`、`6.1631541`；状态文件均记录 `world_size=2`、`process_group_backend=hccl`、`backend=acl`、`fallback_policy=error`、fallback `0→0`、`steps=2`。每个 rank 的 NPU 进程内逻辑设备编号为 0；launcher 记录分别绑定物理设备 3 和 5。

这次验证修复了几个连续暴露的双卡边界：rank 子进程同时设置 CANN 和 Swift shim 的可见卡环境，防止 host shim 重载完整设备列表；Swift 入口直接调用 `swift.cli.sft`，避免 `swift.cli.main` 派生第二个解释器重复初始化 HCCL；把 CANN 9.0.0 的 `HCCL_DATA_TYPE_BFP16` 接入 collective dtype 表；让 shim 的通用 shard `all_gather` 通过活动 HCCL ProcessGroup 执行；发布 Accelerate 默认 DDP 参数配置所需的 optional comm-hook 导入名。

Accelerate/Transformers 默认 DDP 初始化会导入 `ddp_comm_hooks` 模块，即使没有选择通信 hook。shim 目前只发布这组 import 名称；显式调用 FP16/BF16 压缩或 PowerSGD hook 会抛 `NotImplementedError`，这些通信算法尚未实现或验收。本次 SFT 未选择自定义 DDP hook。

## 复现与证据

Slurm 脚本使用 `$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/public-sft-world2-20261021-r9/train.sbatch`。job 5925 的源码与输入回执、rank 日志、loss 日志、fallback/device JSON 审计均保存在同目录下的 `receipt-5925.txt`、`input-sha256-5925.txt`、`launcher-logs/job-5925/` 和 `run-5925/`。模型文件与数据文件的 SHA-256 记录在输入回执中。

固定命令形态为：

```bash
srun ... host-shim-python -m jittor.distributed.launch \
  --nproc-per-node 2 --backend hccl --device-ids 3,5 \
  --master-addr 127.0.0.1 --master-port <job-derived-port> \
  -- host-shim-python -m swift.cli.sft \
  --model <tiny-llama> --model_type llama --template llama \
  --dataset <tiny-sft-jsonl> --tuner_type full --max_steps 2 \
  --torch_dtype bfloat16 --bf16 true --fp16 false \
  --attn_impl sdpa --packing false \
  --external_plugins <rank-audit.py> --callbacks msw_world2_audit
```

首次验证尝试曾因 rank 可见卡、Swift 单设备 rank 计数、BF16 HCCL dtype、Accelerate optional import 和 HCCL gather 接线路径依次失败；对应修复在源码提交内。修正后 job 5925 通过。仓库布局门禁通过；`tests/structure` 尝试在登录节点运行后因高 CPU 占用和多项环境相关失败/错误被中断，没有可接受的完整结果。未运行 CPU smoke/full 或 CPU 训练门禁。

## 边界

只验证了该 tiny Llama 配置、BF16、单机两卡、两步 SFT 和默认 DDP communication hook。没有验证双卡推理、DPO/GRPO、分片训练、多机、吞吐或长程训练。job 5925 JSON 中两个 rank 报告逻辑 `npu:0`；物理卡映射以 launcher 的 `device=3` / `device=5` 记录为准。
