# Ascend ms-swift 公开 TransformersEngine 推理

- **状态：** 当前源码在真实 Ascend NPU 上完成 Swift 公开推理；4 个请求均执行真实模型前向并返回 4 个 greedy token。
- **日期：** 2026-10-21
- **同步基线：** `b2fa61449531341caabf418e639fd41e5e4ff0c7`（`origin/2.0-refactor`）
- **源码：** `d0cdbfdac0042d7aca865f375273aff861e9354e`
- **验证范围：** Swift 4.5.2 `swift.cli.infer`、TransformersEngine、tiny Llama + Swift LoRA adapter、FP32 eager attention、单 NPU、cache 开启、贪心生成。
- **维护者：** ms-swift / Torch compatibility 维护者
- **复查条件：** Swift infer template/generation、Transformers `DynamicCache`、LoRA loader 或 JTorch ACL generation path 改动时复查。

## 结果

Slurm job 5928 在设备物理 ID 2 上运行，Swift 输出 4 个请求、共 16 个生成 token；observer 在模型的实际 generation owner 上注册公开 forward hooks，没有替换模型的 `generate` 或 forward。每条请求均记录 4 次真实模型 forward 和 `DynamicCache`，输入、logits、KV cache 与 generated token 均驻留在 `npu:0`。首个 decode 输出的 logits 为 `[1,1,128]`，两层 K/V cache 长度从 15 增至每个生成步继续增长。ACL fallback policy 为 `error`，计数为 0。

四个返回分别为 `t63 t63 t63 t63`、`t63 t63 t63 t63`、`t63 t63 t63 t63`、`t45 t99 t39 t63`。Swift 记录整个 4 请求运行耗时 87.0 秒；observer 在每个 forward 同步并保存输入/输出快照，该时间只用于复现记录，不能作为推理性能结论。

## 复现与证据

作业脚本：`$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/public-infer-current-20261021-r3/infer.sbatch`。job 5928 回执、输入 SHA-256、Slurm 日志、Swift 公开输出以及 observer 的每请求 JSON/NPZ 证据均在同一外部 run 目录。

调用入口和关键参数：

```bash
srun ... host-shim-python -m swift.cli.infer \
  --model <tiny-llama> --model_type llama \
  --adapters <locked-swift-lora-adapter> --use_swift_lora true --tuner_type lora \
  --infer_backend transformers --merge_lora false --load_args false \
  --torch_dtype float32 --attn_impl eager --val_dataset <infer.jsonl> \
  --max_batch_size 1 --max_new_tokens 4 --temperature 0 --num_beams 1 \
  --max_length 64 --template ms_swift_inference_evidence \
  --external_plugins <observer.py> --result_path <isolated-output.jsonl>
```

## 边界

本项证明固定 fixture 的 adapter 加载、真实 TransformersEngine generation、单卡 ACL execution 和 KV cache 设备驻留。未做 Torch NPU 数值 parity、吞吐对比、多卡推理、vLLM/SGLang 后端或长上下文验证。历史推理 job 9537 的源码 SHA 为 `47b507b8`，job 7af2 的源码 SHA 为 `745dcd7e`；它们属于旧测量，本报告仅记当前源码的 Swift shim 运行。
