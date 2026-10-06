# ms-swift 公开入口与模型资源审计（静态）

基线：Jittor integration HEAD d6443be4a0c6b7dec7122468c3804924a58e67b4；ms-swift 88d7279。此次只在 cscg-qh00 登录节点静态阅读文件及包目录，未导入 Python、未运行模型、未提交 Slurm 作业，因此没有新增 L0–L5 数值通过项。

| 功能面 | 静态证据 | 当前判定 | 解除条件 |
| --- | --- | --- | --- |
| 其他真实模型家族及 Qwen2.5 | ModelScope 模型目录下 Qwen2.5-0.5B、Qwen2.5-0.5B-Instruct、ModernBERT-base、Qwen3-Embedding-0.6B 等仅有 .mdl 元数据；对模型缓存搜索 model.safetensors、pytorch_model.bin、model-00001-of-*.safetensors，仅找到 Qwen2-0.5B 的 988097824 字节权重 | resource-blocked；L0–L5 not-run。目录存在不等于 checkpoint 可加载 | 获取合法、完整的真实权重与 tokenizer/config，再在 Slurm NVIDIA worker 运行原生与 strict CUDA 零 fallback 配对 |
| swift eval | ms-swift swift/pipelines/eval/eval.py 顶层导入 evalscope.constants、evalscope.run、evalscope.summarizer；隔离 venv 的 site-packages 没有 evalscope 包目录 | resource-blocked；L0–L5 not-run。没有运行评测，也没有把缺依赖推断成 shim 失败 | 在可复现、双方相同的隔离依赖环境安装锁定版本，准备真实离线评测数据后用 Slurm worker 配对 |

既有 Qwen2-0.5B 及 adapter/公开推理证据仍按各自报告限定；此审计不改变历史 PASS、失败或五轮跳过状态。集成工作树此前干净，原始工作树 30 个修改加 2 个未跟踪文件未触碰。其他尚未覆盖的性能、训练、恢复与服务面保持各自 not-run/blocked，不因本审计升级。
