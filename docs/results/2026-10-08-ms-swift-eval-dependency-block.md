# ms-swift `swift eval` CUDA 验收：依赖前置被阻断

- 状态：阻断，未运行任何模型评估；不是兼容通过或兼容失败。
- 日期：2026-10-08。
- 基线：Jittor `e0d958829013b7db00fe4e204fc86fa85b3d250e`；同步上游 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：公开 `swift eval`、`general_mcq` 本地两条样例、Qwen2-0.5B、Native eval backend 与 Transformers 推理 backend。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Slurm worker 可访问配置的软件包代理，或在 worker 可读的离线 wheelhouse 中提供 `evalscope>=1.0.0`。

ms-swift 的 `requirements/eval.txt` 声明 `evalscope>=1.0.0`。既有 Python 3.11 环境没有该依赖。Slurm 13140 在原生预检导入阶段退出；没有加载模型，shim 未运行。Slurm 13163 在 worker 上确认 RTX 4090 和模型/数据哈希后，因为计算节点没有 `python` 命令而退出，未安装依赖或运行模型。修正为使用环境内 Python 的 Slurm 13164 创建了独立 `--system-site-packages` venv，但 pip 下载 `evalscope` 时，配置代理连接被拒绝（`ProxyError: [Errno 111] Connection refused`），随后以 `No matching distribution found` 退出。该 venv 和所有日志均留在未版本化的 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-public-eval-general-mcq-v3/`；前两次尝试也分别保存在 `...general-mcq-v1/` 与 `...general-mcq-v2/`。未修改共享 Python 环境。

因此本次没有原生输出、shim 输出、设备驻留或 fallback 数据，不能据此给 `eval` 任何数值兼容结论。按该功能面的 L0-L5 门槛，均记为 blocked/not-run；`sample`、metrics、callbacks/plugins 也未运行。网络恢复或离线 wheelhouse 可用后，从独立原生 oracle 开始重启该面验收。
