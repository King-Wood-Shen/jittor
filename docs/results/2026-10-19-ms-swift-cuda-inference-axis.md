# ms-swift CUDA 推理与优化开关矩阵

- 状态：首轮六场景矩阵、同步后 greedy 复验及 SDPA 公开入口 CUDA 补测完成；Swift 入口 cache-off 未触发，不授予整个 ms-swift、外部推理引擎或投机解码的 Jittor 内核覆盖。
- 日期：2026-10-19；补测：2026-10-20。
- 首轮同步基线：`origin/2.0-refactor` `3577b70b4ca72296c8e49cbf37ebb51fe3c76497`；六场景完整矩阵 HEAD 为 `f6fe012b1a17a7ab9377c196a7836ea9df7afb5f`。
- 本轮恢复同步：`origin/2.0-refactor` `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`；同步后 HEAD `6e947cd727c35ac9151e80ffcb50f2868c8d4b94`。
- 上游源码：外置 ms-swift checkout `88d727951203256baa564c643c651b6f8d90fd7e`；验证不修改该 checkout。
- 维护者：ms-swift CUDA 推理适配。
- 复查条件：Jittor HEAD、ms-swift/Transformers/PEFT 版本、生成配置、模型权重、adapter 权重或引擎分派变化。

## 固定实验与判据

Slurm job `105611`（RTX 4090，完成/exit 0）执行，原生 PyTorch 与 Jittor Torch shim 使用同一外置 ms-swift 源码、Qwen2-0.5B 本地权重和同一 LoRA checkpoint。两侧安装身份均为 Torch `2.6.0`、Transformers `4.57.6`、PEFT `0.17.1`、ms-swift `4.6.0.dev0`，并分别断言 oracle 不是 shim、候选确为 shim。入口为 `swift.infer_engine.TransformersEngine.infer`；六个情景分别新建引擎，调用公开 `InferRequest` 与 `RequestConfig`。观察实际模型 `forward` 的输入形状、设备和 `use_cache`，记录输出 token ID；shim 最终 `use_cuda=1` 且 backend fallback 增量为 0。加载 adapter 使用 `TransformersEngine(adapters=[...])`，合并使用加载后模型的 PEFT `merge_and_unload()`，再由同一个 Swift 引擎推理。

完整脚本、Slurm 提交文件、两侧逐项输出与日志未版本化，位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261019-inference-axis-v1/`。原始模型 `model.safetensors` SHA-256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；adapter `adapter_model.safetensors` SHA-256 为 `2c06c96d42696f442b5c7d255cdf83d2d7576ddd8b1025d31e97486ab4bffbfd`。实验中的 5 token 长度只用于触发 prefill/decode、采样、批处理与 LoRA 路径，不构成性能结论。`duration_s` 原值留在原始结果；含冷启动与 JIT，不用作速度比较。

## 本轴覆盖矩阵

| 场景 | 实际执行归属与判据 | 当前结论 |
| --- | --- | --- |
| Transformers greedy、KV cache on | Swift `TransformersEngine.infer` → 模型 `generate`，独立 PyTorch 对照；逐步读取 CUDA 输入与输出 | 两侧 5 个 token ID 完全一致；首步 `[1,25]`、后续 `[1,1]`；shim fallback 0 |
| Transformers sampling | `RequestConfig(temperature=0.8, top_k=8, top_p=0.8, seed=1234)`；检查准备后的 `do_sample` 与两侧 token | 两侧 `do_sample=true`，5 个 token ID 完全一致；该 prompt 恰与 greedy 相同，不外推随机分布一致 |
| Transformers batch | 同一次 `infer` 两条请求，实际首步 batch size 2 | 两条响应各 5 个 token ID 完全一致；两侧首步 `[2,25]`、后续 `[2,1]` |
| Transformers adapter 加载 | Swift 引擎加载 PEFT LoRA；观察底层模型 CUDA `forward` | 两侧 `PeftModelForCausalLM`，5 个 token ID 完全一致；该短输出与 base 相同，不证明长输出或权重轨迹相等 |
| Transformers adapter 合并 | 在 Swift 引擎内将 PEFT 模型合并为 Qwen2 基类并继续推理 | 两侧合并后为 `Qwen2ForCausalLM`，5 个 token ID 完全一致；48 个 LoRA 层实际非零，合并后的最大权重变化两侧同为 `2.016779e-6`；不覆盖 Swift export CLI |
| Transformers KV cache off | 尝试把引擎及模型 `generation_config.use_cache` 设为 false；检查准备配置与真实 `forward` | Swift 公开 `infer` 两侧准备后的 `use_cache` 与全部真实 `forward` 参数仍为 true，不能计入该入口的 cache-off |
| vLLM | Swift 建立 `vllm.LLMEngine` 或 `AsyncLLMEngine` 并向该实例提交请求 | 源码确认控制层分派；本作业无真实引擎内核验证 |
| SGLang | Swift 建立 `sglang.Engine`，调用其 `async_generate` | 源码确认控制层分派；本作业无真实引擎内核验证 |
| LMDeploy | Swift 调用 `lmdeploy.pipeline`，由其 backend 执行 | 源码确认控制层分派；本作业无真实引擎内核验证 |
| speculative decoding | Swift 把 `speculative_config` 交给 vLLM，或把 speculative 参数交给 SGLang `ServerArgs` | ms-swift 源码有该公开引擎参数，但当前隔离环境未安装 vLLM、SGLang 或 LMDeploy；未安装、未下载、未执行，不授予 JTorch 计算覆盖 |

ms-swift 的 `RequestConfig` 没有 `use_cache` 参数。其 `prepare_generation_config` 生成新的 `GenerationConfig` 时只转发部分字段；这次即使把引擎的配置和模型配置设为 false，准备后的配置及实际模型调用仍为 true。原生 PyTorch 与 shim 呈现同一现象，归属在 Swift/Transformers 生成配置契约，不应以 token 相同冒称关闭缓存的路径已通过。

为限定计算能力边界，Slurm job `105725` 另在 Swift 加载出的相同模型上直接调用公开的 Transformers `model.generate(use_cache=False)`。两侧五次真实 `forward` 的 `use_cache` 均为 false，输入长度从 6 逐步增至 10；原生 PyTorch 与 shim 的输出 token ID 完全一致，shim CUDA 且 fallback 0。这只证明底层无缓存生成可运行，不提升 Swift `TransformersEngine.infer` 的 cache-off 覆盖。

独立 Slurm merge probe 在同一固定 adapter 的 48 个实际 LoRA 层上比较合并前 `base_layer.weight` 与合并后基础模型权重；adapter B 最大绝对值两侧均为 `5.711696e-6`，合并权重最大绝对变化两侧均为 `2.016779e-6`。Shim 在该 probe 中仍为 CUDA 且 fallback 0。它证明合并不是空动作；短文本的 token 未变化是该输入的结果。

外部引擎的执行归属见所绑定 ms-swift 源码 `swift/infer_engine/{vllm_engine,sglang_engine,lmdeploy_engine}.py`。vLLM 在其他独立任务中可能有 Jittor adapter；这不等于本次 Swift vLLM 请求进入了 JTorch。此矩阵只把已实际执行并有设备与 fallback 证据的 Transformers 路径列为 CUDA 计算验证。

## 仓库门禁与限制

同一 Slurm 节点上的 `bash tools/check_repo_layout.sh` 通过。首次完整 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 `1511 passed, 6 skipped, 3 failed`；其中报告页未加入 toctree、生成的 `MANIFEST.in` 未更新是本任务新增文档造成，已补齐；另一失败为该解释器缺少 `asv`（`No module named asv`），是环境依赖，不以文档改动规避。随后 Slurm job `105785` 的布局、manifest `--check --project .` 与两个受影响结构文件定向复验通过，`23 passed, 50 subtests passed`。完整及定向日志分别在上述外置 state 的 `structure.log`、`structure-targeted.log`、`layout-final.log` 与 `manifest-check.log`。

本次没有修改 Jittor 计算代码，故未把历史训练门禁或旧实验重标成当前 HEAD 的推理结果。短输出验证的是指定 Qwen2 模型、输入、dtype 与六个情景；没有大上下文、吞吐或延迟性能结论，也没有外部引擎在 JTorch 上执行的证据。

## 同步后复验

恢复任务时发现 `origin/2.0-refactor` 前进至 `5a120a315ed2f213d937b546116101006643f87d`。该范围还包括 Jittor executor/FusedOp/liveness/运行时 flag 调整；按实际影响未重跑全部六格，而在合并树 `7131681d43e9b0dfa6ef0ad6ef5225647ea3271e` 上复验一条真实 Swift CUDA greedy 请求，并重跑 CPU core 与结构门禁。

Slurm job `105900` 原生 PyTorch 侧和 job `105928` 修正探针导入顺序后的 shim 侧均使用上面绑定的 ms-swift、模型与 tokenizer。Swift `TransformersEngine.infer` 在两侧实际执行五次 CUDA forward，形状 `[1,25]` 后四次 `[1,1]`；token ID 与首轮完整矩阵一致，Shim `use_cuda=1`、fallback 0。`105900` 首次 Shim 探针失败是探针先 import 原生 `torch` 后启用 shim；修正导入顺序后只重跑 Shim 并通过。这是探针设置错误，未修改产品代码。

Slurm job `105905` 在合并树上通过 CPU core：native `98 passed, 41 skipped, 1 xfailed`；Torch `224 passed, 1 skipped`；合计 `322 passed, 42 skipped, 1 xfailed`。完整结构门禁首次的两个报告登记失败已修复；复验通过布局与生成 manifest 检查。计算节点安装钉定的 `asv==0.6.6` 时代理连接被拒绝（日志 `asv-install.log`），所以结构复跑仅排除该 ASV CLI 测试，其余为 `1511 passed, 8 skipped, 1 deselected, 1058 subtests passed`。另有 4 项 environment skips：两个测试解释器未安装 jittor、两个未安装 pytest-xdist；详情见 `structure-postsync-noasv.log`。未排除 ASV 的完整结构门禁仍待可联网的维护环境补跑。

首轮六场景矩阵数据仍绑定首轮 HEAD `f6fe012b1`；当前合并树另有一条 greedy CUDA 复验，不能把后续矩阵格扩写成完整当前树覆盖。

本轮同步后仓库布局检查通过。Torch shim 结构门禁执行为 `1513 passed, 6 skipped, 1 failed,
1066 subtests passed`；唯一失败为 `test_asv_entrypoints.py::TestAsvEntryPoints::test_source_config_is_executable_without_nox_path_rewriting`，
子进程解释器缺少 `asv`（`No module named asv`）。本地 Python 与已用隔离环境均未预装 ASV；此前
`asv==0.6.6` 安装日志记录代理拒绝连接，且当前不重试安装。另 4 个环境 skip 分别来自测试解释器
未安装 jittor 与 pytest-xdist；其余测试通过。此门禁结果绑定本报告补测后的仓库状态，不能据此声称
完整结构门禁全绿。

## 2026-10-20 同步后优化开关补测

本轮同步后 HEAD 为 `6e947cd727c35ac9151e80ffcb50f2868c8d4b94`，远端基线为
`d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`。复用同一个 Qwen2-0.5B 本地模型，不复制或下载权重；
模型文件 SHA-256 仍为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
ms-swift 源码 SHA 为 `88d727951203256baa564c643c651b6f8d90fd7e`，其
`transformers_engine.py` SHA-256 为 `02c63b0488bbd817725b73ba27a103b421fe9127ff5e098ace58fd0979b32efb`。
依赖身份为 Torch `2.6.0`（安装 RECORD SHA-256
`7b34dbec0f00ca04925fc326ddd73a2ec7ae6409a488bab37eab856143daa147`）、Transformers
`4.57.6`（`8e7e1cb85024919785ff2f6ff4d0d38c21524addfaf55ea39eb78056c5d7ddef`）、PEFT
`0.17.1`（`239bb3fba644163d4306d04c5381826e72c6af84ef07d3a62004ecea58d56d68`）。ms-swift
从以上固定 git checkout 导入，版本 `4.6.0.dev0`。原始探针与 NVML 采样记录未版本化，位于
`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-inference-postsync/`。

| 新增项 | 公开入口与模型 | CUDA 观测与输出 | fallback / NVML 峰值 |
| --- | --- | --- | --- |
| Swift SDPA attention | `TransformersEngine(attn_impl='sdpa').infer(InferRequest, RequestConfig)`；Qwen2-0.5B；RTX 4090，CUDA 设备 6 | 配置回读为 `sdpa`；实际 `forward` 输入依次 `[1,25]`, `[1,1]`, `[1,1]`，均为 `cuda:0`；`use_cache=true`；输出 token `[785,6722,315]`，与既有 eager greedy 前 3 token 相同 | `use_cuda=1`、fallback 增量 0；`nvidia-smi`/NVML 设备峰值 2486 MiB（基线 1 MiB） |

本补测只新增 SDPA 格，不重复既有 greedy、sampling、batch、adapter 加载/合并或 cache-off 探针。计时未作验收。
首轮和合并后历史场景的原始记录没有 NVML 采样，故这些场景的 NVML 峰值标为“未采集”，不从
Slurm CPU 内存字段或 allocator 数据推算。ms-swift 源码确认 speculative 参数只传给 vLLM/SGLang
引擎；当前 vLLM、SGLang、LMDeploy 均未安装。对应源码文件 SHA-256 分别为 vLLM
`51244ca497c5338dfca3f3480615333f2506b3b6c137e33f7d126f33fa9254cd`、SGLang
`9f4fbd470c8a62a086abfbc6ffe1f5c8eb31843b52604100a90c47860dcb2db0`。未尝试安装依赖，故没有
外部推理引擎或 speculative CUDA 结果。该环境也未安装 `flash-attn` 或 `liger-kernel`，因此未运行
Swift 的 flash attention 选项或额外 Liger 推理优化；没有因本轮验证而拉取新依赖。
