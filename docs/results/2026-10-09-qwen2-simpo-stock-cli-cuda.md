# Qwen2-0.5B stock `swift rlhf --rlhf_type simpo` CUDA 对拍

- 状态：历史运行键 `20261009-qwen2-simpo-fixedbatch-cli-v1` 按当时协议保留其单样本输入、首步前向、梯度与末态权重对拍结论（L0/L1/L2/L4 partial，L3 not-run，L5 blocked）。当前实现树固定单样本完整构造清单 L0 通过；新恢复键使 L3 达到 partial：同进程恢复至 step 6 成功，新进程恢复在 step 5 更新后于 step 6 backward 触发 NanoVector slice overflow。旧 L1/L2/L4 未被本次提升；L5 仍 blocked。该证据不代表多样本 sampler 或 ms-swift 整体兼容。
- 日期：2026-10-09。
- 基线：Jittor `6d20774d9f6e3d2fcc269905fcefbe5c62b09c7e`（上游 `2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B causal LM、公开单卡 `swift rlhf --rlhf_type simpo`，全参数 FP32/eager，beta 2.0、simpo_gamma 1.0、cpo_alpha 0、SGD、单条 preference 数据、batch 1、三步。
- 维护者：ms-swift CUDA 适配。
- 复查条件：扩展到多条固定 preference 数据并显式核验样本顺序；补 optimizer 状态、恢复轨迹后再升级 L2/L3；ms-swift SimPO/CPO trainer、Torch sampler 或 shim 数值实现变化时复查。

Qwen2 权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；单行 preference 数据 SHA256 为 `f2f99ab3ae84aa4f10a7e7690d302e6668ac831ac8f24e740d7ca62f00103e10`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、TRL 0.24.0。Slurm 13939 预检和 13945 训练均在 `cscg-qh04` RTX 4090（GPU UUID `GPU-4b797e8c-3982-9b01-dc17-43d77960643c`）完成。ms-swift 的 SimPO 参数处理把公开 `rlhf_type=simpo` 转为 `CPOTrainer(loss_type='simpo')`；本次明确关闭 CPO 附加 NLL 项（`cpo_alpha=0`）。

运行键 `20261009-qwen2-simpo-fixedbatch-cli-v1` 中原生 oracle 先运行，strict shim 后运行。两侧均通过 stock CLI 完成三步并保存 `checkpoint-3`。三步 `input_ids`、`labels` 和 `attention_mask` 逐元素相同；全部 290 项初始参数名称、shape、dtype、trainable 标志和值 SHA256 相同且位于 CUDA。strict shim 父子进程开始和结束均记录 `use_cuda=1`、`fallback=0`。

首步 chosen/rejected logits shape 为 `[1,29,151936]`，最大绝对误差 `8.16137e-5`，相对 L2 分别 `2.57146e-6` 与 `2.53968e-6`；其他聚合前向输出最大绝对误差不超过 `6.20e-6`。三步 loss 为 native `1.84327781, 1.83792770, 1.83255649`，shim `1.84328341, 1.83793056, 1.83257294`，最大差 `1.64509e-5`。

每一步均比较全部 290 项梯度、共 494,032,768 个 FP32 元素。最大绝对差依次为 `3.98606e-7`、`5.79283e-7`、`7.45466e-7`；相对 L2 为 `6.11241e-6`、`8.03060e-6`、`1.34174e-5`。step-3 checkpoint 的 290 个权重键对齐，170 个张量更新，末态最大绝对差 `7.45058e-9`、相对 L2 `7.21459e-11`。

| 层 | 状态 | 证据与缺口 |
|---|---|---|
| L0 | partial | 真实模型、tokenizer、SimPO trainer/optimizer 经公开 CLI 构造；290 项初态与 CUDA 参数设备核验一致。完整 state/buffer 和 tokenizer/template 映射未审计。 |
| L1 | partial | 三步输入完全相同；首步 chosen/rejected 完整 logits 与聚合输出达容差。hidden state、后续步逐层前向未采集。 |
| L2 | partial | 三步全部 trainable 梯度和末态模型权重达容差；optimizer state 与逐步直接更新未单独比较。 |
| L3 | partial | 历史键未运行恢复；当前树恢复续测见下文：strict shim 同进程通过 checkpoint-3→4→6，新进程从 checkpoint-4 恢复后 step 5 更新、step 6 backward 失败，完整 L3 未通过。 |
| L4 | partial | native 与 strict shim 均经 stock `swift rlhf --rlhf_type simpo` 完成三步并保存 checkpoint；仅覆盖一条固定样本。 |
| L5 | blocked | 前置层未完整通过；shim 训练约 568.6 s 含首次 JIT，未执行预热后至少 10 次稳态性能与显存协议。 |

## 当前实现树补充：固定样本 L0 清单

运行键 `20261012-qwen2-simpo-fixedbatch-l0-inventory-v4` 在 Jittor `f9454ec6ca03e9699a5b386af01b126d98200e25`（`python/jittor` tree `b066dc9e303d2d3e7eb2c4bd4a824ab68657e835`、`compat` tree `558002060ff0d29791daa7402c2ecc6fcd5fa49e`）和 ms-swift `88d727951203256baa564c643c651b6f8d90fd7e` 上，完成了 native 与 strict shim 的构造清单。Slurm jobs 18166/18168 因 no-build 缓存环境指纹不一致在构造前退出；18173 的 harness 错把公开参数 `simpo` 归一化为 `cpo` 判为失败。新键 v4/job18174 成功写出双方完整清单，但嵌入的比较器随后因要求优化器实现模块路径完全相同而退出。对已写出的 JSON 进行只读离线语义比对后通过；保留原始作业的 FAILED 状态，不把 harness 退出改写成作业成功。

两侧均构造同一 Qwen2ForCausalLM、Qwen2TokenizerFast、Swift Template 和 CPOTrainer，公开请求 `rlhf_type=simpo` 经 Swift 解析为 `rlhf_type=cpo`，训练器 `loss_type=simpo`；单条数据、beta=2、simpo_gamma=1、cpo_alpha=0。两侧参数名顺序、shape、dtype、device 和 `requires_grad` 一致：290 个参数、全部可训练，共 494,032,768 trainable 元素；291 个 state keys 和 1 个 buffer 的名称、shape、dtype、device 一致。native 与 shim 的优化器语义类均为 SGD、调度器均为 LambdaLR，290 参数分组映射、共同显式 `lr=1e-5` / `weight_decay`、scheduler state 一致。实现模块路径按各 runtime 原样记录；native 在 param group 中展开了继承的 SGD 默认项，shim 将这些默认项留作隐式值，因此不声称完整 group 字典相等。strict shim 进程实际 CUDA=1、fallback=0；双方 `train()` 均未调用，没有前向、反向、更新或 checkpoint。

job18174 运行于 `cscg-qh04` 的 RTX 4090（compute capability 8.9，GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）。本轮结论只补齐当前实现树的固定样本 L0 构造/元数据清单；历史数值对拍仍按原基线和协议单独解释，多样本 sampler 顺序、L3 恢复和 L5 稳态性能仍未由本次清单覆盖。运行原始 JSON、四次作业日志和离线比较结果保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-simpo-fixedbatch-l0-inventory-v4/`。

原始输入、梯度、比较 JSON、checkpoint 和日志未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261009-qwen2-simpo-fixedbatch-cli-v1/`。本报告只覆盖上述固定单样本配置。

## 当前实现树补充：固定样本 L3 恢复续测

新运行键 `20261012-qwen2-simpo-fixedbatch-l3-restore-current-tree-v7` / job 18399 绑定 Jittor `cae9c3b3b52995b634d66cc9af4e2949a5195ad5`（`python/jittor`、`compat`、`tests/structure` tree 分别为 `b066dc9e303d2d3e7eb2c4bd4a824ab68657e835`、`558002060ff0d29791daa7402c2ecc6fcd5fa49e`、`ba9f708326f0fa3ef1c24d46537282c530f90da9`）和 ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。只读复用此前 job 18240 的 native same/new-process 轨迹，以及历史键中 native/shim 的 checkpoint-3；未重跑 native oracle 或改写源 checkpoint。作业于 `cscg-qh04` RTX 4090 完成到恢复诊断，Slurm 状态为 FAILED（exit 1，3:29）；失败被保留，不能把同进程子路径的成功改写为整格成功。

strict shim 的同进程路径从 checkpoint-3 恢复模型、buffer、optimizer、scheduler 及 Python/NumPy/Torch CPU/CUDA RNG；恢复后的 290 个模型参数精确匹配，buffer 未变化，optimizer 参数组及空 state 与 checkpoint 一致。旧 scheduler 序列化只做已知窄归一化：旧 checkpoint 的 `_is_initial=False` 作为瞬态字段省略，缺失的 `verbose` 按默认 `False` 补齐；`last_epoch`、`_step_count`、`_last_lr` 等状态字段仍精确匹配，且更新前和保存 checkpoint 时另核 scheduler 计数与学习率。固定输入身份 SHA256 为 `c0d3290c58970dd673166db89e2c88520e1080a0473cbf79a09e5ced5c0e789d`。step 4–6 的 loss 为 `1.82720077, 1.82187378, 1.81653666`；每步均有 290 项有限非零 CUDA 梯度和真实 SGD 更新，243/290 参数张量存储值改变，最大绝对更新为 `9.5367431640625e-7`。same-process checkpoint-4/6 的 global step 与 scheduler 状态分别为 `(4, last_epoch=4, step_count=5)`、`(6, last_epoch=6, step_count=7)`。

独立新进程从该次生成的 checkpoint-4 恢复同一状态；step 5 输入身份与同进程轨迹相同，梯度与参数更新有效。step 6 的 batch 身份也已记录，但 backward 在 Jittor `grad_optional` 中以 `nano_vector.h:41: slice overflow` 失败，因而没有 step-6 loss、更新或新 checkpoint。strict shim 的父/子 worker CUDA 标记均为 1、fallback=0；source checkpoint、native oracle 与 `jittor_core`/`jit_utils_core` 哈希审计未见变化。该栈未定位到具体触发因素；它与既有 observer-correlated 故障相似，但不能据此断定根因为观察器或认定为同一缺陷。按既有五轮边界，不继续无界 hook 组合试探；L3 保持 partial，待有实质新证据时再选独立最小复现。

文档门禁 Slurm 13953 在 RTX 4090 worker 上通过：仓库布局检查成功，Torch 模式结构测试为 1386 passed、6 skipped、1027 subtests passed。6 个 skip 中 2 个因该测试解释器未安装 Jittor、2 个因未安装 pytest-xdist，其余由门禁分类记录。
