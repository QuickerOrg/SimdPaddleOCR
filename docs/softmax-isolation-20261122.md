# softmax 修复隔离与 GPU 紧凑末端优化实验

本实验沿用 `codex/gpu-ocr-experiments`，不替换现有服务，不修改子模块提交，也不改变默认 OCR 选项。

## 可证伪假设

1. CPU 融合投影增加 softmax 分母是 CPU 成本上升的原因：只撤回该提交后，每请求 CPU 时间应下降；独立投影微基准也应更快。
2. GPU 两次词表读取与共享内存树形归约有额外开销：减少读取及同步后，末端 dispatch 的 GPU timestamp 应缩短，索引和概率仍需正确。
3. 原先 4 并发单轮的 11%～13% 吞吐下降可能包含环境影响：同引擎控制变量、交替多轮后，应能区分稳定代价和轮间波动。

## 五个隔离版本

| 名称 | 引擎 | 末端 |
|---|---|---|
| cpu-logit | 固定 cb80be7，只将 bd861eb 涉及的 8 个源码文件恢复为父提交版本 | CPU，返回旧 logit，仅作诊断控制 |
| cpu-prob | 固定 cb80be7 | CPU，当前 softmax 修复 |
| gpu-logit | 固定 cb80be7 | GPU，替换为 9df60ef 的原 logit 着色器，仅作诊断控制 |
| gpu-prob | 固定 cb80be7 | GPU，现有两遍读取的概率着色器 |
| gpu-online | 固定 cb80be7 | GPU，单遍在线 softmax 概率候选 |

CPU 撤回在 RAMDisk 独立源码副本内完成；GPU logit 对照也在独立源码副本中构建。正式源码不提供返回错误 logit 分数的运行开关，避免把诊断控制误作生产配置。二进制精确散列与撤回 patch 存在运维报告目录。

## 候选实现

保持现有 `ctc_compact` 为默认实验内核，仅在 `SIMD_OCR_EXPERIMENT_COMPACT_HEAD=1` 时才可能使用 GPU 紧凑末端。

- `SIMD_OCR_EXPERIMENT_COMPACT_KERNEL=subgroup`：保留两遍读取，使用 subgroup max/min/add 替代多数全组树形归约。
- `SIMD_OCR_EXPERIMENT_COMPACT_KERNEL=online`：每条 lane 只读取一次词表，维护局部 `(max, sum(exp(x-max)))`，再用 subgroup 归约和一次全组同步合并，返回 `1 / denominator`。
- 未设置或其它值：继续使用原两遍读取的 baseline。

三个内核各有独立 EngineExtensions 实例，避免不同选择命中同一编译图缓存。新着色器同时提供 fp32 与 fp16 存储资源，计算和归约使用 fp32；索引平局仍选最小类别，六字节打包协议不变。subgroup 候选需要 Vulkan 1.1 subgroup 基础及算术能力，本次只验证本机 RTX 4080。

## 验证方法

- GPU 索引和概率：六组 batch/width/kernel 测试与同图 dense GPU logits 的独立 double 精度 softmax 参考比较，分数误差上限 5e-6；另以禁用 fp16 的进程重复验证。
- 完整单元测试：87 通过，设备故障注入一项按既有 opt-in 规则跳过。
- GPU timestamp：沿用 VkInferenceSession.DebugProfile，每个 dispatch 取五次最小值，三次独立进程重复，并额外交换内核顺序重复三次。它不包含 HTTP、输入传输或主机等待；各 dispatch 最小值之和不是一次完整端到端墙钟。
- CPU 投影：实际 Small 模型权重，固定正弦特征输入，rows=8/80/320，threads=1/4，每个 case 预热 1 秒、计量 1 秒；修复前后交替三轮，并核对类别索引 SHA。
- HTTP：五版使用同一包装层、模型和六张图，4 并发，DOTNET_PROCESSOR_COUNT=16，无 CPU 硬配额。每组并发预热 10 秒、计量 60 秒，共四轮，组内顺序反转平衡。第一轮额外对每版采集 96 图×2 次质量响应；各轮内部对照的预热条件相同，跨轮绝对值不直接作为升级收益。

GPU timestamp 显示 online 的末端 dispatch 比原概率版缩短约 14%～26%，逆序验证约 12%～28%。候选的 96 图文字与坐标和原 GPU 概率版一致，分数最大差约 5e-7；这没有消除 GPU 路径相对 CPU 路径原有的字符边界及分数差异。

HTTP 四轮共 182,922 次正式请求全部成功：CPU softmax 修复使每请求 CPU 时间增加 14.64%，吞吐配对汇总下降 3.08%；GPU 修复前后吞吐汇总仅差 0.13%，未复现原先单轮的大幅回退。online 相对现 GPU 概率版汇总快 2.19%，但每轮分别为 +10.71%、-5.28%、-4.28%、+8.64%，CPU 时间还增加约 2.28%，没有足够证据支持切换默认。CPU 独立投影的 6 种形状/线程配置中，三轮 CPU 成本均增加，汇总范围约 14%～33%。

完整的多轮性能、CPU 成本、投影微基准、质量差异及剩余限制，见运维仓库 `ubuntu-server-20261122/softmax-isolation-20261122/README.md`。此次优化仍在显式实验开关之后，不能从微秒级内核收益直接推导正式服务吞吐。
