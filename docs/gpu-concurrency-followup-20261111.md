# 顾问反馈复核：并发扫描与单波提交（2026-11-11）

## 结论

顾问提出“区分单路延迟和并发吞吐、补并发扫描”的方向正确。重测最终版本后确认：**GPU紧凑末端有单路延迟优势；将紧凑路径改为单波，在批量场景又有明确收益。** 初轮只突出4路吞吐，结论不够完整。

本轮执行了1/2/4/6/8路扫描、单波对照、关键点30秒确认及GPU内核时间戳诊断。没有执行跨请求合批、多队列或重新引入CPU/GPU分流；本机未安装Nsight，仍缺完整队列时间线，不能据此断言GPU瓶颈是“派发而非算力”。

## 先修正数据和推断口径

1. 顾问引用的B单路47.2张/秒、21.0ms来自 `compact-probe.json`，是修复前返回概率/冷路径不同的原型，59次完整DTO不一致。`compact-details.json`的166次不一致也是修复前数据。这两份已在原报告中明确标为排除项，不能代表最终B。最终B重新测量后仍然较快，见下表。
2. CPU占用16.5/24核，不能单独证明没有资源瓶颈；内存带宽、线程池调度、不同CPU核能力和GPU等待都会影响扩展。扫描后A从4到6路继续提高，6到8路基本持平，不能继续按线性外推。
3. `1000/136≈7.4ms`只是完成间隔，不是GPU执行时间。不能据此计算单路GPU忙碌率35%；不同并发下时钟、布局及CPU/GPU重叠都可能变化。
4. 源码显示三波在 `Submit` 中一次性全部排入GPU队列，然后主机按波等待。不是“提交一波、等完、再提交下一波”。单波改变的不仅是提交次数，还包括单位交错范围、屏障数量及CPU消费者唤醒时机。尚不能精确归因哪项贡献最大。

## 环境及控制

沿用本地fork分支 `codex/gpu-ocr-experiments`，基线提交 `9b82a859c615f06e22d7f5cafdc35b4ba2f398ad`，固定 `Sdcb.SimdOnnx 0.1.0-preview.8`（引擎提交cf149048）。Core Ultra 7 270K Plus、RTX 4080、Windows 11、Release、无调试器。

仍是同4张通知/新闻/英文文章/收据，Small 0.2.0模型，方向分类CPU、CTC对齐开启、PNG解码计入。排除HTTP/Base64、磁盘读取、加载和预热，以及qk-ocr-lite自己的版面/框后处理。吞吐墙钟包含完整DTO序列化一致性检查；调用延迟不含该检查或等待队列。

- A：GPU识别主干＋CPU字典投影/ArgMax，原三波。
- B：GPU识别主干＋GPU紧凑末端，三波。
- C：与B相同，仅紧凑路径改为一波。

所有测量关闭分阶段采样。用与测量相同的独立调用线程模型预热8秒，再建立热状态基线。扫描每配置2轮×12秒，第二轮反向遍历并发度及A/B/C顺序。关键点另各跑30秒，顺序A1/B1/C1/C6/B6/A6。

Windows电源计划在阶段前后均检查为同一个Balanced计划，没有修改用户系统设置；未锁CPU/GPU时钟、未关闭桌面程序。固定计划不等于消除了所有环境噪声。

## 并发曲线：每点两轮均值

| 并发 | A 张/秒 | B 张/秒 | C 张/秒 | A 平均ms | B 平均ms | C 平均ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.71 | 49.38 | 47.96 | 25.05 | 20.12 | 20.71 |
| 2 | 80.54 | 94.25 | 105.50 | 24.68 | 21.06 | 18.80 |
| 4 | 129.92 | 135.79 | 163.58 | 30.62 | 29.29 | 24.30 |
| 6 | 159.25 | 148.67 | 178.21 | 37.47 | 40.16 | 33.49 |
| 8 | 158.92 | 146.25 | 179.17 | 50.10 | 54.44 | 44.45 |

延迟按该点所有完成调用汇总，吞吐按等长窗口的完成数汇总。不能把6路的平均延迟直接除以6当作单张交互延迟。

扫描范围内的峰值：A约159.25（6路）、B约148.67（6路）、C约179.17（8路）。C从6到8路吞吐仅增约0.54%，平均延迟从33.49升到44.45ms，所以**本负载更倾向C的6路作为批量均衡点**。这些是1–8路范围内的观测峰值，不是所有硬件/并发/模型的全局上限。

## 关键点30秒确认

| 方案 | 并发 | 张/秒 | 平均延迟ms | CPU等效核 |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 34.03 | 29.22 | 4.21 |
| B | 1 | 47.93 | 20.72 | 1.84 |
| C | 1 | 47.17 | 21.08 | 1.86 |
| A | 6 | 142.17 | 42.04 | 16.75 |
| B | 6 | 143.70 | 41.59 | 5.94 |
| C | 6 | 176.13 | 33.90 | 7.02 |

- 最终B单路延迟确实低于A：扫描约降低19.7%，30秒确认约降低29.1%。因此存在交互延迟收益，但不能承诺固定22%或29%。
- C的6路相比B约提高22.6%吞吐，平均延迟也降低约18.5%。相比同轮A提高约23.9%，进程CPU占用降低约58.1%。
- 单波不是单路的明确赢家：C单路略慢于B；批量时C更好。因此本轮保留可选开关，不强制所有GPU调用单波。
- A仍有波动：扫描6路159.25、确认142.17；4路两轮138.75/121.08。B/C整体更稳定，但本实验仍不能提供生产SLA或统计置信区间。不能将最有利的一轮包装成普遍性能。

## GPU时间戳与引擎事实

本机没有 `nsys` 或Nsight Systems安装。本轮使用固定引擎内部 `VkInferenceSession.DebugProfile`：在同一次命令缓冲中，对每次dispatch前后写GPU时间戳，重复5次。以下是**每个内核各自最小时间的和**，不是某一帧实际总GPU忙碌时间；还排除了上传、回读、主机排队和请求流水线，不能拿它与端到端延迟相除计算占用率。

使用独立REC形状、合成浮点输入，比较主干截断输出与GPU紧凑末端：

| batch / 输入宽 | 主干内核和ms | 含紧凑末端ms | 其中投影GEMM ms | ArgMax ms |
| --- | ---: | ---: | ---: | ---: |
| 1 / 320 | 0.821 | 0.853 | 0.022 | 0.005 |
| 1 / 800 | 0.844 | 0.891 | 0.043 | 0.006 |
| 1 / 1152 | 0.880 | 0.960 | 0.061 | 0.006 |
| 8 / 320 | 1.067 | 1.185 | 0.101 | 0.008 |
| 16 / 320 | 1.582 | 1.785 | 0.193 | 0.011 |

实际派发名称包含 `conv_cm`、`conv_cm_n32`、`conv_cm_n64`：主干已经使用协作矩阵内核，不能假设主干还未启用该能力。新增词表投影为普通 `gemm(MatMul.12)`，随后是 `ctc_compact`。是否值得继续改投影的张量核路径，应依据它在真实负载中的占比，不能只看理论峰值。

固定引擎创建一个计算队列（`QueueCount=1`），三波在等待前已全部提交；单波的单位组由约8–12个单位统一交错执行。源码中的全单位arena布局不按波次复用，单波并非新增三倍单位arena，但本轮未测实际显存峰值。

参考：[固定版本VkUnitRunner](https://github.com/sdcb/SimdOnnx/blob/cf1490486ba488184f3c5f3210acf965843344a3/src/Sdcb.SimdOnnx/Backends/Vulkan/VkUnitRunner.cs)、[固定版本VkDevice](https://github.com/sdcb/SimdOnnx/blob/cf1490486ba488184f3c5f3210acf965843344a3/src/Sdcb.SimdOnnx/Backends/Vulkan/VkDevice.cs)。诊断使用内部方法，升级引擎需重新验证。

另外，4张图实际CTC时间步总数分别为1352/1624/2012/968，平均1489，并非约700。按18710列、fp16粗算，完整投影平面约55.7MB/图；吞吐179时约10GB/s。它仍远低于显卡标称带宽，但仅凭这个数字不能严格限定融合收益小于10%，因为还涉及屏障、派发和计算方式。

## 正确性与验证

- 扫描30组共43610次窗口内调用，30秒确认6组共17734次，总计**61344次**。全部与各自热状态完整DTO基线一致，无GPU回退日志、无失败会话。
- 所有扫描/确认使用同一库二进制SHA256和同样输入哈希，模式内跨轮次/并发的基线也一致。
- B/C在90图上各自全部成功且重复稳定，所有结果字段精确一致（未设置浮点容差）。单波未新增分数、文本、检测框或CTC对齐差异。
- 原来B相对A的53张分数变化、1张字符对齐变化，以及最大logit分数契约问题仍存在；本轮不宣称已解决，也没有修改分数定义。
- 新增6个不同宽度单位的三波/单波数值测试，核实真实runner分别3/1波，逐单位输出相同。
- 完整测试：**63通过、1跳过，共64**；跳过项仍是需显式开启的设备丢失测试。net10.0/netstandard2.0构建及脚本语法/数据核验通过。

## 改动与开关

Core仅增加实验单波开关及对应条件：

```text
A: SIMD_OCR_EXPERIMENT_COMPACT_HEAD=0
B: SIMD_OCR_EXPERIMENT_COMPACT_HEAD=1, SIMD_OCR_EXPERIMENT_SINGLE_WAVE=0
C: SIMD_OCR_EXPERIMENT_COMPACT_HEAD=1, SIMD_OCR_EXPERIMENT_SINGLE_WAVE=1
```

都在启动前设置，不支持在同一批次执行中动态改环境变量。GPU紧凑末端仍默认关闭；单波开关也默认关闭。生产默认、公开API、模型、依赖版本、服务器和桌面Agent均未切换。

工具补了热状态基线、匹配调用线程的定时预热、波数/单位数、p95、单波标记、内核诊断及独立核验脚本。新增脚本在 `test/Sdcb.SimdPaddleOCR.GpuExperiments/`。原始数据在gitignore的 `artifacts/gpu-concurrency-followup/`，完整日志 `T:/TEMP/simd-followup-*.log`。代码保留于本地分支，未提交或推送。

```powershell
$buildRoot=Get-Content artifacts/gpu-experiments/build-root.txt
dotnet build test/Sdcb.SimdPaddleOCR.GpuExperiments -c Release --artifacts-path $buildRoot
$dll="$buildRoot/bin/Sdcb.SimdPaddleOCR.GpuExperiments/release/Sdcb.SimdPaddleOCR.Benchmarks.dll"
./test/Sdcb.SimdPaddleOCR.GpuExperiments/run-concurrency-followup.ps1 -BenchmarkDll $dll
./test/Sdcb.SimdPaddleOCR.GpuExperiments/run-followup-confirmation.ps1 -BenchmarkDll $dll
$env:PYTHONUTF8='1'
python test/Sdcb.SimdPaddleOCR.GpuExperiments/summarize-followup.py artifacts/gpu-concurrency-followup
dotnet $dll --gpu-profile F:/Docs/Quicker/_packages/quicker.ocr.v6small/0.2.0 artifacts/gpu-concurrency-followup/gpu-kernels-extended.json
```

汇总还要求输出目录存在B/C的90图回归文件；分别设置对应开关，使用 `--corpus <models> <images> <output.json>` 生成 `corpus-B.json`、`corpus-C.json`。重复执行会覆盖同名结果，保留历史时请改输出目录。

## 后续优先级

单请求交互优先验证B；批量任务优先验证C的4–6路，而不是直接堆并发。仍需在目标服务器和真实混合语料上验证。

下一步先抓真实Nsight队列时间线，确认C中是否仍有主机派发空隙，再决定跨请求同宽度合批。仅保持宽度相同也不能直接承诺数值一致，batch变化可能改变内核选择，必须验证“与谁合批都不改变结果”。多队列涉及引擎与同步模型，尚未实现；CPU/GPU分流的冷/热路由一致性可修，但暂不引入第二套数值策略。融合和投影张量核优化继续放在测量之后。
