# GPU 识别末端迁移实验（2026-11-11）

后续针对顾问反馈的并发扫描、最终版本单路延迟和单波试验，见[补充复核报告](gpu-concurrency-followup-20261111.md)。本页保留初轮口径。

## 结论

已在用户 fork `QuickerOrg/SimdPaddleOCR` 的本地分支 `codex/gpu-ocr-experiments` 完成原型与对照。

**主要收益是减少CPU算力，不是稳定提高吞吐。** 主对照两轮各30秒、4路并发，原CPU投影末端平均141.68张/秒、16.52个逻辑核等效占用；GPU紧凑末端136.10张/秒、5.45核。吞吐约下降3.94%，CPU占用下降67.0%，每图CPU时间下降65.7%。候选默认关闭，尚不建议直接替换产品。

90图对照全部成功且各自重复稳定，文字及检测框90/90一致；53张存在识别分数变化，最大行分数绝对差0.033354；1张英文图的一个CTC起点移动一列。字符框使用者仍有兼容风险，不能称作无损替换。

## 版本与范围

- 本地仓库：`D:/Work/SimdPaddleOCR`，origin为用户fork，基线 `9b82a859c615f06e22d7f5cafdc35b4ba2f398ad`。
- 当前fork为3.0开发版，使用固定 `Sdcb.SimdOnnx 0.1.0-preview.8`，包元数据记录引擎提交 `cf1490486ba488184f3c5f3210acf965843344a3`。不是此前qk-ocr-lite使用的2.0.0。
- 没有改引擎包、版本线、权重、字典、输入分辨率、方向开关、REC批大小或生产默认；GPU扩展使用引擎公开的GraphPass/ExtOp接口，没有修改Sdcb.SimdOnnx源码。
- 机器：Core Ultra 7 270K Plus、RTX 4080、Windows 11、.NET SDK 10.0.403；Release JIT、无调试器。
- Small模型复用本机 `F:/Docs/Quicker/_packages/quicker.ocr.v6small/0.2.0`，方向分类固定CPU。额外的GPU分类探索另记，不混入主表。
- 4张图仍是通知、中文新闻、英文文章、收据；开CTC对齐。PNG解码后直接以BGRA32传入库，包含解码、OCR和字符对齐，排除磁盘读取、HTTP/Base64、qk-ocr-lite的框收紧/版面后处理。批次墙钟还包含DTO序列化一致性检查。

因此本表**不能和此前产品Core约75.7张/秒相除来计算优化倍数**：版本和调用边界均不同。

## 实际做了什么

基线流程：GPU识别主干→回传特征→CPU融合字典投影/ArgMax→CTC解码。

实验流程：GPU识别主干→GPU字典投影→GPU ArgMax→只回传紧凑编号/分数→CPU CTC解码。

- 通过GPU编译扩展替换终端Softmax为紧凑ArgMax输出；字典投影仍使用引擎的GPU GEMM。
- 支持单批及多形状Vulkan批次，沿用现有会话池、波次调度和错误回退。
- 单个时间步输出6个存储元素：两字节字符编号、四字节分数位模式。每字节拆成0–255数值，保证fp16存储不会把大于2048的字符编号四舍五入；分数的32位位模式也可完整还原。
- 基线回传的是特征，不是完整词表平面；不能用“词表长度/6”声称本次传输收益。候选在显存中仍产生完整投影结果，尚未实现GEMM和ArgMax单内核融合。
- 额外增加 `rec_projection` 分阶段计时，覆盖批量/单批CPU末端。计时默认关闭。

实验开关：启动进程前设置 `SIMD_OCR_EXPERIMENT_COMPACT_HEAD=1`。仅验证.NET 10 Vulkan，匹配rank-3、1024–65536项词表的终端Softmax；不是公开产品配置。未开启时继续原路径。

## 分阶段证据

首轮1路、5秒、有采样的稳定后续运行共186次调用，累计：CPU投影1459ms、rec_graph2266ms、各行cls_graph1906ms、det_graph700ms、检测预处理343ms、裁剪291ms、识别预处理149ms。

这些计数包含嵌套和文字行并行，**不能相加为端到端耗时，也不能直接当作CPU时间占比**。它证明CPU末端值得实验，最终CPU收益以进程CPU时间度量。

## 30秒主对照

同一共享引擎4个独立调用线程，预载图片、串行建立结果基线、并发预热；每个进程正式30秒，截止后不再发新调用，仅等待在途任务完成。吞吐只计30秒内完成的结果。按A/B/B/A顺序启动独立进程；正式计时关闭采样。

| 方案 | 第一轮张/秒 | 第二轮张/秒 | 平均张/秒 | 平均CPU核等效 | CPU毫秒/图 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A：CPU投影末端 | 140.63 | 142.73 | 141.68 | 16.52 | 116.70 |
| B：GPU紧凑末端 | 137.60 | 134.60 | 136.10 | 5.45 | 40.05 |

四组共16667次窗口内调用，各组完整DTO与各自串行基线全部一致，无GPU失败/回退。跨A/B的分数差异见下节。

清理后的补充确认轮：A为101.43张/秒、12.19核、120.26 CPU毫秒/图；B为132.33张/秒、5.33核、40.31 CPU毫秒/图左右（精确值见JSON）。两者使用同一库二进制SHA256。A绝对吞吐出现明显波动，但每图CPU时间仍接近原两轮；B维持接近原表现。没有同步采样证明波动来自后台竞争、线程池或其他因素，因此**不把这轮B高于A当作稳定加速证据，也不隐藏该轮数据**。

观察到候选GPU负载接近饱和；没有完成GPU逐算子时间分解，不能把略降速直接归因于某一个内核。降低CPU需求不等于整机更省电，本轮没有测插座功耗。

## 90图回归与分数口径

同一目录90张PNG逐图读取，两种方案分别独立进程；每图调用两次，序列化结果逐次相同。包含微图及文档，不含本仓桌面侧微图补边策略，也不是完整450图独立精度评测。

- 180/180图/方案组合无异常，全部会话仍活跃，无回退日志。
- 全文90/90相同，检测框90/90相同。
- 53/90有行识别分数变化，最大绝对差0.033354；不是0.033个百分点。
- `ui-03-english-article.png` 第7行第36个CTC span（均按1起计）`StartColumn`由92变93。文字不变，但可能影响估计字符框。

GPU投影使用引擎fp16存储路径，与CPU融合投影的数值精度/归约方式不同。本轮未把这类差异归因到单个算子。

**另一个已发现的主线问题：** 当前引擎 `CtcProjection.ArgMax/ArgMaxUnits` 返回最大logit，库紧凑解码路径将其直接写入识别分数字段；实际主线有大于1的分数。本实验保持这个既有最大logit口径，没有把分数归一化修复混入性能实验。最初尝试返回概率会改变输出尺度，该尝试已撤回。`compact-probe.json`（59次mismatch）及`compact-details.json`（166次mismatch）均来自修复前原型；不能拿它们的单路/并发性能代表最终版。`compact-fixed.json`和正式轮才是修复后的数据。后续应单独确认/修复库与引擎的分数契约，不能把本报告的分数当作校准概率。

## 其他探索

- 5秒有采样短跑，将CPU投影线程/预处理线程限制为4/4：约128.8→95.8张/秒，CPU16.72→10.90核。仅减少线程会损失吞吐，不作为本轮推荐。
- 8秒探索GPU分类、默认预算：约136.5张/秒、14.88核；GPU分类加紧凑末端约133.25张/秒、4.10核。只覆盖4图、未按主表做完整回归/多轮，不推广为正式结果。
- CPU/GPU交替分担不同单位末端的10秒原型约141.6张/秒、11.94核，但首请求与热请求路径不同，354条完整DTO基线比较不通过，未当作合格方案；该分摊代码已撤回。

## 验证、代码和复现

- 新增数值验证4项：大字符编号/分数位编码、长度拒绝、两种batch/width下与GPU稠密投影逐行ArgMax及最大logit精确一致。
- 完整UnitTests：62通过、1跳过（需显式开启的设备丢失测试），共63。没有执行故意制造GPU设备丢失的测试。
- net10.0与netstandard2.0 Release构建通过，无警告；Python汇总核验和 `git diff --check`通过。
- 未验证其他GPU/CPU、Linux、Metal、medium/tiny端到端、大图长期显存、生产Server/Agent集成或混淆包。

主要文件：`Experimental/GpuCompactHead.cs`、`Experimental/Shaders/ctc_compact.comp/.spv`、`OnnxSharp/GpuOcrSession.cs`、`PaddleOcrRecognizer.cs`、`PipelineProfiler.cs`；工具在 `test/Sdcb.SimdPaddleOCR.GpuExperiments`，数值测试在 `CompactGpuHeadTests.cs`。

原始数据在gitignore的 `artifacts/gpu-experiments/`：完整profile、A/B/B/A和第三轮、90图输出、探索数据及summary。模型/输入/二进制哈希见环境记录和JSON；构建/测试完整日志在 `T:/TEMP/simd-*.log`。没有提交或推送远端，没有修改qk-ocr-lite产品依赖。

```powershell
$buildRoot=Get-Content artifacts/gpu-experiments/build-root.txt
dotnet build test/Sdcb.SimdPaddleOCR.GpuExperiments -c Release --artifacts-path $buildRoot
$dll="$buildRoot/bin/Sdcb.SimdPaddleOCR.GpuExperiments/release/Sdcb.SimdPaddleOCR.Benchmarks.dll"
$models='F:/Docs/Quicker/_packages/quicker.ocr.v6small/0.2.0'
$images='D:/Work/qk-ocr-lite/fixtures/images'
# 0为基线，1为候选；正式对照按0/1/1/0顺序运行，各用不同输出名。
$env:SIMD_OCR_EXPERIMENT_COMPACT_HEAD='1'
dotnet $dll $models $images artifacts/gpu-experiments/repeat.json 30 4 0 0 --decode
# 90图对照同样分别设置0和1，再输出到不同文件。
dotnet $dll --corpus $models $images artifacts/gpu-experiments/repeat-corpus.json
Remove-Item Env:SIMD_OCR_EXPERIMENT_COMPACT_HEAD
$env:PYTHONUTF8='1'
python test/Sdcb.SimdPaddleOCR.GpuExperiments/summarize.py artifacts/gpu-experiments
dotnet test test/Sdcb.SimdPaddleOCR.UnitTests/Sdcb.SimdPaddleOCR.UnitTests.csproj -c Release -f net10.0 --artifacts-path $buildRoot
```

Shader由glslang16.6.0编译，SPIR-V随实验源码保存；重编命令为 `glslang -V ctc_compact.comp -o ctc_compact.spv`，fp16版另加 `-DF16=1`并输出`ctc_compact_h.spv`。本机编译器位于 `artifacts/gpu-experiments/glslang/bin/`。

## 后续建议

优先解决分数契约与字符对齐精度，再考虑推广。若目标是低CPU占用，当前原型已经值得进一步评估；若目标是更高吞吐，需要继续优化GPU GEMM/ArgMax融合及显存中间平面，不宜把“把矩阵移上GPU”直接等同于加速。所有默认开关保持原状。
