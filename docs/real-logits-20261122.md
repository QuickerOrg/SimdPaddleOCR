# 真实 logits 分布和 CTC 近乎平局案例

本轮只增加实验 CLI `RealLogitsExperiment`，没有修改识别库、GPU/CPU内核、分数精度或默认开关，也未部署服务。

## 结果

- 先测六张性能图与英文文章，再扩到96图。统一Small常规检测路径中60图有输出，509行、20676个预测位置；36图无框不参与统计，未模拟HTTP小图单行/补边快路径。
- T=23：最终最大值理想向量跳过率87.10%，当前逐lane运行最大值仅0.000279%，首tile广播65.48%。T=25首tile机会57.70%。因此简单加入原lane跳过不值得，首tile行种子值得进入后续原型，但没有测得CPU/RPS收益。
- 首tileT=25的double截断模拟最大绝对误差5.85e-8；它不包含未来实现的ExpCore/FP32误差。现有CPU分母相对double参考已有输出位置最大约3.79e-5差异，需单独建账。
- 英文文章第92步CPU空格logit21.331380、blank21.327156；FP16输出量化后都为21.328125，blank按最小索引胜出。空格run从[92,94)移到[93,94)，首帧分数0.50105435→0.99821067，共边548.17975→550.8456。
- 同一份GPU特征的projection-only图逐元素复现完整GPU logits，最大差0。CPU只做输出FP16舍入就能复现该CTC变化；单独量化权重仍选择空格。
- 两次独立捕获的输入、特征和六种密集logits张量解压SHA一致。不能用修改softmax或偏向非blank的平局规则修复已丢失的排序信息。

## 诊断方法

从完整OCR结果重建原透视裁剪、方向处理、归一化和same-width/max16批量形状，使用原Vulkan截断会话取得真实特征。CPU dense投影保留同一FMA顺序并单独处理标量列尾；逐位置验证ArgMax一致，重放CTC结果与原流水线的文字、run和分数匹配，行分数最大差0。

计数器严格按8个元素调用ExpCore的实际粒度统计，不把一个有效候选当作一次标量指数。前三种下界为理想最终最大值、当前lane最大值、首tile最大值广播后的有效lane界；首16列强制执行，标量尾部单独统计。统计和截断数学模拟不是优化内核。

启动有三组解析fixture：已知75%/0%/50%向量机会、原lane初始化与种子条件的差别、非整tile列尾winner。所有实测位置检验C×exp(−T)上界。诊断阶段不测吞吐。

## 命令

```powershell
dotnet build test/Sdcb.SimdPaddleOCR.GpuExperiments/Sdcb.SimdPaddleOCR.GpuExperiments.csproj -c Release --artifacts-path T:/Compile/build/SimdPaddleOCR-real-logits-20261122
dotnet T:/Compile/build/SimdPaddleOCR-real-logits-20261122/bin/Sdcb.SimdPaddleOCR.GpuExperiments/release/Sdcb.SimdPaddleOCR.Benchmarks.dll --real-logits <Small模型目录> <图片清单.json> <输出目录>
```

模型目录含det.onnx、cls.onnx、rec.onnx、ppocrv6_dict.txt。图片清单为Id/Path/FocusText对象数组，FocusText指定需要保留详细张量的完整行文本。CLI核验本机RTX4080，当前计数器限定AVX2且每行T满足实际packed shape gate。

运维仓库 `ubuntu-server-20261122/run-real-logits.py` 提供独立进程及显存/温度/时间保护，`summarize-real-logits.py`汇总并核验重复捕获。模型由历史F盘备份按旧hash恢复，未更换模型。

完整报告：运维仓库 `ubuntu-server-20261122/real-logits-20261122/README.md`。原始张量为gzip压缩little-endian float32；问题行logits[200,18710]、特征[200,120]。原始数据和诊断二进制进入受限F盘备份，不放入源代码仓库。
