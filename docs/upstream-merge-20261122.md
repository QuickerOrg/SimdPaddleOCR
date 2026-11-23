# 2026-11-22 上游同步与 GPU 末端适配

从 `99d0fa6` 的 GPU 实验分支合并上游 `a9324e3`，合并提交为 `9df60ef`。

## 上游变化

- `a79f896`：不再引用独立 Sdcb.SimdOnnx NuGet 包，改为递归初始化 `external/SimdOnnx` 子模块，将引擎编译进 OCR 程序集；引擎类型为 internal。当前子模块固定在 `cb80be7370f280c4ca62f1679c71f05688a66001`。
- `45ce328`：修正 netstandard2.0 测试中 ISA 标识类型冲突。
- `7956975`、`b2ad0ad`、`4323f03`：整理源码内嵌前后的同机性能数据及文档，移除旧性能表引用。
- `102ef06`：融合 CTC 投影返回胜出类别的 softmax 概率，修正此前 RecognitionScore 直接使用原始 logit 的问题。引擎所有 CPU ISA 路径同步维护 softmax 分母。
- `a9324e3`：新增整页 0/90/180/270 度方向分类、旋转和坐标还原 API；独立于 PaddleOcrAll，不自动插入既有识别流程。

## 合并处理

唯一文本冲突在 PipelineProfiler：保留实验的 `rec_projection` 槽位 21，将整页预处理、推理槽位顺延至 22、23，StageCount 为 24。两项实验环境开关仍默认关闭。

GPU 紧凑末端原本也返回原始 logit。为适配新的 CTC 分数约定，着色器保留 ArgMax 和索引平局规则，使用稳定的 `exp(logit - max)` 求和，返回 `1 / sum`。求和使用 fp32，六个字节槽的索引/分数编码保持不变；fp32 和 fp16 存储的 SPIR-V 均已重编。GPU 单元测试同时检查类别索引与独立 double 精度 softmax 参考值。

本次没有修改检测模型、文本模型、默认选项或生产服务；整页方向分类未在 HTTP 压测中启用，既有文本行方向分类保持启用。

## 验证

- net10.0 完整单元测试：83 通过、0 失败，1 项主动设备故障模拟按设计跳过。
- netstandard2.0 构建通过；内嵌上游 Intrinsics.cs 存在一项既有 CS0108 警告，未改变其兼容语义。
- 本机 HTTP 比较以旧 memfix 发布包为历史基线，分别关闭/开启 GPU 末端，与合并后的完整内嵌引擎构建比较。新版本实际使用内嵌引擎；旧发布目录残留的独立引擎 DLL 不参与新 OCR 推理。
- 测试包装层同时适配独立、内嵌引擎的显卡诊断，通过实际引擎类型反射获取设备，不误报旧引擎实例。

具体吞吐、CPU 成本、96 图逐字段质量比较和二进制散列，见运维仓库 `ubuntu-server-20261122/upstream-merge-20261122/README.md`。此次比较涵盖引擎版本及分数计算变化，不能把全部差异单独归因于源码内嵌方式。

## 构建提示

新 checkout 必须执行 `git submodule update --init --recursive`，不能继续依赖旧 `UseLocalSimdOnnx` 参数。实验着色器编译使用现有 glslang：

```powershell
& artifacts/gpu-experiments/glslang/bin/glslang.exe -V src/Sdcb.SimdPaddleOCR/Experimental/Shaders/ctc_compact.comp -o src/Sdcb.SimdPaddleOCR/Experimental/Shaders/ctc_compact.spv
& artifacts/gpu-experiments/glslang/bin/glslang.exe -V -DF16=1 src/Sdcb.SimdPaddleOCR/Experimental/Shaders/ctc_compact.comp -o src/Sdcb.SimdPaddleOCR/Experimental/Shaders/ctc_compact_h.spv
```
