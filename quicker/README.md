# Quicker 专用分支

维护分支为 `codex/quicker`。使用方通过Git子模块固定具体提交，不在构建时自动获取分支最新提交。

本分支基于已验证的GPU优化栈，并包含 `9e4eb84` 的可选闲置会话回收。默认库行为仍保持原有会话保留方式；宿主通过公开 `SessionPool` 配置启用回收，执行并发不由空闲保留数限制。

## 可复现构建

```sh
git clone --branch codex/quicker --recurse-submodules https://github.com/QuickerOrg/SimdPaddleOCR.git
dotnet build src/Sdcb.SimdPaddleOCR/Sdcb.SimdPaddleOCR.csproj -c Release -f net10.0
```

`external/SimdOnnx`固定在 `cb80be7370f280c4ca62f1679c71f05688a66001`。`QuickerVulkanCompatibility.props`使用本目录内六个兼容文件替换同名上游文件，内容来自此前已验证的固定引擎副本，包括显式选卡、分配失败兜底、内存策略和分配追踪。不修改上游子模块工作区，不要求Python预处理或手工替换DLL。

兼容代码为上游MIT许可文件的派生版本，许可副本在 `SimdOnnx/LICENSE`。升级引擎gitlink时须逐文件合并兼容差异并重新验证，不能只更新gitlink。

`QuickerVulkanCompatibility=false`可用于对照原始引擎；不是生产默认。兼容文件的编译只适用于net10.0，netstandard2.0继续使用原CPU实现。Linux小型CPU可见显存堆的部署仍须显式选择已验证的 `SIMD_OCR_EXPERIMENT_VK_MEMORY_POLICY=device-local`，不能将该策略无条件用于所有UMA设备。

## 宿主集成

服务端直接ProjectReference本仓库项目，与自身源码一起构建。模型包可继续引用已有的1.0.0版本。不得让服务端仍解析旧推理NuGet并在发布后手工覆盖DLL。

闲置策略的API、资源所有权及共享模型边界见[会话池说明](../docs/idle-session-pool-20261124.md)。非线程安全的激活和命令状态仍按会话隔离，只读权重和兼容编译计划继续共享。

宿主可用 `OcrBackendDiagnostics.GetVulkanDeviceName()` 在预热前核验所选显卡，避免依赖内部引擎反射或临时测试Controller；后端不可用时抛出异常，netstandard2.0不支持Vulkan。
