# 闲置推理会话回收与共享模型数据

新策略是可选的：默认不改变现有会话保留行为。执行并发由应用层决定；`MaxPooledSessions` 限制的是归还后最多缓存多少会话，不限制正在执行的请求数。

## 使用方式

```csharp
var pool = new OcrSessionPoolPolicy
{
    IdleTimeout = TimeSpan.FromSeconds(60),
    MinimumRetainedSessions = 1,
};
var options = new PaddleOcrOptions
{
    Detector = new() { SessionPool = pool },
    Classifier = new() { SessionPool = pool },
    Recognizer = new() { SessionPool = pool },
};
```

各阶段分别拥有自己的池和最小保留数。`MinimumRetainedSessions` 自动限制在该阶段的 `MaxPooledSessions` 以内，因此原有 `MaxPooledSessions=0` 仍代表不缓存。空闲时长必须为正，保留数不能为负。

隔离实验还可设置 `SIMD_OCR_EXPERIMENT_SESSION_IDLE_SECONDS` 和 `SIMD_OCR_EXPERIMENT_SESSION_IDLE_KEEP`。显式 `SessionPool` 优先于实验环境变量；默认没有这些变量且 `SessionPool=null` 时，继续使用原来的池。环境变量入口为实验用途，尚未作为服务器正式配置部署。

定时器按约半个空闲期限检查，周期限制在100ms到30s之间。回收可能晚于设定期限一个检查周期，线程池繁忙时还可能延迟。无需下一个请求触发回收。

## 所有权和动态扩容

- 租借移除池中会话。池为空时，检测器/识别器在锁外创建新会话；已编译模型保持共享。
- 归还记录单调时钟时间。后台仅摘除已归还且过期的会话，保留最近使用的热会话。
- GPU/推理执行以及驱动资源释放都不占用池锁。回收前先在锁内转移所有权，租借者不可能再取到同一被回收会话。
- 池关闭后才归还的会话直接销毁，不重新入池。整个OCR对象的Dispose仍应在应用层排空请求后执行，不把旧的模型销毁语义改成可与任意在途推理并行。
- 识别器保留原有best-fit规则：优先取足够大且最小的会话；都不够大时选择最大已有会话，避免所有会话被不同尺寸逐渐撑大。
- 模型、权重、CPU计划以及全局GPU图缓存不因闲置会话回收而主动清空。回收面向可变激活、批次arena、输入/输出暂存、会话命令资源。

持续8并发时仍可能需要原来的显存峰值；这项机制降低的是空闲保留量。再次突发要重新分配和记录部分会话资源，需要测试首批与尾延迟，而不是只看热机吞吐。

## 只读信息共享边界

同一检测器或识别器的所有会话已经共享同一个`CompiledModel`。CPU模型和打包投影不按请求重新创建；Vulkan模型常量按`(VkDevice, 源OnnxModel对象, f16)`共享，计算图按模型/尺寸/编译选项复用。新的会话只需要自己的可变状态。

不同`OnnxModel`对象不会因为文件名相同就自动全局去重。这避免了内容不同却同名、设备或精度不同的错误共享。多个OCR引擎希望共用一个已加载模型时，可通过已有的逐模型构造函数借用同一组`Model`对象，并由外层统一管理模型生命周期；不要让多个具有所有权的`PaddleOcrAll(PaddleOcrModelSet)`同时负责销毁同一套模型。

可安全共享：只读权重、字典、同键的打包常量、兼容尺寸/选项的编译计划。不可直接让并发请求共用同一块可写激活arena、KV/中间张量、命令缓冲或提交/等待状态；跨流复用必须有严格的生命周期和同步证明。

因此本次没有增加另一套全局权重缓存，也没有把不同请求的可写显存强行合并。跨独立OCR引擎的模型加载合并是另一个所有权问题，收益需按实际重复权重测量，不等同于回收数GiB的会话工作缓冲。

## 验证

`IdleSessionPoolTests`覆盖过期/保留、租出会话不被释放、best-fit、零缓存上限、关闭后归还、无请求时的定时回收，以及并发租借/归还/回收的资源所有权。

服务器A/B及共享常量分配追踪记录位于运维仓库`ubuntu-server-20261124/ocr-idle-pool`。新机制目前仅在独立候选包中测试，未替换正式服务。
