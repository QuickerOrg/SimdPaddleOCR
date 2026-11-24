# Linux OCR 精确缩放与运行配置优化

2026年11月23日。保留现有默认和公共API，不更新SimdOnnx子模块；新增 `SIMD_OCR_EXPERIMENT_EXACT_RESIZE=1` 显式实验开关，在进程启动前设置。正式服务未替换。

## 实现与边界

`Experimental/ExactImageResize.cs` 在PPOCRPreprocess的DET/CLS/REC入口接管支持的BGR24、NHWC、BGR归一化路径；其它格式、NCHW、非AVX2及不适用形状交回原ImageTensor。整页方向中心裁剪路径保持原实现。

保留OpenCV兼容的11位整数插值系数、水平累加、两段右移及垂直取整顺序，最后仍查原PixelNormalization表。AVX2一次计算8个通道元素，按通道偏移查同一归一化表，不使用近似浮点插值或更低精度概率。输出宽度之外的padding不写入。

横向系数以输入/输出宽度为key只读共享；最多1024项、数组有效载荷合计32MiB，每项输出宽度最多4096，额外尺寸或缓存满后使用临时池化表。缓存不保存像素或识别结果。线程协作保持原按行分配方式。

开关默认关闭，初始化后在该进程内固定。CPU架构、GPU模型、输入布局及应用并发不同，收益也会不同，不能直接作为所有平台的默认替代。

## 验证与证据

Linux CLI新增 `--resize-validate <输出JSON>`：600组、约759万float输出逐位匹配，覆盖边界、stride、tensor padding、向量尾部、放大/缩小及1/4线程；另有并发缓存、容量上限、同形状更换输入及不支持格式的回归测试。完整96图接口比较及实际吞吐结果以运维仓库 `ubuntu-server-20261123/linux-tuning/README.md` 为准。

EPYC服务器上的运行配置扫描显示，CPU可用核范围和缓存局部性对该负载影响较大；盲目增加线程或进程没有带来相应收益。这些配置只用于该服务器的隔离进程，不在库中硬编码CPU亲和性，也没有改全局调度或电源策略。

最终验收以同六图、正确softmax概率、GPU FP32末端进行，区分同并发速度与增加并发后的容量。历史Windows和Linux使用不同CPU/GPU/驱动，不将整机差距解释为操作系统的固定开销。模型和分数契约保持不变。


## 隔离候选组合

- `SIMD_OCR_EXPERIMENT_EXACT_RESIZE=1`：CPU整数缩放与归一化；`SIMD_OCR_EXPERIMENT_RESIZE_GATHER=1`进一步对横向插值使用AVX2 gather。gather严格避开源行末尾，剩余字节使用标量路径，不能靠源缓冲的额外padding避免越界。
- `SIMD_OCR_EXPERIMENT_EXACT_CROP=1`：AVX2可分离双三次采样。保持原坐标映射和字节舍入，距舍入不连续点1e-6内回原方法；不更换插值类型。
- `SIMD_OCR_EXPERIMENT_GPU_DET_FUSED=1`：GPU本地整数缩放＋原归一化LUT＋DET图，同一次提交；源数据和系数经host上传，不直接让GPU持续读取system RAM。只支持经过检查的BGR24、单输入FP16模型、NHWC与push descriptors配置，返回原概率图给原检测后处理。

GPU缩放池最多四个会话，缓冲随形状增长并在进程内复用；会话与设备/归一化绑定，设备重建后不能复用旧会话。实验GPU失败仍遵循现有fallback机制，但验收会将任何fallback判失败，防止把CPU回退计入GPU性能。

独立的GPU缩放往返、增多进程、扩大CPU核范围、SMT、CLS＋暂定REC联合提交及inline消费等方向没有实测可用收益。后两者的主库代码已撤下，探索快照和结果留在运维仓库，不因理论计算量较少而纳入候选。

HTTP测试新增`.NET --http-load-wire`工具，四个请求在途，以收包结束计时，并对除两个计时字段以外的完整JSON正文作哈希一致性校验。原Python口径仍保留；改变客户端的收益不能全部计作库优化。纯OCR的`--phase-profile`在串行与并发预热后计量，支持`SIMD_OCR_PHASE_ITERATIONS`指定每线程调用次数，四专用线程；它不包含PNG解码、映射、序列化和HTTP。

本轮所有源码开关默认关闭，固定引擎gitlink保持cb80be7370f280c4ca62f1679c71f05688a66001。实际Linux包使用运维归档的显存/index兼容引擎副本，主库源归档与实际运行包的区别需结合build-manifest和兼容源码还原，不能仅凭HEAD重建实验。


## 本轮最终结果

四并发、原Python口径三轮60秒合计134.60次/秒，P95轮均值56.27ms、服务器CPU25.30ms/请求，相对原Linux95.88次/秒改善约40.4%。相同.NET收包客户端下，Linux149.30、Windows本轮163.23次/秒，差8.53%；收包P95分别43.97–43.99和42.63–51.20ms。旧Python口径相对历史Windows仍慢18.45%，不能声称完全追平。不同客户端、不同CPU与不同GPU/驱动的边界见运维报告。

两条末端各96图、545块完整接口字段零差异；所有正式服务未替换。最后尝试的横纵插值直接合并原型收益不足0.5%、CPU略增，已撤下，其源和记录保存于运维原型目录。最终库SHA55c4607cf7d581998278bfefaedd547a5b7581341853330fc255a25fc167de63与三轮验收库一致。
