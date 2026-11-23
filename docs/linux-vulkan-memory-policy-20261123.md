# Linux Vulkan 显存分配隔离修复

2026年11月23日。基于5ee11bd的OCR实验代码、固定引擎cb80be7，仅在独立引擎副本验证；本仓库代码及引擎gitlink保持原状，现有服务未替换。

目标4080 SUPER有32 GiB显存，但CPU可见设备本地堆只有246 MiB。旧策略将权重及输入暂存优先放入该堆，导致权重分配失败后落系统内存，并出现驱动映射告警。新策略显式开启后，权重进入普通GPU显存、暂存进入系统内存，通过既有Upload复制，计算精度和算法不变。

实现与完整证据在运维仓库 `ubuntu-server-20261123/linux-memory/`：apply_memory_policy.py、engine-memory-policy.patch、README.md。开关为 `SIMD_OCR_EXPERIMENT_VK_MEMORY_POLICY=device-local`，默认关闭。没有直接修改第三方子模块。

两图追踪显示16,833,216 bytes权重全部进入普通显存，原系统权重10,869,248 bytes降为零，小型可见显存堆分配降为零。CPU概率及GPU FP32两条末端分别比较新旧策略，96图CLI509行、HTTP545块结果一致（排除时间字段）。四轮四并发共23,244请求成功：旧策略8进程8次分配重试、11条驱动告警，新策略8进程均零；GPU FP32吞吐93.63→94.58图/秒，CPU末端97.29→99.56图/秒，配对区间均包含零，不能声称显著提速。

固定60 OCR请求/秒目标负载下，36次ASR/TTS计量请求均成功，样本不足以推导延迟SLA。最终正式服务PID、重启次数、OCR环境及DLL散列未变，隔离实例全部停止。本次主要解决分配稳定性，仍需长期和边界验证后再考虑进入正式版本。

完整运行包、补丁源代码与原始结果备份到 `F:/Strata/Ubuntu-gpu-server-20261122/Linux显存分配修复20261123`，SHA校验见backup-result.json。
