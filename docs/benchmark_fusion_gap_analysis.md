# Benchmark 融合算子差距分析方案

## 1. 背景与目标

Q4 重点任务是基于 Benchmark 评估 NPU 与 GPU 在真实模型网络中的融合算子差距，并回答两个问题：

1. **融合范围是否一致**：NPU 和 GPU 是否将相同的算子组合成了相同的融合子图。
2. **相同融合下谁更快**：当两边形成相同融合时，NPU 与 GPU 的融合 Kernel 性能差距是多少。

最终需要把理论上的编译器、Kernel 和调度差距，映射到真实模型的 E2E 性能差距中，形成可定位、可复现、可跟踪的分析结论。

---

## 2. 当前看板能回答什么

当前看板主要包含以下数据：

| 数据 | 当前含义 |
|---|---|
| NPU E2E | 模型端到端执行耗时 |
| NPU OP | Profiler `step_trace_time.csv` 中 `Computing` 的平均耗时 |
| OP 加速比 | `eager_OP_avg_time / compile_OP_avg_time` |
| GPU E2E | GPU 模型端到端耗时，具体取决于已有 GPU baseline |

当前数据可以说明“编译后整体计算耗时是否下降”，但还不能直接说明：

- 哪些融合在 GPU 上存在、NPU 上没有；
- 一个融合组具体包含哪些算子；
- NPU 和 GPU 的相同融合组是否使用了相同的输入 shape、dtype 和布局；
- 相同融合组的 Kernel 性能差距来自计算、访存还是调度。

当前处理逻辑主要读取 `step_trace_time.csv`，虽然流水保留了 `op_statistic.csv`、`kernel_details.csv` 和 `trace_view.json`，但还没有建立融合组、图节点和运行 Kernel 之间的关联。

因此，当前的 OP 加速比是**模型级聚合指标**，不是融合算子级指标。

---

## 3. GAP 的统一分类

### 3.1 融合范围 GAP

融合范围 GAP 关注“融合了多少、融合了哪些算子”。

典型情况：

```text
GPU:  convolution + bias + relu      -> 一个融合 Kernel
NPU:  convolution                    -> 一个 Kernel
      add                            -> 一个 Kernel
      relu                           -> 一个 Kernel
```

此时 NPU 的主要问题不是单个 Kernel 一定更慢，而是融合没有形成，导致：

- Kernel 数量增加；
- 中间 Tensor 写回内存；
- Kernel launch 次数增加；
- layout 转换和同步增加；
- E2E 耗时被放大。

### 3.2 相同融合下的性能 GAP

相同融合下，只比较算子序列、shape、dtype 和布局都匹配的融合组。

例如：

```text
GPU: convolution + relu -> 0.80 ms
NPU: convolution + relu -> 1.35 ms
```

此时融合范围相同，但 NPU 仍然慢，重点应转向：

- Triton Kernel 实现；
- tile 和 block 配置；
- 访存合并和带宽利用率；
- layout 与数据格式；
- CANN/ACLNN Kernel 实现；
- 同步和 launch 开销。

### 3.3 非融合因素 GAP

如果 OP 耗时差距不大，但 E2E 差距明显，需要单独排查：

- graph break；
- eager fallback；
- Kernel launch 和调度；
- 内存分配和释放；
- layout 转换；
- graph capture/replay；
- 编译、warmup 和同步开销。

这类差距不能直接归因于融合算子。

---

## 4. 需要采集的数据

每条记录的最小粒度应为：

```text
模型 × train/inference × backend × device × dtype × dynamic 配置
```

### 4.1 图和融合结构

需要记录：

| 字段 | 说明 |
|---|---|
| `graph_id` | 编译图编号 |
| `fusion_id` | 融合组编号 |
| `ops` | 融合组内的算子序列 |
| `op_count` | 融合组内算子数量 |
| `shape_signature` | 输入输出 shape 特征 |
| `dtype` | 输入、输出和中间 dtype |
| `layout` | 数据布局或格式 |
| `graph_break` | 是否存在图断点 |
| `fallback` | 是否回退到 eager 或单算子 |

### 4.2 Kernel 运行数据

需要记录：

| 字段 | 说明 |
|---|---|
| `kernel_name` | 运行时 Kernel 名称 |
| `kernel_hash` | 生成代码或 Kernel 的稳定标识 |
| `call_count` | 调用次数 |
| `kernel_time_ms` | 单次或总运行耗时 |
| `stream` | 所在设备 Stream |
| `bytes_read/write` | 访存量，能获取时记录 |
| `flops` | 计算量，能估算时记录 |

### 4.3 运行环境

所有 NPU/GPU 数据都要带上：

```text
model
suite
mode
backend
device
dtype
batch_size
dynamic
aclgraph/cudagraph
torch commit
torch_npu commit
triton commit
CANN version
warmup
active
```

没有这些信息时，两个结果即使模型名相同，也不能直接做严格对比。

---

## 5. 融合分析结果格式

建议生成轻量的 `fusion_manifest.json`，不要长期保存完整 Inductor debug 目录。

示例：

```json
{
  "model": "resnet50",
  "mode": "inference",
  "device": "npu",
  "backend": "triton_experimental",
  "graph_id": "graph_3",
  "fusion_id": "fusion_12",
  "ops": [
    "aten.convolution",
    "aten.add",
    "aten.relu"
  ],
  "op_count": 3,
  "shape_signature": "NCHW:1x64x56x56",
  "dtype": "float32",
  "kernel_name": "triton_fused_conv_add_relu",
  "kernel_hash": "...",
  "call_count": 50,
  "kernel_time_ms": 1.23,
  "fallback": false
}
```

建议配套生成：

```text
fusion_manifest.json       融合结构和算子序列
fusion_runtime.csv         融合 Kernel 运行耗时
fusion_gap_summary.json    面向看板和报告的汇总结果
```

---

## 6. 需要统计的指标

### 6.1 融合范围指标

```text
融合覆盖率 = 被融合覆盖的可融合算子数 / 可融合算子总数
```

重点展示：

- GPU 融合组数量；
- NPU 融合组数量；
- GPU/NPU 融合算子覆盖率；
- GPU 独有融合组；
- NPU 独有融合组；
- graph break 数量；
- fallback 算子数量；
- 融合 Kernel 数量变化。

如果需要体现真实网络影响，可以增加运行时间加权指标：

```text
时间加权融合覆盖率 = 融合 Kernel 计算耗时 / 总算子计算耗时
```

### 6.2 相同融合性能指标

只对融合签名匹配的组进行比较：

```text
GPU/NPU = GPU 融合 Kernel 耗时 / NPU 融合 Kernel 耗时
```

其中：

- 大于 1：NPU 更快；
- 小于 1：GPU 更快；
- 接近 1：相同融合下性能接近。

建议同时记录 P50、P90 和平均值，避免单次 profiler 采样造成误判。

### 6.3 E2E 分解

模型级报告建议拆成：

```text
E2E = 算子计算 + Kernel launch/调度 + 内存/布局转换 + fallback + 其他框架开销
```

这样可以避免把所有 E2E 差距都归因于融合。

---

## 7. 按 GAP 类型定位问题

| 现象 | 优先判断 | 主要排查方向 |
|---|---|---|
| GPU 已融合，NPU 未融合 | 融合范围 GAP | pattern matcher、lowering、Triton-Ascend 融合规则、graph break |
| 两边融合相同，NPU 更慢 | 相同融合性能 GAP | Kernel、tiling、访存、数据布局、CANN 实现 |
| NPU 融合更多但 E2E 更慢 | 过度融合或调度 GAP | launch、同步、内存、layout、融合粒度 |
| OP 接近但 E2E 差距大 | 框架运行时 GAP | fallback、调度、内存管理、graph capture |
| NPU/GPU 图结构不同 | 图结构 GAP | 动态 shape、输入处理、dtype、后端特化配置 |
| 只有个别模型异常 | 模型特化或数据问题 | YAML 配置、输入 shape、权重、精度和 fallback |

---

## 8. NPU 与 GPU 对齐要求

GPU 对比必须使用尽可能一致的条件：

1. 相同的 PyTorch Benchmark commit；
2. 相同模型和输入数据；
3. 相同 train/inference 模式；
4. 相同 batch size、dtype 和 AMP 配置；
5. 相同 warmup、active 和 iteration 配置；
6. 明确是否开启 CUDA Graph/ACLGraph；
7. 排除模型下载、编译和首次初始化时间；
8. 至少重复 3 次，报告中位数和波动范围。

当前只有 GPU E2E 数据时，不能推导 GPU OP 或 GPU 融合数据。GPU 需要重新执行 profile，建议使用 PyTorch Profiler/Kineto 采集 CUDA Kernel 和算子事件，再与 NPU profiler 结果统一成相同的记录格式。

由于 A100 与 NPU 的硬件架构不同，GPU/NPU 结果适合用于：

- 定位融合差距；
- 定位相同融合下的性能差距；
- 指导后端优化优先级。

不应把单个硬件上的绝对耗时直接解释为另一种硬件的理论上限。

---

## 9. 建议的实施节奏

### 第一阶段：建立可比基线

不修改现有 daily/weekly 流水，只增加专项开关，例如：

```bash
BENCH_FUSION_ANALYSIS=1
```

先选择 10 个左右精度通过、无网络和 OOM 问题的模型：

- CV：`resnet50`、`densenet121`、`mobilenet_v2`、`repvgg_a2`；
- HuggingFace：`BertForMaskedLM`、`MobileBertForMaskedLM`、`T5ForConditionalGeneration`；
- TIMM：`ghostnet_100`、`tf_efficientnet_b0`、`swin_base_patch4_window7_224`。

NPU 和 GPU 各跑 eager/compile，先确认融合组可以稳定匹配。

### 第二阶段：形成模型级 GAP 表

每个模型输出：

```text
E2E、OP、融合覆盖率、相同融合数量、GPU/NPU 相同融合性能、fallback 数量
```

并按“融合范围 GAP / 相同融合性能 GAP / 运行时 GAP”分类。

### 第三阶段：扩展到 weekly 全量

只对通过精度且 profile 完整的模型进入融合分析。网络、缺数据、OOM、timeout 等问题先单独归类，不和后端融合问题混在一起。

### 第四阶段：接入看板

在模型详情中增加两个区域：

1. **融合范围对比**：NPU/GPU 融合组和覆盖率；
2. **相同融合性能对比**：匹配融合组的耗时及 GPU/NPU 比值。

顶部只展示汇总指标，详细融合组放在模型详情或独立分析页，避免主表过于拥挤。

---

## 10. 汇报时可以使用的结论模板

> 本次分析将模型性能差距拆分为融合范围差距和相同融合下的 Kernel 性能差距。首先在相同模型、输入、dtype、运行模式和版本条件下，对 NPU 与 GPU 的编译图进行对齐，统计两边的融合组、融合算子覆盖率和 fallback 情况；随后只对算子序列、shape、dtype 和布局一致的融合组进行耗时比较。这样可以区分“没有融合导致的差距”和“融合已经一致但 Kernel 性能不足导致的差距”，并进一步判断问题属于编译器融合规则、Triton/CANN Kernel、内存布局还是运行时调度。

最终每个模型都会形成一张 GAP 表：

```text
模型 | 模式 | GPU 融合覆盖率 | NPU 融合覆盖率 | 相同融合数 |
GPU 相同融合耗时 | NPU 相同融合耗时 | GPU/NPU | 主要 GAP 类型
```

这张表可以直接作为后续 Triton-Experimental 优化、算子融合补齐和性能回归跟踪的依据。
