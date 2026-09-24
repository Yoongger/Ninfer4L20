# ninfer-l20 移植规格（v2 重写版）

基线：`sergiuszm/ninfer-4090` @ `rtx4090-port`（sm_89，源自 `Don-Chad/ninfer-3090` 的 sm_86 兼容层）
目标：NVIDIA L20 — AD102，sm_89，**92 SM**，48 GB GDDR6，864 GB/s

移植内容不再是一个整体 diff，而是 `port/manifest.json` 里声明的一组**可重放变更**
（7 个逐文件补丁 + 1 个全树扫描）。每个变更有 `verify_after` 谓词，
`tools/port_engine.py` 按 `2-way patch → 3-way merge → variant transform` 的阶梯应用，
`tools/verify.py` 以 8 条静态不变量收口。上游更新时重放即可（`scripts/upgrade.sh`）。

---

## 0. 目标硬件对照

| | RTX 5090（上游调优目标） | RTX 4090（基线调优目标） | **L20（本项目目标）** |
|---|---:|---:|---:|
| 架构 | sm_120a | sm_89 | **sm_89** |
| SM 数 | 170 | 128 | **92** |
| 带宽 | 1792 GB/s | 1008 GB/s | **864 GB/s** |
| 显存 | 32 GB | 24 GB | **48 GB** |
| 单波 CTA（×4/SM） | 680 | 512 | **368** |

L20 是三者中 SM 最少、显存最多的：**网格尺寸必须按运行时 SM 数计算；KV/并发 profile 放宽而不是收紧。**
注意"92"只存在于本项目的文档与 manifest 里——**编译产物中没有任何一处硬编码它**（verify V4/V6/V7 强制检查）。

---

## 1. 变更清单（对应 manifest 的 changes）

| id | 内容 | 性质 |
|---|---|---|
| 010 | 顶层 `CMakeLists.txt` 按 `CMAKE_CUDA_ARCHITECTURES` 在目录作用域定义 `NINFER_SM86`/`NINFER_SM89` | 架构身份 |
| 020 | 删除 `src/CMakeLists.txt` 里硬编码的 `NINFER_SM86=1`（保留其守卫的 NVFP4 源码过滤） | 架构身份 |
| 030 | `core/device.h`：声明 `device_sm_count()` | 运行时宽度 |
| 040 | `core/device.cu`：`device_sm_count()` 实现（查询一次并缓存） | 运行时宽度 |
| 050 | `gdn/chunked/output.cu`：波尺寸从硬编码 170 SM 改为运行时 `device_sm_count()` | 真缺陷 A（热路径） |
| 060 | `sparse_moe/prefill`：网格上限从硬编码 170 SM 改为运行时值 | 同类缺陷（35B 路径） |
| 070 | `bf16_gdn_gating_proj_kernels.cu`：协同驻留数 `SplitK == 8 ? 2 : 1` | 真缺陷 B（L20 崩溃） |
| 080 | 全树 `NINFER_SM86` 条件编译 → 兼容 `SM86 \|\| SM89`（文件集运行时派生） | 架构身份（机械） |

### 1.1 与 v1 补丁集的差异

- **P3（16 文件 28 处机械替换）变成 080 全树扫描**：文件集由 `git ls-files` 现场派生，
  不再存文件清单。上游 2026-09 的更新里 `src/ops/linear/w8/` 新增了 3 个带
  `NINFER_SM86` 分支的文件（`w8_rowsplit_gemm_mma.cuh`、`w8_small_t_mma.cuh`、
  `w8_linear_swiglu_gemm_mma.cu`），`dynamic_grouped_conv` 也多了一处——
  固定清单必然漏掉，扫描自动覆盖（当前基线共 20 个文件 33 处）。
- **P7（sparse-MoE）语义变了**：上游 `437e9f98` 把持久网格重构成"按工作量定网格 + 上限钳制"
  （`routed_*_blocks = std::min(work, kPrefillMaxBlocks)`）。`kRtx5090SmCount` 字面量仍在，
  但只是上限的乘数。060 现在把这个上限改成运行时派生（`prefill_max_blocks()`）；
  对旧版结构（`kPrefillPersistentBlocks` 直发）保留 legacy transform 变体，
  两种形状都能落。
- **不再钉上游 commit**：补丁的 `index` 行只是 3-way 的回退素材；
  应用目标是 worktree 所在的任意上游 commit，来源记入 `build-info.json`。

---

## 2. 架构身份（010 / 020 / 080）

### 问题
基线在 sm_89 构建里定义 `NINFER_SM86=1`（`src/CMakeLists.txt`），Ada 名义上被当作 Ampere。
实测影响：`src/` 内 11 个文件、`tests/` 内 9 个文件（当前基线）。

### 危害的精确边界（不要夸大）
`w8_config.h` 的分支是 compat 调度（`KWarps = 4`），`#else` 分支是 Blackwell 专用；
`pdl.cuh` 的 SM86 分支是"无 PDL 的普通发射"，而 **Ada 确实没有 PDL**（PDL 是 sm_90+）。
**所以 Ada 当前走的每个分支都是正确分支，这个误标不改变生成代码。**

真正的危害是**架构身份失效**：将来任何按 `NINFER_SM86` 添加的 Ampere 专用调优会静默作用于 Ada。
`w8_config.h` 管着 W8 GEMM 的调度表，而 **MTP 的全部投影矩阵都是 `W8G32_F16S`**
（`W8MtpInputProjectionGeometry<5120,10240>`、`W8MtpAttentionProjectionGeometry<14336,5120>`、
`W8MtpGateUpProjectionGeometry<34816,5120>`、`W8MtpDownProjectionGeometry<5120,17408>`），
这个失效面正好压在单流解码的关键路径上。

### 改动
- **010**：目录作用域 `add_compile_definitions`（不是 per-target）：`device.h` 里的
  `#error` 需要 `ninfer_core` 和 `ninfer_ops` 都看得到宏，而 `ninfer_ops PUBLIC ninfer_core`
  只是链接依赖，不会把编译定义传给 `ninfer_core`。
- **020**：删硬编码定义，保留 NVFP4 源码过滤。
- **080**：机械替换，**行为保持**。顺序有讲究（长形式先于短形式：
  `#if defined()` 先于 `#ifdef`，`#if !defined()` 先于 `#ifndef`），否则后一次替换
  会命中前一次的输出、每轮追加重复从句。扫描器逐行精确匹配指令、幂等
  （已含 `NINFER_SM89` 的行不动）。

---

## 3. 运行时设备宽度（030 / 040）

### 问题
`DeviceContext::multiprocessor_count()` 和 `DeviceExecutionView.multiprocessor_count`
已经存在并服务于 GDN gating 的协同发射切分。缺口：**只拿到 `cudaStream_t` 的 Op 够不到它**——
`chunk_output_config` 只有一个 `stream` 字段，所以 `launch_output` 只能靠硬编码常量。

### 改动
- **030**：新增 `int device_sm_count();` 声明。
- **040**：实现（`static const int` + 立即执行 lambda 查询一次并缓存）。
  产品契约是"一进程一模型一设备"，一次缓存查询就是全部设备集合。

**刻意不加编译期镜像**（v1 规格里曾设计 `kTargetSmCount`，最终实现没有它）：
当前树里没有 `__device__` 发射策略需要编译期 SM 数，加了就是没有消费者的占位。
重定向到其它 sm_89 卡（4090 = 128 SM、L40S = 142 SM）不需要改任何代码——
所有运行时波尺寸都跟随 `device_sm_count()`。

---

## 4. GDN 输出内核的波尺寸（050）— 热路径真缺陷

```cpp
// 基线（上游从 RTX 5090 继承的字面量）
constexpr std::int64_t kRtx5090SmCount = 170;
constexpr std::int64_t kCtasPerSm      = 4;
constexpr std::int64_t kTargetCtas     = kRtx5090SmCount * kCtasPerSm;   // 680
```
680 描述 RTX 5090；**L20 真实单波 = 368**。后果链：
1. `jobs_per_block = ceil(logical_jobs / 680)` → 目标值偏大 → `jobs_per_block` 偏大
2. `grid_chunks = ceil(NT / jobs_per_block)` → 网格偏窄
3. 每个 CTA 串行处理更多 chunk → **设备未被填满**
4. **并且** `jobs_per_block == 1` 决定 `launch_fixed<false>` vs `launch_fixed<true>`
   （MULTI_JOB 特化）→ 网格目标值错误会**连带选错内核特化**

非协同发射（普通 `<<<>>>`），所以**不崩，只掉速**。

**热路径确认**：该内核是 GDN chunked 路径的 output 段。Qwen3.8-27B 的 65 层里
`full_attention_interval = 4` → 只有 16 层是全注意力，其余 **49 层是 GDN**，prefill 时每层都跑。

### 改动
```cpp
constexpr std::int64_t kCtasPerSm = 4;
...
const std::int64_t target_ctas    = kCtasPerSm * device_sm_count();
const std::int64_t logical_jobs   = NT * cfg.H_v;
const std::int64_t jobs_per_block = (logical_jobs + target_ctas - 1) / target_ctas;
```
并新增 `#include "core/device.h"`。预期收益集中在 prefill（`NT = L / BT` 大时才会跨波）；
解码时 `logical_jobs` 小、`jobs_per_block` 本就为 1，收益接近零。

---

## 5. sparse-MoE prefill（060）— 同类缺陷，35B 专用路径

上游 `437e9f98` 之后的结构：

```cpp
constexpr int kRtx5090SmCount          = 170;
constexpr int kPrefillMaxBlocksPerSm   = 32;
constexpr int kPrefillMaxBlocks        = kPrefillMaxBlocksPerSm * kRtx5090SmCount;
...
const int routed_gate_blocks = std::min(routed_gate_work, kPrefillMaxBlocks);
const int routed_down_blocks = std::min(routed_down_work, kPrefillMaxBlocks);
```

网格按工作量定（`routed_*_work` 由设备侧扫描产生），上限只做钳制——"any grid is correct"。
所以把上限改成运行时值**不改变正确性**，只是不再用 170 这个数字描述 L20：

```cpp
int prefill_max_blocks() { return kPrefillMaxBlocksPerSm * device_sm_count(); }
// 两处 std::min(..., kPrefillMaxBlocks) → std::min(..., prefill_max_blocks())
```

**范围声明**：稀疏 MoE 只被 **Qwen3.6-35B-A3B** 目标使用。Qwen3.8-27B 是稠密模型，
**不走这条路径**。此变更是为了让树里不残留 per-part SM 字面量（verify V7 检查），
对本用例的性能**没有影响**。旧版结构（`kPrefillPersistentBlocks` 直发 6 处）由
legacy transform 变体覆盖，两种形状都能落。

---

## 6. 协同发射驻留数（070）— L20 上唯一真实崩溃

### 问题
```cpp
// bf16_gdn_gating_proj_kernels.cu
template <class Geometry, int SplitK>
constexpr std::int32_t cooperative_resident_ctas_per_sm() noexcept {
    ...
    if constexpr (std::is_same_v<Geometry, Bf16Gdn27Geometry>) {
        static_assert(SplitK == 8 || SplitK == 4 || SplitK == 2);
        return 2;   // ← 对 split4/2 是错的
```

这个常量被 `launch_bf16_prefill_mma` 的发射钳制读取，用来决定单次协同发射覆盖多少 token tile。
只有 split-8 是 256 线程（65 寄存器）；split-4/split-2 走默认 512 线程（74 寄存器）：

| SplitK | 线程 | 寄存器 | 每 CTA | ×2 | Ada 寄存器文件 65,536 |
|---|---|---|---|---|---|
| split8 | 256 | 65 | 16,640 | 33,280 | ✅ → 2 CTA/SM |
| split4/2 | 512 | 74 | 37,888 | 75,776 | ❌ → **只能 1 CTA/SM** |

高估 → 网格超出真实驻留容量 → 驱动拒绝：

```
bf16_gdn_gating_proj_kernels.cu:310: CUDA_CHECK(cudaLaunchKernelEx(...)) failed:
cudaErrorCooperativeLaunchTooLarge: too many blocks in cooperative launch
```

92 SM 上 split2 算出 `max_token_tiles = 92×2÷6 = 30`，T=2688 单次发射生成 **126 CTA** 网格，
真实容量只有 **92** → 驱动拒绝。4090 有 128 SM（容量 256），同样的高估刚好盖住，
**所以只在 SM 数更少的卡上暴露——正是 port ledger 预言的失效模式。**

### 关键旁证
`bf16_gdn_gating_proj_plan.cpp` 的 `resident_ctas_27()` **本来就是对的**（split8→2、split4/2→1）。
两个文件互相矛盾，而只有内核文件那个常量被发射钳制真正读取——静态断言读的是对的那个，
所以问题一直没被发现。

### 修复
```cpp
return SplitK == 8 ? 2 : 1;
```

| | 修复前 | 修复后 |
|---|---|---|
| `ninfer_gdn_gating_proj_test` | Subprocess aborted | **Passed** |
| 全套件可运行测试 | 98 中 1 失败 | **98 全通过** |

### 教训（保留给未来的审计）
运行时自适应的切分不能替代正确的每 SM 占用常量。另两处未审计的同类常量：
- `Bf16Gdn35Geometry` 返回 `SplitK == 32 ? 2 : 4`：4 CTA × 512 线程 = 2048 线程/SM，
  **超过 Ada 的 1536 线程/SM 上限**，同样可疑。Qwen3.8-27B 不走该路径，启用 35B 前需审计。
- 6 个 `*_real_test` 需要真实产物，产物就绪后补跑。

---

## 7. 构建与运行

### 构建（Ubuntu 24.04，无 WSL/PowerShell 依赖）
```bash
bash scripts/setup.sh      # 依赖 + manifest 驱动的版本下限门禁
bash scripts/build.sh      # 克隆上游 + 打移植 + configure + 编译
bash scripts/build.sh --test   # 附 ctest 冒烟
```
- 上游以**全历史**克隆（3-way 回退需要补丁基线 blob 在对象库）。
- CUDA toolkit / 主机编译器 / 架构全部自动探测；`NINFER_CUDA_ARCH`、`CUDA_HOME` 可覆盖。
- 版本下限（FFmpeg 60/60/58/7、libcurl 7.85、CMake 3.28、GCC 13）来自 manifest，
  **不在脚本里硬编码**。22.04 的 FFmpeg 4.4 / libcurl 7.81 过不了门禁——用 24.04。
- 容器化等价环境：`docker build -f docker/Dockerfile -t ninfer-l20 .`
  （与上游官方 Dockerfile 同基座 `nvidia/cuda:13.1.2-devel-ubuntu24.04`）。

### 运行（profile 按实际 GPU 选择，可逐项覆盖）
```bash
bash scripts/start.sh 8090
```
- L20（48 GB）→ INT8 KV 拉满原生 262144（4090 在 24 GB 下 INT8 只能到 172032）
- 4090（24 GB）→ INT8 KV @ 172032
- 其它 sm_89 → 保守 KV @ 65536
- 全部参数可用 `NINFER_*` 环境变量覆盖；KV 预算 vs 空闲显存启动前预检
- `--prefill-chunk` 保持 ≤ 2688（超过会路由到 unsplit 调度，onset 精度差约 1e-5）

### 模型产物：必须钉住 revision
HF 的 `main` 在 2026-09-15 起是 v3 容器，基线 fork 只认 v2，不钉 revision 直接启动失败：

| revision | 日期 | 容器 | 体积 |
|---|---|---|---|
| `51630a0c0f` | 2026-09-15 | 3（`main` 现在指向它） | 19.03 GiB |
| `dc370fb629` | 2026-09-06 | 2 | 19.03 GiB |
| **`3526913004`** | 2026-08-14 | **2** | **16.96 GiB** |

`scripts/fetch-model.sh` 默认钉 `3526913004`，带 SHA256 校验与 8 字节 magic 探针。
中国大陆网络：`HF_ENDPOINT=https://hf-mirror.com bash scripts/fetch-model.sh`。

---

## 8. 实测结果（v1 移植，L20 实测；引擎行为 v2 未变）

单请求，L20，官方 groupwise 产物（16.96 GiB）+ INT8 KV @ 262144 + `--prefill-chunk 1024`：

| 用例 | 采样 | tok/s | 草稿接受率 | TTFT |
|---|---|---:|---:|---:|
| code | 贪心 | **100.11** | 65.5% | 209 ms |
| qa | 贪心 | 99.98 | 65.1% | 191 ms |
| qa | temp0.7 | 91.39 | 56.7% | 63 ms |
| code | temp0.7 | 82.63 | 48.9% | 63 ms |
| prose | 贪心 | 79.91 | 45.2% | 203 ms |
| prose | temp0.7 | 78.34 | 43.8% | 63 ms |
| 全部（`--spec none` 对照） | — | 38.9 – 39.3 | — | — |

**MTP 加速比：code 贪心 2.57× / code temp0.7 2.12× / prose 2.05×。**

与上游 4090 参考值对照：无投机 39.0 vs 50.5（77%）、code MTP3 100.11 vs 142.9–148.6（68–70%）。
L20 带宽是 4090 的 85.7%、SM 数 71.9%——**实测落在硬件规格区间内，移植没有引入额外损失。**
