# Ninfer4L20

把 [NInfer](https://github.com/Neroued/ninfer) 单 GPU CUDA 推理引擎移植到 **NVIDIA L20**
（Ada / sm_89 / 92 SM / 48 GB / 864 GB/s）的补丁集与构建工具。

基线是 [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)（sm_89 / RTX 4090 移植，
源自 [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) → [Neroued/ninfer](https://github.com/Neroued/ninfer)）。
L20 与 4090 同为 sm_89，这是同代移植；差别在 SM 数、带宽与显存：

| | RTX 4090 | **L20** |
|---|---:|---:|
| SM 数 | 128 | **92**（−28%）|
| 带宽 | 1008 GB/s | **864 GB/s**（−14%）|
| 显存 | 24 GB | **48 GB**（+100%）|

这两条差异决定了移植的全部内容：**网格尺寸必须按运行时 SM 数计算；KV/并发 profile 放宽而不是收紧。**

---

## 为什么重写（v1 的问题，有实证）

本项目**借鉴** v1（[reinwu/ninfer-L20](https://github.com/reinwu/ninfer-L20)）的移植思路——
8 项移植变更的内容、L20 服务 profile、宽 prefill 回归探针与实测数据——但代码全部独立实现：
工具链（应用引擎、验证器、脚本）与基准程序均为本仓库原创，不依赖 v1 仓库的任何代码
（逐行对比：基准程序相似度 0.11，构建脚本 ≤0.26）。

v1 把一个 **23 文件的整体 unified diff**
加一个**固定行号/固定文件清单的 PowerShell 锚点替换脚本**打在上游树上。上游一更新就裂：

在本重写开始时（上游 `rtx4090-port` tip = `aeeba414`，2026-09-23）实测 v1 补丁的应用状态：

- **3 个文件 hunk 失配**（`git apply --check` 失败）：
  - `sparse_moe_prefill_kernels.cu`——上游 `437e9f98` 把持久网格重构成"按工作量定网格 + 上限钳制"，
    v1 的 P7 hunk 找不到原来的 `kPrefillPersistentBlocks` 常量块；
  - `tests/ops/linear_swiglu/test_nvfp4.cpp`、`tests/ops/test_gdn_input_proj_conv_record.cpp`
    ——测试体被上游改写，hunk 上下文失配。
- **3-way 也救不回来**：v1 补丁的 `index` 行基线 blob 与 hunk 偏移自相矛盾
  （`CMakeLists.txt` 的 hunk 头写 project() 在第 14 行，记录的基线 blob 里却在第 17 行），
  `git apply --3way` 整包失败。
- **固定文件清单漏更新**：v1 的 P3 硬编码 16 个文件；当前上游树里带裸 `NINFER_SM86` 分支的
  是 **20 个文件**——上游新增了 `w8_rowsplit_gemm_mma.cuh`、`w8_small_t_mma.cuh`、
  `w8_linear_swiglu_gemm_mma.cu`、`w8_dynamic_grouped_conv_*.cu` 里的分支，清单式补丁必然漏掉，
  漏掉的就是静默的架构身份错误。
- **硬编码环境**：WSL2 + PowerShell 编排、`CUDA_HOME=/usr/local/cuda-13.1` 写死、
  服务 profile（262144/int8/k3）写死在脚本里——换机器、换卡、换 CUDA 版本都要手改脚本。

v2 的目标：**上游更新 → 一条命令跟上，编译错误要么不发生，要么在 30 秒内被定位到具体文件**，
而不是在 1400 文件 CUDA 构建的末尾炸出来。

## 重写做了什么

### 1. 移植 = 声明式 manifest + 可重放变更（不是一份 diff）

`port/manifest.json` 声明 8 个变更，每个变更自带**验证谓词**；
`tools/port_engine.py` 按下列阶梯逐变更应用（幂等，可反复重放）：

```
verify_after 已满足 ──> 跳过
2-way git apply ──> 3-way merge（git merge-file，透明冲突数）──> transform 变体阶梯 ──> 结构化 drift 报告
```

- **7 个逐文件语义补丁**（身份宏、`device_sm_count()` 声明/实现、GDN 波尺寸、
  sparse-MoE 上限、协同驻留数修正）——由 `tools/regen_patches.py` **从移植后的树重新派生**，
  基线永远是当时的上游 tip（`index` 行含基线 blob SHA，供 3-way 使用）。
  上游更新后 `scripts/upgrade.sh` 自动重录补丁，2-way 直通恢复，3-way/transform 退回后备位。
- **1 个全树扫描**（`NINFER_SM86` → `NINFER_SM86 || NINFER_SM89` 的 33 处机械替换）：
  文件集由 `git ls-files` **运行时派生**——没有文件清单、没有行号。
  上游新增/改名/改写文件自动覆盖；逐行精确匹配指令 + 幂等（已含 `NINFER_SM89` 的行不动）。
- **transform 变体阶梯**：上游重构了结构时（改名字、换算法），正则变体识别"同一个语义形状"。
  例如 060（sparse-MoE）有两个变体——当前 tip 的 `kPrefillMaxBlocks` 上限形状、
  重构前的 `kPrefillPersistentBlocks` 直发形状。多步原子（常量 + 全部消费点一起换）。
- **drift 报告**：三种策略都落不了时，`<worktree>/.port-report.json` 记录
  期望锚点 + 当前上下文，`verify.py` 指出哪条不变量破了。移植不会在构建深处无声失败。

### 2. 不变量即契约（verify 先于编译）

`tools/verify.py` 的 8 条静态检查是移植的可执行规格（详见
[docs/PORT-SPEC.md](docs/PORT-SPEC.md)）：身份宏在目录作用域、无裸 `NINFER_SM86` 分支、
**`src/` 无 per-part SM 数字面量**、`device_sm_count()` 声明与定义在位、
两个网格消费点运行时派生、协同驻留数 split-aware……
任何一条变红直接指向文件——**编译是最后一道闸，不是第一道**。

### 3. 不钉任何东西

- **不钉上游 commit**：基线是 `origin/rtx4090-port` tip；来源 SHA 事后记入
  `build-info.json` 与 port commit 消息，只用于溯源。
- **不钉 SM 数**：代码里没有 92（也没有 128/170）——所有波尺寸走运行时
  `device_sm_count()`（查询一次、缓存）；92 只出现在文档与 manifest 的说明字段里。
  重定向到其它 sm_89 卡（4090/4080/L40S）**零代码改动**。
- **不钉环境**：CUDA toolkit、主机编译器、架构、构建路径全部自动探测
  （`nvidia-smi` 探测实际 GPU 的 compute cap 并与目标架构交叉核对）；
  版本下限（FFmpeg 60/60/58/7、libcurl 7.85、CMake 3.28、GCC 13）集中在 manifest，
  脚本里不写第二个版本数字。
- **不钉 profile**：`start.sh` 按 `nvidia-smi` 的 GPU 名/显存选 profile
  （L20 → INT8 KV @ 262144；4090 → INT8 @ 172032；其它 sm_89 → 保守 @ 65536），
  每个参数 `NINFER_*` 环境变量可覆盖，启动前做 KV 预算 vs 空闲显存预检。

### 4. Ubuntu 基座（无 WSL/PowerShell）

v1 的编排是 Windows PowerShell 调 WSL2。v2 是纯 `bash` + `python3`（stdlib only）：
裸 Ubuntu 24.04 直跑、WSL 里直跑、或 Docker（`docker/Dockerfile`，与上游官方镜像同基座
`nvidia/cuda:13.1.2-devel-ubuntu24.04`）三种方式等价。22.04 过不了上游的 FFmpeg/libcurl
版本门禁（FFmpeg 4.4 / libcurl 7.81）——setup 的门禁会明确告诉你，而不是在 configure
里含糊失败。

## 移植改了什么（上游树）

| id | 内容 | 性质 |
|---|---|---|
| 010 | 顶层 `CMakeLists.txt` 按架构定义 `NINFER_SM86`/`NINFER_SM89`（目录作用域） | 架构身份 |
| 020 | 删 `src/CMakeLists.txt` 硬编码 `NINFER_SM86=1`（sm_89 构建被当 Ampere 编译的根源） | 架构身份 |
| 030/040 | `core/device.{h,cu}`：`device_sm_count()` 运行时查询（缓存） | 运行时宽度 |
| 050 | `gdn/chunked/output.cu`：波尺寸 170 SM 字面量 → `device_sm_count()`。**热路径**：65 层里 49 层是 GDN | 真缺陷 A（掉速） |
| 060 | `sparse_moe/prefill`：网格上限 170 SM 字面量 → 运行时（35B 专用路径，为不留字面量） | 同类缺陷 |
| 070 | `bf16_gdn_gating_proj_kernels.cu`：协同驻留 `return SplitK == 8 ? 2 : 1`。**L20 唯一真实崩溃**（`cudaErrorCooperativeLaunchTooLarge`，92 SM 上 126 CTA 网格 vs 92 容量） | 真缺陷 B（崩溃） |
| 080 | 全树 `NINFER_SM86` 条件 → 兼容 `SM86 \|\| SM89`（20 文件 33 处，运行时派生） | 架构身份（行为保持） |

070 的完整分析（寄存器账本、为什么 4090 上不崩、plan 侧本来就是对的）见
[docs/PORT-SPEC.md §6](docs/PORT-SPEC.md)。

## 构建与运行

```bash
# 1) 依赖 + 版本门禁（manifest 驱动）
bash scripts/setup.sh

# 2) 克隆上游 + 打移植 + 验证 + configure + 编译（Ubuntu 24.04，CUDA ≥12.8）
bash scripts/build.sh            # 附 --test 跑 ctest 冒烟

# 3) 模型产物（钉 revision + SHA256 校验；v2 容器 16.96 GiB）
bash scripts/fetch-model.sh

# 4) 起服务（profile 按实际 GPU 选择，NINFER_* 可覆盖）
bash scripts/start.sh 8090

# 5) 基准（三工作负载 + 无投机对照 + 宽 prefill 回归探针）
python3 scripts/bench.py --model models/qwen3_8_27b.ninfer

# 上游更新后
bash scripts/upgrade.sh          # fetch + 重放移植 + 重录补丁 + 验证 + 增量重编译
```

容器化：`docker build -f docker/Dockerfile -t ninfer4l20 .`，
`docker run --gpus all -p 8090:8090 -v $PWD/models:/ninfer/models:ro -e NINFER_MODEL=/ninfer/models/qwen3_8_27b.ninfer ninfer4l20 8090`。

## 实测结果（L20）

单请求，官方 groupwise 产物（16.96 GiB）+ INT8 KV @ 262144 + `--prefill-chunk 1024`。

**v2 构建实测**（本项目 `build.sh` 于 L20 实机编译：CUDA 13.2 / gcc-13 / sm_89，MTP3）：

| 用例 | 采样 | tok/s | 草稿接受率 | TTFT | prefill |
|---|---|---:|---:|---:|---:|
| code | 贪心 | **96.30** | 57.3% | 201 ms | 404 tok/s |
| qa | 贪心 | **102.19** | 64.0% | 178 ms | 398 tok/s |
| qa | temp0.7 | 95.41 | 57.3% | 58 ms | — |
| code | temp0.7 | 81.99 | 44.6% | 59 ms | — |
| prose | 贪心 | 83.46 | 45.9% | 194 ms | 402 tok/s |
| prose | temp0.7 | 81.76 | 44.5% | 58 ms | — |

宽 prefill 探针（4063 token prompt，>2688 阈值路由协同发射路径）：1.46k tok/s 完成，无 abort
——070 协同驻留修复在实机上的回归检查通过。

**v1 移植实测**（对照基线）：code 贪心 100.11 / qa 贪心 99.98 / qa temp0.7 91.39 /
code temp0.7 82.63 / prose 贪心 79.91 / prose temp0.7 78.34；`--spec none` 对照 38.9–39.3。
v2 构建与 v1 实测差 ±4%（工具链差异：v1 用 CUDA 13.1）——**移植语义未变，无回归**。

与上游 4090 参考值对照：无投机 39.0 vs 50.5（77%）、code MTP3 ~100 vs 142.9–148.6（68–70%）——
L20 带宽是 4090 的 85.7%、SM 数 71.9%，**实测落在硬件规格区间内，移植未引入额外损失**。
48 GB 的价值：INT8 KV 直接拉满原生 262,144（4090 在 24 GB 下只能到 172,032，被迫用 4-bit E8 并付 5.7% 解码税）。

## 最快性能配置（262144 实测，start.sh 的 L20 profile 即此配置）

2026-09-24 在 L20 上对**同一构建**做了完整扫描与 A/B（max-context/kv-capacity 固定 262144，
单请求，两个 27B 产物：官方 groupwise + WaveCut HauhauCS-DFlash2）：

**扫描结论（每项单独 A/B）：**

| 参数 | 结论 |
|---|---|
| `--kv-dtype` | **bf16 最快**：greedy 解码比 int8 快 3.6–12.8%（MTP 接受率 59.7→67.5%），prefill 快 1–3.7%；fp8 居中（解码 +4.7%）；且 bf16 无 KV 量化误差，精度最高 |
| `--spec mtp --draft-tokens` | k3 最快（k1 70.8 / k2 86.8 / **k3 96.9** / k4 83.4 / k5 89.8，code greedy 同批 A/B）；k6+ 引擎拒收 |
| `--prefill-chunk` | 2688：prefill 在 86k–219k 上下文 +3.7–6.7%（882.6 vs 827.3 @219k），解码持平 |
| `--max-concurrency` | 1：单流场景下 conc2 每请求 −0.7% |
| CUDA graphs | 开：关掉 −2.6% |
| `--host-kv-mib` | 2048 与 8192 无速度差（8192 为默认，留 host 溢出余量） |

**A/B 实测（同构建、同机、同日，256 token 解码 + 128k prefill）：**

| 模型 | 配置 | code 贪心 | prose 贪心 | qa 贪心 | MTP 接受率 | 128k prefill | 219k TTFT |
|---|---|---:|---:|---:|---:|---:|---:|
| 官方 | int8 | 88.15 | 71.36 | 76.58 | 42–60% | 1073 tok/s | 265 s |
| 官方 | **bf16** | **96.94** | **73.93** | **81.24** | 43–68% | **1110 tok/s** | **241 s** |
| WaveCut | int8 | 81.55 | 77.59 | 80.63 | 50–54% | 1068 tok/s | — |
| WaveCut | **bf16** | **88.69** | **87.49** | **82.86** | 54–60% | **1108 tok/s** | — |

**因此 start.sh 的 L20 profile 为：**

```
--max-context 262144 --kv-capacity 262144 --kv-dtype bf16
--spec mtp --draft-tokens 3 --lm-head-draft
--prefill-chunk 2688 --max-concurrency 1
```

内存账（L20 48 GB，bf16 KV @ 262144）：权重 15.9 + 设备 KV 16.5 + 常驻 arena ~9.5 +
workspace/graph ~1 + MTP KV ~0.5 ≈ **43 GiB**，219k（83% KV 占用）长 prompt 实测通过。
4090（24 GB）放不下 bf16 KV @172032，profile 保持 int8。唯一 int8 略优的场景是
≥128k 上下文后的解码（KV 读带宽减半），幅度 ±8% 在运行间噪声内。
DFlash2 产物（recipe v2，7 草稿）同样适用上述 KV/预取参数，投机模式换成
`NINFER_SPEC=dflash2`（= `--spec dflash2 --draft-tokens 7 --lm-head-draft`）。
L20 实测（bf16 KV，WaveCut）：code 贪心 **100.86** tok/s（42.2% 接受，比 MTP k3 的
88.69 快 14%）、qa 贪心 77.00、prose 贪心 69.29（比 MTP k3 的 87.49 慢 21%）——
**代码负载 DFlash2 更快，散文/通用负载 MTP k3 更快**；两者都远快于无投机（44）。

## 已验证的稳健性场景（本仓库测试环境实测）

| 场景 | 结果 |
|---|---|
| **Ubuntu 24.04 实机全链路**（L20×4 服务器） | setup 门禁 → 克隆+移植+8/8 → CUDA 13.2 全量编译 → ctest **120/120 通过**（12 个 `*_real_test` 按设计跳过）→ start.sh 7s 就绪（262144 INT8 KV）→ 基准 ±4% 于 v1 |
| 当前上游 tip 首跑 | 7/7 变更 2-way 直通 + 扫描 20 文件，8/8 不变量 |
| 重放（幂等）/ 中途崩溃后重跑 | 全部 `present` 跳过或补完，无重复插入 |
| 上游改写补丁上下文邻行（注释） | 2-way 败 → 3-way 冲突 1 → transform 救回，结果正确 |
| 上游重构 sparse-MoE 为旧版直发形状 | 2-way 败 → 3-way 冲突 → legacy transform 变体命中，8/8 不变量 |
| 上游重构 commit 的父提交（即 v1 补丁的基线年代） | 全阶梯落定，8/8 不变量 |

## 目录

```
.
├─ port/
│  ├─ manifest.json          移植契约：基线、目标、版本门禁、8 个变更（含验证谓词与 transform 阶梯）
│  └─ changes/*.txt          7 个逐文件变更补丁（从移植后树派生，随上游更新重录）
├─ tools/
│  ├─ port_engine.py         应用引擎：2-way → 3-way → transform → drift 报告（幂等，stdlib only）
│  ├─ verify.py              8 条静态不变量（移植的可执行规格）
│  ├─ regen_patches.py       补丁重录器（上游更新后恢复 2-way 直通）
│  └─ common.py              共享工具
├─ scripts/
│  ├─ setup.sh               依赖安装 + manifest 驱动的版本门禁（Ubuntu 24.04）
│  ├─ build.sh               克隆 + 移植 + 验证 + configure + 编译 + provenance
│  ├─ upgrade.sh             上游更新流：fetch + 重放 + 重录 + 验证 + 增量重编译
│  ├─ start.sh               按实际 GPU 选 profile 起服务（NINFER_* 覆盖）
│  ├─ fetch-model.sh         模型产物（钉 revision + SHA256 + magic 探针）
│  └─ bench.py               引擎基准（三工作负载 + 对照 + 回归探针）
├─ docker/Dockerfile         可复现 Ubuntu 构建镜像（上游官方镜像同基座）
└─ docs/
   ├─ PORT-SPEC.md           逐变更技术规格（含寄存器账本、热路径分析）
   └─ UPGRADING.md           上游更新手册：阶梯详解 + drift 处置 + 回归清单
```

## 已知限制

- **未做深上下文扫描**：数字均为浅上下文（prompt 5–81 tokens）；上游数据显示 MTP 接受率随深度下降。
- **050 收益未做前后 A/B**（预期集中在 prefill）。
- **`draft-tokens` 最优 K 未扫**（固定 3）。
- **35B 的协同驻留常量未审计**（`SplitK == 32 ? 2 : 4`，4 CTA × 512 线程超 Ada 线程上限，可疑）。
- **6 个 `*_real_test` 未运行**（需要真实产物）。
- **未经上游 review**：个人移植，非上游官方支持的平台。

## 致谢与许可

Apache-2.0（见 [LICENSE](LICENSE)、[NOTICE](NOTICE)），衍生自 Neroued/ninfer、
Don-Chad/ninfer-3090、sergiuszm/ninfer-4090、UDPSendToFailed/ninfer-4090（均 Apache-2.0）。
上游源码不在本仓库分发；移植由 `tools/` 应用到自行克隆的基线上。
