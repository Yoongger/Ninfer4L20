# 上游更新怎么办（upgrade 手册）

本项目的存在理由：`sergiuszm/ninfer-4090` 会持续更新，而移植必须跟着它走、
且**不要因为硬编码在更新中编译失败**。v1 的做法（整体 unified diff + 固定行号的
PowerShell 锚点替换）在上游 `437e9f98`（sparse-MoE 重构）之后已经裂开：
3 个文件的 hunk 失配、固定 16 文件清单漏掉 3 个新增文件。v2 把"跟着上游走"
做成一条命令 + 一套可诊断的阶梯。

---

## 1. 日常：上游推了新 commit

```bash
bash scripts/upgrade.sh
```

它做的事（全部幂等）：

1. `git fetch` 上游 tip；
2. 报告上游在 `src/` 里改了什么（diffstat + commit 列表）；
3. 把 `l20-port` 分支重置到新 tip 之上；
4. **重放移植**（`tools/port_engine.py`）——不是重放一个 diff，
   而是逐变更跑应用阶梯（见 §2）；
5. 跑 8 条静态不变量（`tools/verify.py`）；
6. `tools/regen_patches.py` 把 7 个变更补丁**重新记录到新基线上**
   （下次运行恢复 2-way 直通，3-way/transform 退回后备位），
   并在本仓库提交重录的补丁；
7. 增量重编译；
8. 输出移植面 diff：旧 port commit vs 新 port commit——
   告诉你这次上游更新有没有改变移植本身。

**没有硬编码 commit**：基线永远是 `origin/rtx4090-port` 的 tip；
来源 SHA 记录在 `build-info.json`（build 时写）和 port commit 消息里，只用于溯源，
从不作为输入消费。

## 2. 应用阶梯：每个变更怎么落

```
verify_after 谓词已满足？ ──是──> 跳过（幂等）
        │否
2-way `git apply`（补丁上下文还在） ──成──> 完成
        │败
3-way merge（`git merge-file`：补丁基线 blob × 当前文件 × 补丁结果） ──成──> 完成
        │冲突
transform 阶梯（manifest 里声明的"已知源形状"正则变体，按序尝试） ──成──> 完成
        │无匹配
drift 报告（文件、期望锚点、当前上下文）+ verify 定位是哪条不变量破了
```

- **2-way**：补丁是基线 → 移植后 的逐文件 diff，`index` 行带基线 blob SHA。
  上游只改别的行 → 偏移后照样命中（2026-09-23 的 tip 上 7/7 全部 2-way 命中）。
- **3-way**：上游改了补丁**上下文邻行**（如改注释）时，`git merge-file` 做
  透明三方合并；冲突数可读（`git apply --3way` 的黑箱行为被刻意避开）。
  注意：基线 blob 必须在全历史克隆的对象库里——所以 `build.sh` 不用 `--depth 1`。
- **transform**：上游**重构了结构**（改名字、换算法）时，正则变体识别"同一个语义形状"。
  060 有两个变体：当前 tip 的 `kPrefillMaxBlocks` 上限形状、重构前的
  `kPrefillPersistentBlocks` 直发形状。变体内的多步是原子的
  （常量 + 全部消费点一起换），include 步骤幂等（已存在则归一化）。
- **drift**：三种策略都落不了 → 结构化报告（`<worktree>/.port-report.json` + 终端），
  移植提交仍会生成（其余变更不受影响），但 `verify.py` 会红，build 不会盲目进行。

## 3. drift 了怎么办（按发生概率排序）

### 3.1 上游改了补丁的上下文行（最常见）
通常 3-way 或 transform 已经自动处理。若 transform 锚点也变了：
1. 看 `.port-report.json` 里该变更的 `diagnostic.current_context`（漂移区域的真实文本）；
2. 在 `port/manifest.json` 里更新对应 transform 的 `find`/`replace`
   （保持幂等：替换文本重复执行不得再追加内容）;
3. 重跑 `bash scripts/upgrade.sh`（engine 幂等，直接重放）。

### 3.2 上游重构了某个变更的目标结构（如 2026-09 的 sparse-MoE）
1. 读懂上游新结构（`git log -p <file>` 看重构 commit 的意图）；
2. 在 manifest 里**新增一个 transform 变体**描述新形状（放在最前——
   变体按序尝试，新形状优先）；
3. 若新结构让某个变更**失去意义**（上游自己修了），把该变更的
   `verify_after` 改成新结构下成立的谓词即可——engine 会把它判为 `present`，
   补丁/transform 全部不再触发；在 manifest 的 `description` 里注明"上游 <sha> 已覆盖"。

### 3.3 上游移动/改名了文件
1. 改 manifest 里该变更的 `file` 字段；
2. 重放 upgrade；`regen_patches.py` 会按新路径重录补丁。

### 3.4 上游新增/删除了带 `NINFER_SM86` 分支的文件
**什么都不用做**——080 扫描的文件集是运行时派生的，`git ls-files` 现场枚举。
这正是 v1 固定清单的失效点，v2 把它变成结构性质。

### 3.5 上游改了顶层 CMake 的架构门禁
当前 tip 在 `project()` 之前硬钉 `CMAKE_CUDA_ARCHITECTURES=89`（否则 FATAL_ERROR）。
010 的身份宏块插在 `project()` 之后，与该门禁不冲突（门禁先跑、宏后定义）。
若上游把门禁改成允许多架构，检查 010 的 `if/elseif` 分支是否仍覆盖新架构
（86/89 之外要加 `SM120` 一类的新身份）。

## 4. 验证层：不变量即契约

`tools/verify.py` 的 8 条检查是移植的**可执行规格**——上游更新后
任何一条变红，就说明移植的某个性质被打破了，且红的那条直接指向文件：

| # | 不变量 | 对应变更 |
|---|---|---|
| V1 | 顶层 CMake 按架构定义身份宏（目录作用域） | 010 |
| V2 | `src/CMakeLists.txt` 不再强定义 `NINFER_SM86` | 020 |
| V3 | 全树无裸 `NINFER_SM86` 预处理条件（兼容形式必须带 `NINFER_SM89`） | 080 |
| V4 | `src/` 无 per-part SM 数字面量（`SmCount = 170/128/...` 一类） | 050/060 |
| V5 | `device_sm_count()` 有声明有定义（缓存查询） | 030/040 |
| V6 | GDN chunked output 的波尺寸来自 `device_sm_count()` | 050 |
| V7 | sparse-MoE prefill 上限运行时派生 | 060 |
| V8 | 27B GDN gating 协同驻留数 split-aware（`SplitK == 8 ? 2 : 1`） | 070 |

设计原则：**编译是最后一道闸，不是第一道**。1400 文件 CUDA 构建跑 30 分钟才报
`no matching function for device_sm_count()` 是不可接受的——verify 在构建前 30 秒
就报。

## 5. 上游更新后的回归清单

upgrade 的自动部分之外，建议（有产物时）：

1. `bash scripts/build.sh --test` —— ctest 冒烟（重点看
   `ninfer_gdn_gating_proj_test`，070 的回归探针）；
2. `bash scripts/start.sh` 起服务，确认日志里的运行时 SM 数 = 92（L20）；
3. `python3 scripts/bench.py` 跑宽 prefill 探针（≥2688 token 不 abort =
   协同发射路径健康）+ code/prose/qa 对照 v1 数字（100/80/78 一档）；
4. 上游若动了 `w8_config.h` / `w8_rowsplit_*` / `pdl.cuh`（解码关键路径的调度表），
   额外对比 MTP 接受率（v1 code 贪心 65.5%）。
