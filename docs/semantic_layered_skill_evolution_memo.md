# Semantic Layered Skill Evolution on SkillRL — Experiment Design Memo

## 1) Problem framing

目标是在 **不改 SkillRL 主框架语义主线** 的前提下，验证一个更可解释的演化范式：
- 将 SkillRL 中已有蒸馏 skill 视为复合技能单元，拆成三层：**Action / Plan / Scene**；
- 保持 Action 层尽量稳定；
- 将演化压力集中到 Plan 层与 Scene 层；
- 对比“整体 skill 演化（Vanilla）”与“分层演化（Plan、Scene、Plan+Scene）”在性能、泛化与稳定性上的差异。

本文主线只回答一个故事：
**Skill 的语义分层是否能在不改 backbone 的条件下，带来更 clean、更可解释、更可迁移的演化收益。**

---

## 2) Final thesis

将 SkillRL 中蒸馏 skill 拆成三层：
- **Action Layer**：动作原语（atomic elements）
- **Plan Layer**：动作组合形成的可复用计划模板（sub-task structures）
- **Scene Layer**：计划在具体任务上下文/环境条件下的适配机制

核心假设：
1. Action 层应尽量稳定，不是主演化对象；
2. Plan 层是第一主演化对象；
3. Scene 层是第二主演化对象；
4. 仅在 Plan+Scene 层演化，应优于整体 skill 黑箱演化，并更具可解释性与迁移性。

---

## 3) Dominant contribution + non-goals

### Dominant contribution（唯一主贡献）
**Semantic Layered Skill Evolution on top of SkillRL**：
- 在 SkillRL 之上构建 Action/Plan/Scene 分层视角；
- 严格比较 V0~V4（可选 V5）以验证“Plan+Scene 演化优先、Action 稳定优先”的 thesis。

### Non-goals（明确不做主贡献）
- 不提出新 RL backbone；
- 不重写 SkillRL 主循环；
- 不将 SkillNet 5 维评分作为主算法；
- 不设计新 skill format 作为主线贡献；
- 不以 graph/memory pruning/scorer training 为主线。

上述内容最多进入 appendix/optional analysis/future work。

---

## 4) Research questions

- **RQ1**: SkillRL 现有 skill 能否被有效拆解为 Action/Plan/Scene 三层？
- **RQ2**: Plan-only 演化是否优于整体 skill 演化？
- **RQ3**: Scene-only 演化是否带来额外上下文适配收益？
- **RQ4**: Plan+Scene 联合演化是否在性能、泛化、稳定性上最优？
- **RQ5**: Action 层是否应固定？若允许 Action 演化，是否更复杂但不更好？
- **RQ6**: SkillNet 5 维评分能否作为次要解释工具区分不同版本差异？

---

## 5) System variants（必须版本）

- **V0: Vanilla SkillRL**
  - 原始 SkillRL，skill 作为整体演化。
- **V1: Decompose-Only**
  - 仅完成 Action/Plan/Scene 拆解映射，不做分层演化。
- **V2: Plan-Evolve**
  - Action 固定；仅 Plan 层演化。
- **V3: Scene-Evolve**
  - Action 固定、Plan 固定；仅 Scene 层演化。
- **V4: Plan+Scene Evolve**
  - Action 固定；Plan 与 Scene 联合演化。
- **V5: Action-Mutable（optional stress test）**
  - Action 也可演化，仅用于检验“Action 应稳定”的假设。

统一公平约束：
- 相同基础模型、训练步数、预算上限、环境配置、随机种子集合（建议 3 seeds）；
- 相同评估频率与停止准则。

---

## 6) Benchmark plan

### 主线 benchmark（必须）
1. **ALFWorld**（主结果必含）
2. **WebShop**（主结果必含）
3. **Search family**（若在当前代码流水线可自然落地则纳入主结果；否则作为补充）

### 外部验证（可选）
- **ScienceWorld** 仅作为 optional external validation，放附录，不影响主结果结论。

### 数据划分与评估切片
每个环境统一三种切片：
- **Seen scene/task**：训练分布内；
- **Shifted scene**：轻度分布偏移；
- **Unseen scene/task**：分布外任务或组合。

---

## 7) Metrics

## 7.1 Task-Level（主指标）
- Success rate
- Step/action count
- Token cost / latency
- Seen vs shifted vs unseen transfer performance

## 7.2 Layered-Skill Metrics（主线配套指标）
- **Action vocabulary drift / Action stability**
- **Plan reuse rate**
- **Scene adaptation success rate**
- **Plan revision frequency**
- **Plan complexity change**（例如模板长度、分支数、子目标数）
- **Evolution stability**（跨 seed 方差、曲线振荡）

## 7.3 Secondary analysis（仅次要）
SkillNet 5 维质量分：
- Safety
- Completeness
- Executability
- Maintainability
- Cost-Awareness

仅用于解释，不作为训练主信号。

---

## 8) Ablation plan

### A. Layer ablation（必做）
- V0 Vanilla
- V1 Decompose-only
- V2 Plan-only
- V3 Scene-only
- V4 Plan+Scene
- V5 Action-mutable（可选）

### B. Action stability ablation（必做）
- Action fixed
- Action partially mutable
- Action fully mutable

### C. Scene generalization ablation（必做）
- Seen
- Shifted
- Unseen

### D. Optional quality analysis（可选）
- 不使用 SkillNet 评分做训练
- 仅分析与 very light filtering（如做）
- 结果进入 appendix，不改变主线结论

---

## 9) Main tables / figures（论文导向）

### Main Table 1（核心总表）
环境 × 版本（V0~V4）
- Success rate ↑
- Step count ↓
- Token cost ↓
- Seen/Shifted/Unseen 分项

### Main Table 2（分层指标表）
版本（V1~V4，V5 可选）
- Action drift ↓
- Plan reuse ↑
- Scene adaptation success ↑
- Plan revision frequency（合理区间）
- Plan complexity change（可控）
- Evolution stability ↑

### Figure 1（训练动态）
- 各版本 success-rate learning curves（带 seed 方差带）

### Figure 2（可解释性案例）
- 同一任务下 V0 vs V2/V4 的 Plan/Scene 变化示例

### Figure 3（泛化剖面）
- Seen/Shifted/Unseen 三切片柱状对比（V0~V4）

### Appendix Table（可选）
- SkillNet 5 维评分对比，仅解释性证据。

---

## 10) Run order + decision gates

## Stage 1: Feasibility / sanity check
目标：验证三层拆解在工程上可落地、可统计。
- 产出：Action/Plan/Scene 映射覆盖率、最小可解释样例、基础统计脚本输出。
- Gate G1：若拆解不可操作（覆盖率低、无法稳定映射），则收缩为“skill structure analysis”，停止大规模演化实验。

## Stage 2: Main comparison（主结果）
目标：跑通 V0~V4 在 ALFWorld + WebShop（Search 视可落地性加入）。
- 产出：Main Table 1 + Figure 1。
- Gate G2：
  - 若 Plan-only 明显有效而 Scene-only 无增益：主线收敛为 Plan-layer evolution；
  - 若 Plan+Scene 明显优于其余：进入 Stage 3 完整消融。

## Stage 3: Ablation
目标：验证关键机制而非扩展故事。
- B: Action stability（fixed/partial/full mutable）
- C: Scene generalization（seen/shifted/unseen）
- Plan complexity 变化剖面
- 产出：Main Table 2 + Figure 3。
- Gate G3：
  - 若 Scene 有用但 Action stability 假设不成立：重述 thesis 中 Action 固定前提为“弱固定/受限可变”。

## Stage 4: Optional secondary analysis
目标：补充解释，不改主结论。
- SkillNet 5 维评分（离线分析）
- optional very light filtering
- Gate G4：若无解释增益，直接降级 appendix。

---

## 11) Risk analysis

1. **分层边界不清（Action/Plan/Scene 混叠）**
   - 缓解：先定义可执行标注规则与映射优先级；先做 V1 覆盖率检查再进主实验。

2. **Scene-only 收益弱，导致主线不闭环**
   - 缓解：保留 Gate：若 Scene 贡献弱，收敛为 Plan 主线并将 Scene 作为次要发现。

3. **Action-mutable 带来不稳定但偶发高分**
   - 缓解：强调稳定性与方差指标，不以单点最优分替代整体结论。

4. **Benchmark 资源不足（Search/ScienceWorld 难完整跑）**
   - 缓解：确保 ALFWorld + WebShop 先闭环；Search 与 ScienceWorld 只在资源允许时扩展。

5. **SkillNet 喧宾夺主风险**
   - 缓解：在实验协议中明确“仅离线分析，不进主训练目标”。

---

## 12) Repo-level implementation touchpoints（高层，不写代码）

以下仅定义改动触点，不涉及重写主循环：

1. **技能表示读取层（memory_data + skill loading）**
   - 在现有 skill 读取后增加三层视图映射（不改原始 skill 文件结构作为主前提）。

2. **环境交互与轨迹记录层（agent_system / env logging）**
   - 增加可观测字段：Action tokens、Plan template id、Scene adaptation tag。

3. **技能更新/演化策略入口（当前 SkillRL 动态更新路径）**
   - V2/V3/V4 在策略选择阶段限制可变层（Plan-only / Scene-only / Plan+Scene）。

4. **评估与统计脚本层（evaluation/analysis scripts）**
   - 统一输出 task-level + layered-skill metrics；
   - 支持 seen/shifted/unseen 切片汇总。

5. **实验配置层（Hydra configs / recipe scripts）**
   - 通过配置开关定义 V0~V5，不新增并行算法分支。

6. **可解释样例导出层**
   - 导出固定样本上的 Plan/Scene 演化前后对照，用于 Figure 2。

---

## 13) Minimum publishable version（最小可发表闭环）

### 必做（最小闭环）
1. Stage 1 完成并通过 G1（可操作拆解）；
2. 在 **ALFWorld + WebShop** 完成 V0~V4；
3. 提供 Task-level 主指标 + Layered-skill 主指标；
4. 完成 A/B/C 三类消融中的核心子集：
   - A: V0, V1, V2, V4（V3 若资源紧张可降级到补充）
   - B: Action fixed vs full mutable
   - C: Seen vs Unseen（Shifted 可补充）
5. 给出明确结论：
   - Plan 是否是主演化层；
   - Scene 是否提供额外泛化收益；
   - Action 稳定假设是否成立。

### 扩展版（有资源再做）
- 完整 V0~V5 + 全量 shifted 切片 + Search family 全覆盖 + ScienceWorld external validation + SkillNet secondary analysis appendix。

---

## Best path（单一路径，避免分叉叙事）

**先闭环再扩展**：
1. 先用 ALFWorld + WebShop 跑通 V0~V4，拿到主表结论；
2. 再做 Action stability 与 scene 泛化消融巩固 thesis；
3. 最后用 SkillNet 5 维评分做解释性补充；
4. 若任何 gate 不满足，按预设规则收敛主线，不额外开启新故事。
