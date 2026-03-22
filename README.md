# SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

<div align="center">

Bridging the gap between raw experience and policy improvement through automatic skill discovery.

</div>

<p align="center">
<img src="figs/pipeline.png" width="80%" alt="SKILLRL Pipeline Overview">
</p>

## 🔥 News

- **[03/02/2026]** Due to an accidental misconfiguration, we lost several hundred GitHub stars. If you previously starred this repo, we'd appreciate a re-star ⭐!
- **[02/23/2026]** We released all the model checkpoints on HuggingFace! Feel free to use them as warm starts for continued RL training.
- **[02/18/2026]** The code of SkillRL was released!
- **[02/10/2026]** SkillRL paper was released on [arXiv](https://arxiv.org/abs/2602.08234)!

## 📖 Overview

SkillRL is a framework that enables LLM agents to learn high-level, reusable behavioral patterns from past experiences. While traditional memory-based methods store redundant and noisy raw trajectories, SKILLRL abstracts these into a hierarchical skill library.

## 🤖 Key Features

- **Experience-based Skill Distillation**: Transforms successful trajectories into strategic patterns and failed ones into concise lessons from failure.

- **Hierarchical SKILLBANK**: Organizes knowledge into General Skills for universal strategic guidance and Task-Specific Skills for category-level heuristics.

- **Recursive Skill Evolution**: A dynamic mechanism where the skill library co-evolves with the agent's policy during RL by analyzing validation failures.

- **Context Efficiency**: Achieves 10-20% token compression compared to raw trajectory storage while enhancing reasoning utility. 

---

## 🧩 项目逻辑拆解（Project Logic Breakdown）

> 可配合阅读：<https://zread.ai/aiming-lab/SkillRL>

### 1) 顶层模块分工

| 目录 | 职责 | 关键文件 |
|------|------|----------|
| `memory_data/` | 训练所需记忆数据、技能库 JSON 与提示词模板 | `memory_data/alfworld/claude_style_skills.json`、`memory_data/prompt/prompt.txt` |
| `skill_generation/` | 将原始轨迹总结为 `general_skills`、`task_specific_skills`、`common_mistakes` | `skill_generation/alfworld.py`、`skill_generation/webshop.py`、`skill_generation/search.py` |
| `agent_system/` | 环境管理、多轮轨迹收集、奖励管理、技能检索注入 | `agent_system/environments/env_manager.py`、`agent_system/multi_turn_rollout/rollout_loop.py`、`agent_system/memory/skills_only_memory.py` |
| `verl/` | 训练框架主体（PPO/GRPO 训练器、Ray Worker、配置系统） | `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py` |
| `gigpo/` | GiGPO 算法（episode + step 两级优势估计） | `gigpo/core_gigpo.py` |
| `examples/` | 按任务划分的训练/生成启动脚本 | `examples/grpo_trainer/run_alfworld_skills.sh` |
| `tests/` | 单元测试、Ray CPU/GPU 测试与端到端测试 | `tests/` |
| `docs/` | 算法与工程文档 | `docs/algo/ppo.md` |

---

### 2) 端到端数据流总览

```
原始轨迹 (JSON)
      │
      ▼  skill_generation/{alfworld,webshop,search}.py
技能库 JSON (general_skills / task_specific_skills / common_mistakes)
      │
      ▼  verl/trainer/main_ppo.py  ──  TaskRunner.run()
初始化阶段:
  make_envs(config)  ──────────────────→  SkillsOnlyMemory / RetrievalMemory
  TrajectoryCollector(config, tokenizer)
  RayPPOTrainer(config, ..., traj_collector, envs)
      │
      ▼  RayPPOTrainer.fit()  (verl/trainer/ppo/ray_trainer.py)
      │
      ├─ ① 多轮交互  TrajectoryCollector.multi_turn_loop()
      │       env.reset() → 检索技能 → 环境 obs + 技能 → LLM 生成动作
      │       → env.step() → 收集 step_reward → 重复至 done/max_steps
      │       → DataProto (responses, rewards, log_probs, ...)
      │
      ├─ ② 奖励计算  EpisodeRewardManager.__call__()
      │       episode_rewards → token_level_scores → apply_invalid_action_penalty()
      │       → (可选) apply_kl_penalty()  → token_level_rewards
      │
      ├─ ③ 优势估计  compute_advantage()
      │       GAE / GRPO / GiGPO / REINFORCE++ / RLOO / REMAX
      │       → advantages, returns
      │
      ├─ ④ 策略更新
      │       critic_wg.update_critic(batch)   [仅 GAE]
      │       actor_wg.update_actor(batch)
      │
      └─ ⑤ 动态技能进化（可选, 每次验证后触发）
              _update_skills_from_validation()
              → SkillUpdater.analyze_failures()  [Azure o3 API]
              → SkillsOnlyMemory.add_skills()  → 回写技能库 JSON
```

---

### 3) 数据准备：技能库生成

**代码**：`skill_generation/alfworld.py`、`skill_generation/webshop.py`、`skill_generation/search.py`

```
原始记忆 JSON
    ↓ categorize_by_task_type()          # alfworld.py:70
    分任务类型:
      ALFWorld: pick_and_place / look_at_obj_in_light / clean / heat / cool / examine
      WebShop:  apparel / footwear / electronics / accessories / home_decor / beauty_health / other
      Search:   direct_retrieval / multi_hop_reasoning / entity_attribute_lookup / comparison
    ↓ generate_skills_for_task_type()    # alfworld.py:100+
    调用 Azure o3 API (OpenAIClient.generate_response)  # alfworld.py:55
    ↓
输出: claude_style_skills.json
    {
      "general_skills":       [{ skill_id, title, principle, when_to_apply, layer }],
      "task_specific_skills": { "pick_and_place": [...], "clean": [...], ... },
      "common_mistakes":      [{ mistake_id, description, why_it_happens, how_to_avoid }]
    }
```

- 成功轨迹 → 抽象为可迁移的通用/类别技能
- 失败轨迹 → 抽象为 `common_mistakes`（常见错误及规避方式）

---

### 4) 训练初始化与入口

**代码**：`verl/trainer/main_ppo.py`

```python
# 入口（Hydra 配置加载）
@hydra.main(config_path="config", config_name="ppo_trainer")   # main_ppo.py:29
def main(config): run_ppo(config)

# run_ppo() 初始化 Ray 集群，然后启动 TaskRunner  # main_ppo.py:34-51
runner = TaskRunner.remote()
ray.get(runner.run.remote(config))

# TaskRunner.run() 完成以下初始化步骤              # main_ppo.py:56-188
# 1. 下载模型检查点
local_path = copy_to_local(config.actor_rollout_ref.model.path)

# 2. 初始化环境（含技能记忆模块）
envs, val_envs = make_envs(config)                # main_ppo.py:71

# 3. 初始化分词器与处理器
tokenizer = hf_tokenizer(local_path)              # main_ppo.py:77
processor = hf_processor(local_path)              # main_ppo.py:78

# 4. 创建多轮轨迹收集器
traj_collector = TrajectoryCollector(config, tokenizer, processor)  # main_ppo.py:162

# 5. 创建训练/验证数据集
train_dataset = create_rl_dataset(...)            # main_ppo.py:166
val_dataset   = create_rl_dataset(...)            # main_ppo.py:167

# 6. 初始化 PPO 训练器并启动
trainer = RayPPOTrainer(...)                      # main_ppo.py:169-186
trainer.init_workers()                            # main_ppo.py:187
trainer.fit()                                     # main_ppo.py:188
```

---

### 5) 多轮交互收集轨迹

**代码**：`agent_system/multi_turn_rollout/rollout_loop.py`

```python
class TrajectoryCollector:                        # rollout_loop.py:29
    def multi_turn_loop(
        self, gen_batch, actor_rollout_wg, envs, is_train
    ) -> DataProto:
```

**数据流**（每个 training step）：

```
gen_batch (DataProto, 含 raw_prompt)
    ↓  envs.reset(kwargs)                     # env_manager.py:76
    初始观测 obs (text / image / anchor) + 技能上下文
    ↓
    循环 (step = 0 → max_steps):
        preprocess_single_sample()             # rollout_loop.py:43
            obs → input_ids / attention_mask / position_ids
        actor_rollout_wg.generate_sequences()  # LLM 推理
        tokenizer.decode → text action
        envs.step(text_actions)               # env_manager.py
            → next_obs, rewards, dones, infos
        收集: step_reward, is_action_valid, anchor_obs
    ↓
输出 DataProto:
    responses, attention_mask, log_probs,
    token_level_scores (episode_reward),
    step_rewards, is_action_valid, anchor_obs, uid, traj_uid
```

---

### 6) 奖励计算

**代码**：`agent_system/reward_manager/episode.py`、`verl/trainer/ppo/ray_trainer.py:200-224`

```python
class EpisodeRewardManager:                       # episode.py:20
    def __call__(self, data: DataProto):
        # 取 episode 最终奖励（0/1 成功标志），
        # 放到最后一个有效 response token 的位置
        reward_tensor[i, valid_response_length - 1] = episode_rewards[i]  # episode.py:79

# 无效动作惩罚（可选）
apply_invalid_action_penalty(data, coef)          # ray_trainer.py:200
    reward_tensor[i, valid_response_length-1] -= coef * action_invalids

# KL 惩罚（可选，use_kl_in_reward=True 时）
apply_kl_penalty(data, kl_ctrl)                   # ray_trainer.py:152
    kld = kl_penalty(old_log_probs, ref_log_prob)
    token_level_rewards = token_level_scores - beta * kld
```

---

### 7) 优势估计

**代码**：`verl/trainer/ppo/ray_trainer.py:244-362`

```python
def compute_advantage(data, adv_estimator, gamma, lam, ...):  # ray_trainer.py:244
```

| 估计器 | 算法 | 实现位置 |
|--------|------|---------|
| `GAE` | Generalized Advantage Estimation（需 Critic） | `verl/trainer/ppo/core_algos.py:compute_gae_advantage_return` |
| `GRPO` | 同一 prompt 多条轨迹组内归一化 | `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage` |
| `GiGPO` | episode 组 + step 组两级优势，支持相似观测聚类 | `gigpo/core_gigpo.py:compute_gigpo_outcome_advantage` |
| `REINFORCE++` | 带 baseline 的 outcome 优势 | `verl/trainer/ppo/core_algos.py:compute_reinforce_plus_plus_outcome_advantage` |
| `RLOO` | Leave-One-Out 估计 | `verl/trainer/ppo/core_algos.py:compute_rloo_outcome_advantage` |
| `REMAX` | Reward Maximization baseline | `verl/trainer/ppo/core_algos.py:compute_remax_outcome_advantage` |

**GiGPO 数据流**（`gigpo/core_gigpo.py`）：

```
token_level_rewards  ──→ episode-level group advantage
step_rewards         ──→ step-level group advantage
anchor_obs           ──→ are_similar() 文本相似度聚类  (core_gigpo.py:72)
    ↓
advantages = episode_adv + step_advantage_w * step_adv
```

---

### 8) 策略更新

**代码**：`verl/trainer/ppo/ray_trainer.py`

```python
# Critic 更新（仅 GAE 估计器，需要 value 网络）       # ray_trainer.py:1452-1457
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)

# Actor 更新（PPO clip 损失）                         # ray_trainer.py:1459-1466
if self.config.trainer.critic_warmup <= self.global_steps:
    actor_output = self.actor_rollout_wg.update_actor(batch)
# 输入: input_ids, attention_mask, old_log_probs, advantages, returns
# 输出: actor_loss, entropy, pg_clipfrac, ...

# 检查点保存                                          # ray_trainer.py:1125-1156
_save_checkpoint()
# 保存 actor weights + critic weights + dataloader 状态
```

---

### 9) 技能检索注入

**代码**：`agent_system/memory/skills_only_memory.py`

```python
class SkillsOnlyMemory(BaseMemory):               # skills_only_memory.py:36
    def __init__(self, skills_json_path, retrieval_mode="template",
                 embedding_model_path=None, task_specific_top_k=None):
        # 加载技能库 JSON                          # skills_only_memory.py:86-88
        with open(skills_json_path) as f:
            self.skills = json.load(f)
        # embedding 模式下预计算技能向量            # skills_only_memory.py:108-109
        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()
```

**两种检索模式**：

| 模式 | 流程 | 关键代码 |
|------|------|---------|
| **Template**（默认） | `_detect_task_type(task_desc)` 关键词匹配任务类别 → 返回该类别全部技能 + 前 top_k 条通用技能 | `skills_only_memory.py:115-177` |
| **Embedding** | 任务描述 → SentenceTransformer 编码 → 与所有技能向量做余弦相似度 → top-k 排序注入 | `skills_only_memory.py:253+` |

**技能注入时机**（在环境 `reset()` 时）：

```python
# env_manager.py:90-98 (SearchEnvironmentManager.reset 为例)
for task in self.tasks:
    memories = self.retrieval_memory.retrieve(
        task_description=task, top_k=top_k, ...)
    self.retrieved_memories.append(memories)
# retrieved_memories 之后在 build_text_obs() 中拼入 prompt
```

---

### 10) 动态技能进化（可选）

**代码**：`verl/trainer/ppo/ray_trainer.py:837-931`、`agent_system/memory/skill_updater.py`

**触发条件**：每次验证结束后，若某任务类型成功率 < `update_threshold`（默认 0.4）：

```python
def _update_skills_from_validation(
    self, sample_inputs, sample_outputs, sample_scores, success_rate
):                                                # ray_trainer.py:837

    # 1. 检测低成功率任务类型
    for task_key, rate in success_rate.items():
        if rate < threshold:
            low_success_tasks.append(task_type)  # ray_trainer.py:855-860

    # 2. 收集失败轨迹
    failed_trajectories = _collect_failed_trajectories(...)  # ray_trainer.py:869

    # 3. 调用 o3 API 分析失败并生成新技能
    new_skills = skill_updater.analyze_failures(
        failed_trajectories=failed_trajectories,
        current_skills=retrieval_memory.skills,
        evolution_variant=evolution_variant,     # v0/v2/v3/v4
        frozen_layers=frozen_layers,             # ray_trainer.py:898-903
    )

    # 4. 注入新技能到训练环境（不回流验证环境，防止数据泄漏）
    self.envs.retrieval_memory.add_skills(new_skills, category='general')  # ray_trainer.py:913

    # 5. 保存更新后的技能库 JSON
    train_memory.save_skills(save_path)          # ray_trainer.py:927
```

**`SkillUpdater.analyze_failures()`**（`skill_updater.py:45`）：

```
failed_trajectories (task, trajectory, task_type)
    ↓  _build_analysis_prompt()   → 构造包含失败轨迹与当前技能库的 prompt
    ↓  Azure o3 API               → 生成新技能 JSON
    ↓  _parse_skills_response()   → 解析并去重
输出: List[Dict]  新技能列表 (skill_id="dyn_NNN", ...)
```

---

### 11) 三个环境如何接线

**代码**：`agent_system/environments/env_manager.py`

三个环境共用同一套 `EnvironmentManagerBase` 接口与 SkillBank 接入方式，差异如下：

| 环境 | 管理类 | 动作空间 | 技能分类维度 |
|------|--------|---------|-------------|
| ALFWorld | `AlfWorldEnvironmentManager` | 文本指令（pick up / go to / ...） | 6 类家务任务（pick_and_place、clean、heat 等） |
| WebShop | `WebShopEnvironmentManager` | 搜索/点击/购买操作 | 7 类商品类别（apparel、electronics 等） |
| Search | `SearchEnvironmentManager` | 搜索查询文本 | 4 类问题类型（direct_retrieval、multi_hop 等） |

**通用接口**（`agent_system/environments/base.py`）：

```python
class EnvironmentManagerBase:
    def reset(self, kwargs) -> (obs, infos)     # 重置并返回初始观测（含技能上下文）
    def step(self, text_actions) -> (obs, rewards, dones, infos)
    def build_text_obs() -> List[str]           # 将 obs + 检索技能拼装成 LLM prompt
```

---

### 12) 快速阅读建议（从工程角度）

建议按以下顺序阅读代码：

1. `examples/grpo_trainer/run_*_skills.sh`（先看训练参数如何启用技能）
2. `verl/trainer/main_ppo.py`（看初始化流程与各模块组装）
3. `verl/trainer/ppo/ray_trainer.py`（看训练主循环：`fit()`）
4. `agent_system/multi_turn_rollout/rollout_loop.py`（看多轮交互与轨迹收集）
5. `agent_system/environments/env_manager.py`（看环境与记忆模块挂接）
6. `agent_system/memory/skills_only_memory.py`（看技能检索逻辑）
7. `agent_system/memory/skill_updater.py`（看动态技能进化）
8. `skill_generation/*.py`（看技能库构建过程）

---

## 📥 Model Download

You can directly download the model weights by following the links below.

<table>
  <thead>
    <tr>
      <th align="center">Task</th>
      <th align="center">Model</th>
      <th align="center">Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2">🧭 ALFWorld</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Alfworld-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Alfworld-7B-RL">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center" rowspan="2">🛍️ WebShop</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Webshop-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Webshop-7B-RL">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center" rowspan="2">🔍 Search</td>
      <td align="center">SFT Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Search-7B-SFT">🤗 HuggingFace</a></td>
    </tr>
    <tr>
      <td align="center">RL Model</td>
      <td align="center"><a href="https://huggingface.co/Jianwen/Search-7B-RL">🤗 HuggingFace</a></td>
    </tr>
  </tbody>
</table>


---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/aiming-lab/SkillRL.git
cd SkillRL

pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .

pip install openai
```

### Environment Setup

**ALFWorld**
```bash
pip install alfworld
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0

# Download PDDL & Game files and pre-trained MaskRCNN detector
alfworld-download -f
```

**WebShop**
```bash
cd agent_system/environments/env_package/webshop
./setup.sh -d all
```

**Search**
```bash
cd agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

**API Setup**
```
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT=""
```

---

## 🏃 Training

### Memory Data Generation
The first step of our training pipeline uses the base model to generate memory data. This data serves as the foundation for the agent's initial experiences. The specific prompt used to guide this generation can be found at: `memory_data/prompt/prompt.txt`.

### Supervised Fine-Tuning (SFT)
Prior to RL, we perform SFT to endow the model with basic task capabilities and instruction-following alignment. We use [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) as our framework for the SFT stage.

### RL With SkillBank

#### Template Mode

Template mode uses keyword matching to detect the task category and injects all skills for that category into the prompt.  No embedding model is required.

```bash
# ALFWorld
export MODEL_PATH=YOUR_SFT_CKPT
bash examples/grpo_trainer/run_alfworld_skills.sh

# WebShop
bash examples/grpo_trainer/run_webshop_skills.sh

# Search
bash examples/grpo_trainer/run_search_skills.sh
```

Key config flags added by these scripts:

```
+env.use_skills_only_memory=True
+env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json
+env.skills_only_memory.top_k=6              
+env.skills_only_memory.enable_dynamic_update=True
+env.skills_only_memory.update_threshold=0.4
+env.skills_only_memory.max_new_skills=3
```

#### Embedding Mode

Embedding mode uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) to rank skills by semantic similarity to the task description.  Both general skills and task-specific skills are searched cross-category and only the top-k most relevant are injected.  Skill embeddings are pre-computed once at startup.

```bash
export MODEL_PATH=YOUR_SFT_CKPT

python3 -m verl.trainer.main_ppo \
    ... \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=Qwen/Qwen3-Embedding-0.6B \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.task_specific_top_k=5
```

---

## ⚙️ Skill Memory Configuration

All parameters live under `env.skills_only_memory.*` (Hydra / OmegaConf).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skills_json_path` | str | — | **Required.** Path to the skills JSON. |
| `retrieval_mode` | str | `"template"` | `"template"` or `"embedding"`. |
| `embedding_model_path` | str | `"Qwen/Qwen3-Embedding-0.6B"` | Local path or HF model ID.  Only used when `retrieval_mode=embedding`. |
| `top_k` | int | `6` | Number of general skills injected per episode. |
| `task_specific_top_k` | int | `None` | Max task-specific skills per episode.  `None` = all (template) / same as `top_k` (embedding). |
| `enable_dynamic_update` | bool | `False` | Evolve the skill bank during training using validation failures. |
| `update_threshold` | float | `0.4` | Min success rate below which skills are updated. |
| `max_new_skills` | int | `3` | Maximum new skills added per update cycle. |
| `evolution_variant` | str | `"v0"` | Semantic layered evolution variant (`v0`/`v2`/`v3`/`v4`) for dynamic updates. |
| `frozen_layers` | list[str] | `[]` | Layer names that must not be mutated during dynamic updates (e.g. `["action"]`). |
---

## 📋 Skill Bank Format

Skills are stored in a JSON file with three top-level keys:

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Systematic Exploration",
      "principle": "Search every plausible surface exactly once …",
      "when_to_apply": "Anytime the goal object count is not yet met …",
      "layer": "plan"
    }
  ],
  "task_specific_skills": {
    "pick_and_place": [
      {
        "skill_id": "pnp_001",
        "title": "Direct Path Planning",
        "principle": "Navigate directly to the target receptacle …",
        "when_to_apply": "After picking up the object …"
      }
    ],
    "clean": [ … ],
    "heat":  [ … ]
  },
  "common_mistakes": [
    {
      "mistake_id": "err_001",
      "description": "Repeating the same action after it fails.",
      "why_it_happens": "Agent does not track action history.",
      "how_to_avoid": "Check the admissible actions list and try an alternative."
    }
  ]
}
```

### Generating a New Skill Bank

Use the provided generation scripts (requires Azure API access):

```bash
# ALFWorld
python skill_generation/alfworld.py \
    --memory_path memory_data/alfworld/generated_memories_alfworld_total.json \
    --output_path memory_data/alfworld/claude_style_skills.json

# WebShop
python skill_generation/webshop.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json

# Search
python skill_generation/search.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json
```

---

## 📚 Citation
If you find our work helpful, please consider citing:

```bibtex
@article{xia2026skillrl,
  title={SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and Liu, Jiaqi and Zeng, Kaide and Wang, Yu and Han, Siwei and Zhou, Yiyang and Zhao, Xujiang and Chen, Haifeng and others},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}
```

## 🙏 Acknowledgement
We would like to express our gratitude to the open-source community and the following projects for making this work possible: 
[verl-agent](https://github.com/langfengQ/verl-agent), [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [Qwen](https://github.com/QwenLM/Qwen), etc.
