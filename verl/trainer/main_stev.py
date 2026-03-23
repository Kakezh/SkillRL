# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Lightweight no-RL experiment pipeline:
collect trajectories in sandbox environments and evolve skills via STEV-style
textual feedback, without PPO/GRPO weight updates.
"""

import os
import re
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.fs import copy_to_local

MAX_OBSERVATION_LENGTH = 3000
MAX_ACTION_LENGTH = 2000
TASK_TYPE_KEYWORDS = (
    ("clean", "clean"),
    ("heat", "heat"),
    ("cool", "cool"),
    ("examine", "examine"),
)


def _extract_task_description(inp: str) -> str:
    patterns = [
        r'(?:Your task is to|Task:|you need to)[:\s]+(.*?)(?:\n|$)',
        r'(?:goal|objective)[:\s]+(.*?)(?:\n|$)',
    ]
    for pat in patterns:
        match = re.search(pat, inp, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:1000]
    return inp[:1000]


def _detect_task_type_from_input(inp: str) -> str:
    inp_lower = inp.lower()
    if 'look at' in inp_lower and ('lamp' in inp_lower or 'light' in inp_lower):
        return 'look_at_obj_in_light'
    for keyword, task_type in TASK_TYPE_KEYWORDS:
        if keyword in inp_lower:
            return task_type
    return 'pick_and_place'


def _collect_trajectory_candidates(
    inputs: list[str],
    outputs: list[str],
    scores: list[float],
    traj_uids,
    max_failure_score: float,
    max_trajectories: int,
) -> list[dict]:
    selected = []
    seen_uids = set()
    for i, (inp, out, score) in enumerate(zip(inputs, outputs, scores)):
        uid = traj_uids[i] if traj_uids is not None and i < len(traj_uids) else f"traj_{i}"
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        if score <= max_failure_score:
            selected.append({
                "task": _extract_task_description(inp),
                "task_type": _detect_task_type_from_input(inp),
                "trajectory": [
                    {"action": "", "observation": inp[:MAX_OBSERVATION_LENGTH]},
                    {"action": out[:MAX_ACTION_LENGTH], "observation": ""},
                ],
            })
        if len(selected) >= max_trajectories:
            break
    return selected


@hydra.main(config_path="config", config_name="stev_generator", version_base=None)
def main(config):
    run_stev(config)


def run_stev(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = STEVTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class STEVTaskRunner:
    def run(self, config):
        from agent_system.environments import make_envs
        from agent_system.memory.skill_updater import SkillUpdater
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        envs, _ = make_envs(config)
        retrieval_memory = getattr(envs, "retrieval_memory", None)
        if retrieval_memory is None:
            raise RuntimeError(
                "No retrieval memory found. Please enable env.use_skills_only_memory "
                "(and configure env.skills_only_memory.*) for STEV evolution."
            )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="stev_rollout_pool",
        )
        actor_rollout_cls = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=config.actor_rollout_ref,
            role="actor_rollout",
        )
        actor_rollout_wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=actor_rollout_cls,
            device_name=config.trainer.device,
        )
        actor_rollout_wg.init_model()

        skill_updater = SkillUpdater(
            max_new_skills_per_update=config.stev.max_new_skills,
            max_completion_tokens=config.stev.max_completion_tokens,
        )

        global_steps = 0
        os.makedirs(config.trainer.default_local_dir, exist_ok=True)
        for epoch in range(config.trainer.total_epochs):
            for batch_dict in train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                rollout_batch = traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                    is_train=True,
                )
                inputs = tokenizer.batch_decode(rollout_batch.batch["prompts"], skip_special_tokens=True)
                outputs = tokenizer.batch_decode(rollout_batch.batch["responses"], skip_special_tokens=True)
                scores = rollout_batch.batch["episode_rewards"].cpu().tolist()
                traj_uids = rollout_batch.non_tensor_batch.get("traj_uid")

                candidates = _collect_trajectory_candidates(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    traj_uids=traj_uids,
                    max_failure_score=config.stev.max_failure_score,
                    max_trajectories=config.stev.max_failed_trajectories,
                )
                if not candidates:
                    global_steps += 1
                    continue

                new_skills = skill_updater.analyze_failures(
                    failed_trajectories=candidates,
                    current_skills=retrieval_memory.skills,
                    evolution_variant=config.env.skills_only_memory.get("evolution_variant", "v0"),
                    frozen_layers=config.env.skills_only_memory.get("frozen_layers", []),
                )
                if new_skills:
                    skill_category = config.stev.get("skill_category", "general")
                    added = retrieval_memory.add_skills(
                        new_skills,
                        category=skill_category,
                        frozen_layers=config.env.skills_only_memory.get("frozen_layers", []),
                    )
                    if added > 0:
                        save_file = f"updated_skills_step{global_steps}.json"
                        save_path = os.path.join(config.trainer.default_local_dir, save_file)
                        retrieval_memory.save_skills(save_path)
                        print(f"[STEV] step={global_steps} added={added} saved={save_path}")
                global_steps += 1

            print(f"[STEV] Finished epoch {epoch + 1}/{config.trainer.total_epochs}")


if __name__ == "__main__":
    main()
