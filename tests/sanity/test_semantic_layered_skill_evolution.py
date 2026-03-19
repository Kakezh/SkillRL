import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = REPO_ROOT / "agent_system" / "memory"


def _load_memory_module(module_name: str):
    agent_system_pkg = sys.modules.setdefault("agent_system", types.ModuleType("agent_system"))
    memory_pkg = sys.modules.get("agent_system.memory")
    if memory_pkg is None:
        memory_pkg = types.ModuleType("agent_system.memory")
        memory_pkg.__path__ = [str(MEMORY_DIR)]
        sys.modules["agent_system.memory"] = memory_pkg
    agent_system_pkg.memory = memory_pkg

    if "agent_system.memory.base" not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(
            "agent_system.memory.base", MEMORY_DIR / "base.py"
        )
        base_mod = importlib.util.module_from_spec(base_spec)
        sys.modules["agent_system.memory.base"] = base_mod
        base_spec.loader.exec_module(base_mod)

    spec = importlib.util.spec_from_file_location(
        f"agent_system.memory.{module_name}", MEMORY_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"agent_system.memory.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


def test_add_skills_respects_frozen_layers():
    skills_only_memory = _load_memory_module("skills_only_memory")
    SkillsOnlyMemory = skills_only_memory.SkillsOnlyMemory
    with tempfile.TemporaryDirectory() as tmp_dir:
        skills_path = Path(tmp_dir) / "skills.json"
        skills_path.write_text(
            json.dumps(
                {
                    "general_skills": [],
                    "task_specific_skills": {},
                    "common_mistakes": [],
                }
            ),
            encoding="utf-8",
        )
        memory = SkillsOnlyMemory(skills_json_path=str(skills_path))
        added = memory.add_skills(
            [
                {"skill_id": "dyn_001", "title": "Freeze action", "principle": "p", "layer": "action"},
                {"skill_id": "dyn_002", "title": "Update plan", "principle": "p", "layer": "plan"},
            ],
            category="general",
            frozen_layers=["action"],
        )
        assert added == 1
        assert [s["skill_id"] for s in memory.skills["general_skills"]] == ["dyn_002"]


def test_reassign_dyn_ids_assigns_variant_layer():
    skill_updater = _load_memory_module("skill_updater")
    SkillUpdater = skill_updater.SkillUpdater
    updater = SkillUpdater.__new__(SkillUpdater)
    reassigned = updater._reassign_dyn_ids(
        [{"title": "A", "principle": "B"}],
        start_idx=7,
        evolution_variant="v3",
    )
    assert reassigned[0]["skill_id"] == "dyn_007"
    assert reassigned[0]["layer"] == "scene"


def test_analysis_prompt_includes_layer_constraints():
    skill_updater = _load_memory_module("skill_updater")
    SkillUpdater = skill_updater.SkillUpdater
    updater = SkillUpdater.__new__(SkillUpdater)
    updater.max_new_skills_per_update = 2
    prompt = updater._build_analysis_prompt(
        failed_trajectories=[
            {
                "task": "Find a mug",
                "task_type": "examine",
                "trajectory": [{"action": "look", "observation": "nothing"}],
            }
        ],
        current_skills={"general_skills": [], "task_specific_skills": {}},
        next_dyn_idx=5,
        evolution_variant="v2",
        frozen_layers=["action"],
    )
    assert "Generate skills for layer='plan' only." in prompt
    assert "Frozen layers (do not mutate): action" in prompt


def test_analysis_prompt_v4_allows_plan_and_scene():
    skill_updater = _load_memory_module("skill_updater")
    SkillUpdater = skill_updater.SkillUpdater
    updater = SkillUpdater.__new__(SkillUpdater)
    updater.max_new_skills_per_update = 1
    prompt = updater._build_analysis_prompt(
        failed_trajectories=[
            {
                "task": "Buy a safe laptop",
                "task_type": "electronics",
                "trajectory": [{"action": "search", "observation": "few options"}],
            }
        ],
        current_skills={"general_skills": [], "task_specific_skills": {}},
        next_dyn_idx=1,
        evolution_variant="v4",
        frozen_layers=["action"],
    )
    assert "Generate skills for layer in {plan, scene} only" in prompt
