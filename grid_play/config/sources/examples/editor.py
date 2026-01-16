from __future__ import annotations

from typing import Any

import streamlit as st
from grid_universe.levels.factories import (
    create_agent,
    create_box,
    create_coin,
    create_core,
    create_door,
    create_exit,
    create_floor,
    create_hazard,
    create_immunity_effect,
    create_key,
    create_monster,
    create_phasing_effect,
    create_portal,
    create_speed_effect,
    create_wall,
)
from grid_universe.gym_env import GridUniverseEnv
from grid_universe.moves import MOVE_FN_REGISTRY
from grid_universe.objectives import OBJECTIVE_FN_REGISTRY
from grid_universe.renderer.image import (
    ImageMap,
    IMAGE_MAP_REGISTRY,
    DEFAULT_ASSET_ROOT,
)

from grid_play.config.sources.base import register_level_source
from grid_play.config.sources.level_editor import ToolSpec, make_level_editor_source


# -----------------------
# Parameter UIs
# -----------------------


def agent_params() -> dict[str, Any]:
    return {"health": int(st.number_input("Health", 1, 99, 5, key="agent_health"))}


def floor_params() -> dict[str, Any]:
    return {"cost": int(st.number_input("Move Cost", 1, 99, 1, key="floor_cost"))}


def coin_params() -> dict[str, Any]:
    reward = int(st.number_input("Reward (0=none)", 0, 999, 0, key="coin_reward"))
    return {"reward": reward if reward > 0 else None}


def core_params() -> dict[str, Any]:
    reward = int(st.number_input("Reward (0=none)", 0, 999, 10, key="core_reward"))
    required = bool(st.checkbox("Required?", value=True, key="core_required"))
    return {"reward": reward if reward > 0 else None, "required": required}


def key_params() -> dict[str, Any]:
    key_id = st.text_input("Key ID", value="A", key="key_id").strip() or "A"
    return {"key_id": key_id}


def door_params() -> dict[str, Any]:
    key_id = st.text_input("Door Key ID", value="A", key="door_key_id").strip() or "A"
    return {"key_id": key_id}


def moving_params(prefix: str) -> dict[str, Any]:
    direction_label = st.selectbox(
        "Direction",
        ["None", "Up", "Down", "Left", "Right"],
        key=f"{prefix}_dir",
    )
    direction = None
    if direction_label != "None":
        direction = direction_label.lower()
    on_collision_label = st.selectbox(
        "On Collision",
        ["Stop", "Bounce", "Destroy"],
        index=1,
        key=f"{prefix}_on_collision",
    )
    on_collision = on_collision_label.lower()
    speed = int(st.number_input("Speed (tiles/step)", 1, 10, 1, key=f"{prefix}_speed"))
    return {
        "moving_direction": direction,
        "moving_on_collision": on_collision,
        "moving_speed": speed,
    }


def box_params() -> dict[str, Any]:
    pushable = bool(st.checkbox("Pushable?", value=True, key="box_pushable"))
    return {"pushable": pushable, **moving_params("box")}


def monster_params() -> dict[str, Any]:
    damage = int(st.number_input("Damage", 1, 50, 3, key="monster_damage"))
    lethal = bool(st.checkbox("Lethal?", value=False, key="monster_lethal"))
    return {"damage": damage, "lethal": lethal, **moving_params("monster")}


def hazard_params(kind: str) -> dict[str, Any]:
    default_lethal = kind == "lava"
    lethal = bool(st.checkbox("Lethal?", value=default_lethal, key=f"{kind}_lethal"))
    damage = (
        0 if lethal else int(st.number_input("Damage", 1, 50, 2, key=f"{kind}_damage"))
    )
    return {"appearance": kind, "damage": damage, "lethal": lethal}


def speed_params() -> dict[str, Any]:
    mult = int(st.number_input("Multiplier", 2, 10, 2, key="speed_mult"))
    time = int(st.number_input("Time (0=∞)", 0, 999, 0, key="speed_time"))
    usage = int(st.number_input("Usage (0=∞)", 0, 999, 0, key="speed_usage"))
    return {"multiplier": mult, "time": (time or None), "usage": (usage or None)}


def limit_params(prefix: str) -> dict[str, Any]:
    time = int(st.number_input("Time (0=∞)", 0, 999, 0, key=f"{prefix}_time"))
    usage = int(st.number_input("Usage (0=∞)", 0, 999, 0, key=f"{prefix}_usage"))
    return {"time": (time or None), "usage": (usage or None)}


# -----------------------
# Palette
# -----------------------

PALETTE: dict[str, ToolSpec] = {
    "floor": ToolSpec(
        label="Floor",
        icon="⬜",
        factory_fn=create_floor,
        param_map=lambda p: {"cost_amount": int(p.get("cost", 1))},
        param_ui=floor_params,
    ),
    "wall": ToolSpec(
        label="Wall",
        icon="🟫",
        factory_fn=create_wall,
        param_map=lambda p: {},
    ),
    "agent": ToolSpec(
        label="Agent",
        icon="😊",
        factory_fn=create_agent,
        param_map=lambda p: {"health": int(p.get("health", 5))},
        param_ui=agent_params,
    ),
    "exit": ToolSpec(
        label="Exit",
        icon="🏁",
        factory_fn=create_exit,
        param_map=lambda p: {},
    ),
    "key": ToolSpec(
        label="Key",
        icon="🔑",
        factory_fn=create_key,
        param_map=lambda p: {"key_id": p.get("key_id", "A")},
        param_ui=key_params,
    ),
    "door": ToolSpec(
        label="Door",
        icon="🚪",
        factory_fn=create_door,
        param_map=lambda p: {"key_id": p.get("key_id", "A")},
        param_ui=door_params,
    ),
    "coin": ToolSpec(
        label="Coin",
        icon="🪙",
        factory_fn=create_coin,
        param_map=lambda p: {"reward": p.get("reward")},
        param_ui=coin_params,
    ),
    "core": ToolSpec(
        label="Core",
        icon="⭐",
        factory_fn=create_core,
        param_map=lambda p: {
            "reward": p.get("reward"),
            "required": bool(p.get("required", True)),
        },
        param_ui=core_params,
    ),
    "portal": ToolSpec(
        label="Portal",
        icon="🔵",
        factory_fn=create_portal,  # standardized: (pair: BaseEntity | None)
        param_map=lambda p: {},
        description="Click two cells sequentially to pair.",
    ),
    "box": ToolSpec(
        label="Box",
        icon="📦",
        factory_fn=create_box,
        param_map=lambda p: {
            "pushable": bool(p.get("pushable", True)),
            "moving_direction": p.get("moving_direction"),
            "moving_on_collision": p.get("moving_on_collision", "bounce"),
            "moving_speed": int(p.get("moving_speed", 1)),
        },
        param_ui=box_params,
    ),
    "monster": ToolSpec(
        label="Monster",
        icon="👹",
        factory_fn=create_monster,
        param_map=lambda p: {
            "damage": int(p.get("damage", 3)),
            "lethal": bool(p.get("lethal", False)),
            "moving_direction": p.get("moving_direction"),
            "moving_on_collision": p.get("moving_on_collision", "bounce"),
            "moving_speed": int(p.get("moving_speed", 1)),
        },
        param_ui=monster_params,
    ),
    "spike": ToolSpec(
        label="Spike",
        icon="⚓",
        factory_fn=create_hazard,
        param_map=lambda p: {
            "appearance": "spike",
            "damage": int(p.get("damage", 2)),
            "lethal": bool(p.get("lethal", False)),
        },
        param_ui=lambda: hazard_params("spike"),
    ),
    "lava": ToolSpec(
        label="Lava",
        icon="🔥",
        factory_fn=create_hazard,
        param_map=lambda p: {
            "appearance": "lava",
            "damage": int(p.get("damage", 2)),
            "lethal": bool(p.get("lethal", True)),
        },
        param_ui=lambda: hazard_params("lava"),
    ),
    "speed": ToolSpec(
        label="Speed",
        icon="🥾",
        factory_fn=create_speed_effect,
        param_map=lambda p: {
            "multiplier": int(p.get("multiplier", 2)),
            "time": p.get("time"),
            "usage": p.get("usage"),
        },
        param_ui=speed_params,
    ),
    "shield": ToolSpec(
        label="Shield",
        icon="🛡️",
        factory_fn=create_immunity_effect,
        param_map=lambda p: {"time": p.get("time"), "usage": p.get("usage")},
        param_ui=lambda: limit_params("shield"),
    ),
    "ghost": ToolSpec(
        label="Ghost",
        icon="👻",
        factory_fn=create_phasing_effect,
        param_map=lambda p: {"time": p.get("time"), "usage": p.get("usage")},
        param_ui=lambda: limit_params("ghost"),
    ),
    "erase": ToolSpec(
        label="Eraser",
        icon="␡",
        factory_fn=create_floor,
        param_map=lambda p: {"cost_amount": 1},
        description="Reset cell to floor-only.",
    ),
}


# -----------------------
# Asset root resolver (preview)
# -----------------------


def _asset_root_resolver(image_map: ImageMap) -> str:
    return DEFAULT_ASSET_ROOT


# -----------------------
# Register LevelSource
# -----------------------

register_level_source(
    make_level_editor_source(
        name="Grid Universe Level Editor",
        palette=PALETTE,
        image_maps=list(IMAGE_MAP_REGISTRY.values()),  # offer all built-in GU packs
        env_factory=None,  # use default GridUniverseEnv factory
        move_fn_registry=MOVE_FN_REGISTRY,
        objective_fn_registry=OBJECTIVE_FN_REGISTRY,
        asset_root_resolver=_asset_root_resolver,
        env_class=GridUniverseEnv,
    )
)
