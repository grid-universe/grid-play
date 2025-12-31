from __future__ import annotations

from typing import Any

import streamlit as st

from grid_universe.components.properties import MovingAxis
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
from grid_play.config.sources.base import register_level_source
from grid_play.config.sources.level_editor import ToolSpec, make_level_editor_source


# ---------- Parameter UIs ----------


def agent_params() -> dict[str, Any]:
    return {"health": int(st.number_input("Health", 1, 99, 5, key="agent_health"))}


def floor_params() -> dict[str, Any]:
    return {"cost": int(st.number_input("Move Cost", 1, 99, 1, key="floor_cost"))}


def coin_params() -> dict[str, Any]:
    reward = int(st.number_input("Reward (0 = none)", 0, 999, 0, key="coin_reward"))
    return {"reward": reward if reward > 0 else None}


def core_params() -> dict[str, Any]:
    reward = int(st.number_input("Reward (0 = none)", 0, 999, 10, key="core_reward"))
    required = bool(st.checkbox("Required?", value=True, key="core_required"))
    return {"reward": reward if reward > 0 else None, "required": required}


def key_params() -> dict[str, Any]:
    key_id = st.text_input("Key ID", value="A", key="key_id").strip() or "A"
    return {"key_id": key_id}


def door_params() -> dict[str, Any]:
    key_id = st.text_input("Door Key ID", value="A", key="door_key_id").strip() or "A"
    return {"key_id": key_id}


def moving_params(prefix: str) -> dict[str, Any]:
    axis_label = st.selectbox(
        "Axis", ["None", "Horizontal", "Vertical"], key=f"{prefix}_move_axis"
    )
    axis = None
    if axis_label == "Horizontal":
        axis = MovingAxis.HORIZONTAL
    elif axis_label == "Vertical":
        axis = MovingAxis.VERTICAL
    direction = st.selectbox(
        "Direction",
        ["+1 (forward/right/down)", "-1 (back/left/up)"],
        key=f"{prefix}_move_dir",
    )
    dir_val = 1 if direction.startswith("+1") else -1
    bounce = bool(
        st.checkbox("Bounce (reverse at ends)", value=True, key=f"{prefix}_move_bounce")
    )
    speed = int(
        st.number_input("Speed (tiles / step)", 1, 10, 1, key=f"{prefix}_move_speed")
    )
    return {
        "moving_axis": axis,
        "moving_direction": dir_val if axis is not None else None,
        "moving_bounce": bounce,
        "moving_speed": speed,
    }


def box_params() -> dict[str, Any]:
    pushable = bool(st.checkbox("Pushable?", value=True, key="box_pushable"))
    return {"pushable": pushable, **moving_params("box")}


def monster_params() -> dict[str, Any]:
    damage = int(st.number_input("Damage", 1, 50, 3, key="monster_dmg"))
    lethal = bool(st.checkbox("Lethal?", value=False, key="monster_lethal"))
    return {"damage": damage, "lethal": lethal, **moving_params("monster")}


def hazard_params(kind: str) -> dict[str, Any]:
    default_lethal = kind == "lava"
    lethal = bool(st.checkbox("Lethal?", value=default_lethal, key=f"{kind}_lethal"))
    damage = (
        0 if lethal else int(st.number_input("Damage", 1, 50, 2, key=f"{kind}_damage"))
    )
    return {"damage": damage, "lethal": lethal}


def speed_params() -> dict[str, Any]:
    mult = int(st.number_input("Multiplier", 2, 10, 2, key="speed_mult"))
    time = int(st.number_input("Time (0=∞)", 0, 999, 0, key="speed_time"))
    usage = int(st.number_input("Usage (0=∞)", 0, 999, 0, key="speed_usage"))
    return {"multiplier": mult, "time": (time or None), "usage": (usage or None)}


def limit_params(prefix: str) -> dict[str, Any]:
    time = int(st.number_input("Time (0=∞)", 0, 999, 0, key=f"{prefix}_time"))
    usage = int(st.number_input("Usage (0=∞)", 0, 999, 0, key=f"{prefix}_usage"))
    return {"time": (time or None), "usage": (usage or None)}


# ---------- Palette ----------

PALETTE: dict[str, ToolSpec] = {
    "floor": ToolSpec(
        label="Floor",
        icon="⬜",
        builder=lambda p: create_floor(cost_amount=int(p.get("cost", 1))),
        param_ui=floor_params,
    ),
    "wall": ToolSpec(
        label="Wall",
        icon="🟫",
        builder=lambda _p: create_wall(),
    ),
    "agent": ToolSpec(
        label="Agent",
        icon="😊",
        builder=lambda p: create_agent(health=int(p.get("health", 5))),
        param_ui=agent_params,
    ),
    "exit": ToolSpec(
        label="Exit",
        icon="🏁",
        builder=lambda _p: create_exit(),
    ),
    "coin": ToolSpec(
        label="Coin",
        icon="🪙",
        builder=lambda p: create_coin(reward=p.get("reward")),
        param_ui=coin_params,
    ),
    "core": ToolSpec(
        label="Core",
        icon="⭐",
        builder=lambda p: create_core(
            reward=p.get("reward"),
            required=bool(p.get("required", True)),
        ),
        param_ui=core_params,
    ),
    "key": ToolSpec(
        label="Key",
        icon="🔑",
        builder=lambda p: create_key(p.get("key_id", "A")),
        param_ui=key_params,
    ),
    "door": ToolSpec(
        label="Door",
        icon="🚪",
        builder=lambda p: create_door(p.get("key_id", "A")),
        param_ui=door_params,
    ),
    "portal": ToolSpec(
        label="Portal",
        icon="🔵",
        builder=lambda _p: create_portal(),
        description="Click two cells sequentially to pair.",
    ),
    "box": ToolSpec(
        label="Box",
        icon="📦",
        builder=lambda p: create_box(
            pushable=bool(p.get("pushable", True)),
            moving_axis=p.get("moving_axis"),
            moving_direction=p.get("moving_direction"),
            moving_bounce=bool(p.get("moving_bounce", True)),
            moving_speed=int(p.get("moving_speed", 1)),
        ),
        param_ui=box_params,
    ),
    "monster": ToolSpec(
        label="Monster",
        icon="👹",
        builder=lambda p: create_monster(
            damage=int(p.get("damage", 3)),
            lethal=bool(p.get("lethal", False)),
            moving_axis=p.get("moving_axis"),
            moving_direction=p.get("moving_direction"),
            moving_bounce=bool(p.get("moving_bounce", True)),
            moving_speed=int(p.get("moving_speed", 1)),
        ),
        param_ui=monster_params,
    ),
    "spike": ToolSpec(
        label="Spike",
        icon="⚓",
        builder=lambda p: create_hazard(
            "spike",
            int(p.get("damage", 2)),
            bool(p.get("lethal", False)),
        ),
        param_ui=lambda: hazard_params("spike"),
    ),
    "lava": ToolSpec(
        label="Lava",
        icon="🔥",
        builder=lambda p: create_hazard(
            "lava",
            int(p.get("damage", 2)),
            bool(p.get("lethal", True)),
        ),
        param_ui=lambda: hazard_params("lava"),
    ),
    "speed": ToolSpec(
        label="Speed",
        icon="🥾",
        builder=lambda p: create_speed_effect(
            multiplier=int(p.get("multiplier", 2)),
            time=p.get("time"),
            usage=p.get("usage"),
        ),
        param_ui=speed_params,
    ),
    "shield": ToolSpec(
        label="Shield",
        icon="🛡️",
        builder=lambda p: create_immunity_effect(
            time=p.get("time"),
            usage=p.get("usage"),
        ),
        param_ui=lambda: limit_params("shield"),
    ),
    "ghost": ToolSpec(
        label="Ghost",
        icon="👻",
        builder=lambda p: create_phasing_effect(
            time=p.get("time"),
            usage=p.get("usage"),
        ),
        param_ui=lambda: limit_params("ghost"),
    ),
    # Eraser handled specially by the editor engine (place_tool); builder not used
    "erase": ToolSpec(
        label="Eraser",
        icon="␡",
        builder=lambda _p: create_floor(cost_amount=1),
        description="Reset cell to floor-only.",
    ),
}

# ---------- Register LevelSource ----------

register_level_source(
    make_level_editor_source(
        name="Grid Universe Level Editor Example",
        palette=PALETTE,
    )
)
