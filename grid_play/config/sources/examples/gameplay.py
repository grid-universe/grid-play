from __future__ import annotations

from typing import Callable

from grid_universe.state import State
from grid_universe.examples import gameplay_levels
from grid_universe.renderer.image import ImageMap
from grid_universe.env import GridUniverseEnv

from grid_play.config.sources.base import register_level_source
from grid_play.config.sources.level_selection import (
    Builder,
    make_level_selection_source,
)


BUILDERS: dict[str, Builder] = {
    "L0 Basic Movement": gameplay_levels.build_level_basic_movement,
    "L1 Maze Turns": gameplay_levels.build_level_maze_turns,
    "L2 Optional Coin Path": gameplay_levels.build_level_optional_coin,
    "L3 One Required Core": gameplay_levels.build_level_required_one,
    "L4 Two Required Cores": gameplay_levels.build_level_required_two,
    "L5 Key & Door": gameplay_levels.build_level_key_door,
    "L6 Hazard Detour": gameplay_levels.build_level_hazard_detour,
    "L7 Portal Shortcut": gameplay_levels.build_level_portal_shortcut,
    "L8 Pushable Box": gameplay_levels.build_level_pushable_box,
    "L9 Enemy Patrol": gameplay_levels.build_level_enemy_patrol,
    "L10 Shield Powerup": gameplay_levels.build_level_power_shield,
    "L11 Ghost Powerup": gameplay_levels.build_level_power_ghost,
    "L12 Boots Powerup": gameplay_levels.build_level_power_boots,
    "L13 Capstone": gameplay_levels.build_level_capstone,
}


def _env_factory(
    initial_state_fn: Callable[..., State], image_map: ImageMap
) -> GridUniverseEnv:
    sample = initial_state_fn()
    return GridUniverseEnv(
        render_mode="rgb_array",
        initial_state_fn=initial_state_fn,
        width=sample.width,
        height=sample.height,
        render_image_map=image_map,
    )


source = make_level_selection_source(
    name="Grid Universe Gameplay Example",
    builders=BUILDERS,
    builder_returns_gridstate=False,
    env_factory=_env_factory,
    image_maps=None,  # defaults to registry
)

register_level_source(source)
