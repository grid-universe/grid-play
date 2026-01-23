from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import streamlit as st

from grid_universe.state import State
from grid_universe.grid.gridstate import GridState
from grid_universe.grid.convert import to_state as gridstate_to_state
from grid_universe.env import GridUniverseEnv
from grid_universe.renderer.image import ImageMap, IMAGE_MAP_REGISTRY

from .base import BaseConfig, LevelSource
from ..shared_ui import seed_section, image_map_section

# Builders produce either immutable State or mutable GridState
Builder = Callable[..., State] | Callable[..., GridState]


@dataclass(frozen=True)
class LevelSelectionConfig(BaseConfig):
    level_name: str
    render_image_map: ImageMap


def make_level_selection_source(
    *,
    name: str,
    builders: dict[str, Builder],
    builder_returns_gridstate: bool,
    env_factory: Callable[[Callable[..., State], ImageMap], GridUniverseEnv],
    # If None, defaults to the registry values. If a single map is supplied, the picker is hidden.
    image_maps: list[ImageMap] | None = None,
) -> LevelSource:
    """
    Construct a LevelSource for a set of named builders.
    """
    level_names: list[str] = list(builders.keys())
    offered_maps: list[ImageMap] = list(
        IMAGE_MAP_REGISTRY.values() if image_maps is None else image_maps
    )
    if not offered_maps:
        offered_maps = list(IMAGE_MAP_REGISTRY.values())

    def _default_config() -> LevelSelectionConfig:
        return LevelSelectionConfig(
            level_name=level_names[0],
            seed=0,
            render_image_map=offered_maps[0],
        )

    def _build_config(current: object) -> LevelSelectionConfig:
        st.info(name, icon="🎮")
        base = (
            current if isinstance(current, LevelSelectionConfig) else _default_config()
        )

        # Level selection
        base_level_name = getattr(base, "level_name", level_names[0])
        level_name = st.selectbox(
            "Level",
            level_names,
            index=level_names.index(base_level_name)
            if base_level_name in level_names
            else 0,
            key=f"{name}_level_select",
        )

        # Seed (shared UI)
        seed = seed_section(key=f"{name}_seed")

        # Image map (shared UI; single option -> no picker)
        image_map = image_map_section(
            base,
            label="Image Map",
            key=f"{name}_image_map",
            options=offered_maps,
        )

        return LevelSelectionConfig(
            level_name=level_name,
            seed=seed,
            render_image_map=image_map,
        )

    def _make_env(cfg: LevelSelectionConfig) -> GridUniverseEnv:
        level_name: str = cfg.level_name
        builder = builders.get(level_name)
        if builder is None:
            raise ValueError(f"Unknown level name: {level_name}")

        def _initial_state_fn(**_ignored: object) -> State:
            seed_val = cfg.seed if cfg.seed is not None else 0
            result = builder(seed_val)
            if builder_returns_gridstate:
                return gridstate_to_state(cast(GridState, result))
            return cast(State, result)

        return env_factory(_initial_state_fn, cfg.render_image_map)

    return LevelSource(
        name=name,
        config_type=LevelSelectionConfig,
        initial_config=_default_config,
        build_config=_build_config,
        make_env=_make_env,
    )
