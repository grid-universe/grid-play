from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import streamlit as st

from grid_universe.state import State
from grid_universe.levels.grid import Level
from grid_universe.levels.convert import to_state as level_to_state
from grid_universe.gym_env import GridUniverseEnv
from grid_universe.renderer.texture import TextureMap, TEXTURE_MAP_REGISTRY

from .base import BaseConfig, LevelSource
from ..shared_ui import seed_section, texture_map_section

# Builders produce either immutable State or mutable Level
Builder = Callable[..., State] | Callable[..., Level]


@dataclass(frozen=True)
class LevelSelectionConfig(BaseConfig):
    level_name: str
    render_texture_map: TextureMap


def make_level_selection_source(
    *,
    name: str,
    builders: dict[str, Builder],
    builder_returns_level: bool,
    env_factory: Callable[[Callable[..., State], TextureMap], GridUniverseEnv],
    # If None, defaults to the registry values. If a single map is supplied, the picker is hidden.
    texture_maps: list[TextureMap] | None = None,
) -> LevelSource:
    """
    Construct a LevelSource for a set of named builders.
    """
    level_names: list[str] = list(builders.keys())
    offered_maps: list[TextureMap] = list(
        TEXTURE_MAP_REGISTRY.values() if texture_maps is None else texture_maps
    )
    if not offered_maps:
        offered_maps = list(TEXTURE_MAP_REGISTRY.values())

    def _default_config() -> LevelSelectionConfig:
        return LevelSelectionConfig(
            level_name=level_names[0],
            seed=0,
            render_texture_map=offered_maps[0],
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

        # Texture map (shared UI; single option -> no picker)
        texture_map = texture_map_section(
            base,
            label="Texture Map",
            key=f"{name}_texture_map",
            options=offered_maps,
        )

        return LevelSelectionConfig(
            level_name=level_name,
            seed=seed,
            render_texture_map=texture_map,
        )

    def _make_env(cfg: LevelSelectionConfig) -> GridUniverseEnv:
        level_name: str = cfg.level_name
        builder = builders.get(level_name)
        if builder is None:
            raise ValueError(f"Unknown level name: {level_name}")

        def _initial_state_fn(**_ignored: object) -> State:
            seed_val = cfg.seed if cfg.seed is not None else 0
            result = builder(seed_val)
            if builder_returns_level:
                return level_to_state(cast(Level, result))
            return cast(State, result)

        return env_factory(_initial_state_fn, cfg.render_texture_map)

    return LevelSource(
        name=name,
        config_type=LevelSelectionConfig,
        initial_config=_default_config,
        build_config=_build_config,
        make_env=_make_env,
    )
