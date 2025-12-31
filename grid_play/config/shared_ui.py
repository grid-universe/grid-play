from __future__ import annotations
from typing import Protocol

import streamlit as st
from grid_universe.renderer.texture import TEXTURE_MAP_REGISTRY, TextureMap


class HasTextureMap(Protocol):
    @property
    def render_texture_map(self) -> TextureMap: ...


def seed_section(*, key: str) -> int:
    st.subheader("Random seed")
    return st.number_input("Random seed", min_value=0, value=0, key=key)


def _labels_for_texture_maps(texture_maps: list[TextureMap]) -> list[str]:
    # Prefer registry names when the object identity matches; else provide fallback labels
    inverse: dict[int, str] = {id(v): k for k, v in TEXTURE_MAP_REGISTRY.items()}
    return [inverse.get(id(tm), f"Pack {i + 1}") for i, tm in enumerate(texture_maps)]


def texture_map_section(
    current: HasTextureMap,
    *,
    label: str = "Texture Map",
    key: str = "texture_map",
    options: list[TextureMap] | None = None,
) -> TextureMap:
    """
    Texture-map selector with optional candidate list.

    Behavior:
    - If options is None, fall back to all maps in TEXTURE_MAP_REGISTRY (values).
    - If len(options) <= 1, returns the sole map without rendering a picker.
    - If multiple, renders a selectbox with friendly labels (registry names if available).

    Returns the selected TextureMap (or the sole provided map).
    """
    texture_maps: list[TextureMap] = list(
        TEXTURE_MAP_REGISTRY.values() if options is None else options
    )
    if not texture_maps:
        # Defensive: no maps available; this should not happen under normal usage
        st.warning("No texture maps available; using default registry mapping.")
        texture_maps = list(TEXTURE_MAP_REGISTRY.values())

    if len(texture_maps) == 1:
        # Single choice: no UI; return directly
        return texture_maps[0]

    # Multiple choices: render selector
    st.subheader(label)
    labels = _labels_for_texture_maps(texture_maps)
    try:
        current_index = texture_maps.index(current.render_texture_map)
    except ValueError:
        current_index = 0

    chosen_label = st.selectbox(
        label,
        labels,
        index=current_index,
        key=key,
    )
    map_index = labels.index(chosen_label)
    return texture_maps[map_index]
