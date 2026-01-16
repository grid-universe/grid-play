from __future__ import annotations
from typing import Protocol

import streamlit as st
from grid_universe.renderer.image import IMAGE_MAP_REGISTRY, ImageMap


class HasImageMap(Protocol):
    @property
    def render_image_map(self) -> ImageMap: ...


def seed_section(*, key: str) -> int:
    st.subheader("Random seed")
    return st.number_input("Random seed", min_value=0, value=0, key=key)


def _labels_for_image_maps(image_maps: list[ImageMap]) -> list[str]:
    # Prefer registry names when the object identity matches; else provide fallback labels
    inverse: dict[int, str] = {id(v): k for k, v in IMAGE_MAP_REGISTRY.items()}
    return [inverse.get(id(tm), f"Pack {i + 1}") for i, tm in enumerate(image_maps)]


def image_map_section(
    current: HasImageMap,
    *,
    label: str = "Image Map",
    key: str = "image_map",
    options: list[ImageMap] | None = None,
) -> ImageMap:
    """
    Image-map selector with optional candidate list.

    Behavior:
    - If options is None, fall back to all maps in IMAGE_MAP_REGISTRY (values).
    - If len(options) <= 1, returns the sole map without rendering a picker.
    - If multiple, renders a selectbox with friendly labels (registry names if available).

    Returns the selected ImageMap (or the sole provided map).
    """
    image_maps: list[ImageMap] = list(
        IMAGE_MAP_REGISTRY.values() if options is None else options
    )
    if not image_maps:
        # Defensive: no maps available; this should not happen under normal usage
        st.warning("No image maps available; using default registry mapping.")
        image_maps = list(IMAGE_MAP_REGISTRY.values())

    if len(image_maps) == 1:
        # Single choice: no UI; return directly
        return image_maps[0]

    # Multiple choices: render selector
    st.subheader(label)
    labels = _labels_for_image_maps(image_maps)
    try:
        current_index = image_maps.index(current.render_image_map)
    except ValueError:
        current_index = 0

    chosen_label = st.selectbox(
        label,
        labels,
        index=current_index,
        key=key,
    )
    map_index = labels.index(chosen_label)
    return image_maps[map_index]
