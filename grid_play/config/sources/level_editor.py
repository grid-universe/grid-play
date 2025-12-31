from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast

import streamlit as st

from grid_universe.gym_env import GridUniverseEnv
from grid_universe.levels.convert import to_state
from grid_universe.levels.entity import BaseEntity
from grid_universe.levels.grid import Level
from grid_universe.moves import MOVE_FN_REGISTRY, default_move_fn
from grid_universe.objectives import OBJECTIVE_FN_REGISTRY, default_objective_fn
from grid_universe.renderer.texture import (
    TextureMap,
    DEFAULT_TEXTURE_MAP,
    TextureRenderer,
    TEXTURE_MAP_REGISTRY,
    DEFAULT_ASSET_ROOT,
)
from grid_universe.state import State
from grid_universe.types import MoveFn, ObjectiveFn

from grid_play.config.sources.base import BaseConfig, LevelSource
from grid_play.config.shared_ui import texture_map_section


# ------------ Editor model ------------

EditorParams = dict[str, Any]


@dataclass(frozen=True)
class EditorToken:
    type: str
    params: EditorParams


WorkingCell = list[EditorToken]
WorkingRow = list[WorkingCell]
WorkingGrid = list[WorkingRow]

# Immutable snapshot representation: grid[y][x] -> tuple[EditorToken, ...]
TokenGridSnapshot = tuple[tuple[tuple[EditorToken, ...], ...], ...]


@dataclass(frozen=True)
class EditorConfig(BaseConfig):
    width: int
    height: int
    turn_limit: int | None
    move_fn: MoveFn
    objective_fn: ObjectiveFn
    render_texture_map: TextureMap
    grid_tokens: TokenGridSnapshot  # immutable snapshot of the authored grid


ParamUI = Callable[[], dict[str, Any]]
BuilderFn = Callable[[dict[str, Any]], BaseEntity]


@dataclass(frozen=True)
class ToolSpec:
    """Immutable tool specification for the palette."""

    label: str
    icon: str
    builder: BuilderFn
    param_ui: ParamUI | None = None
    description: str | None = None
    multi_place: bool = False  # reserved for future extension


# ------------ Defaults & helpers ------------


def _default_editor_config() -> EditorConfig:
    width, height = 9, 7
    # Initialize with floor-only tokens
    base_row: tuple[tuple[EditorToken, ...], ...] = tuple(
        (EditorToken(type="floor", params={"cost": 1}),) for _ in range(width)
    )
    snapshot: TokenGridSnapshot = tuple(base_row for _ in range(height))
    return EditorConfig(
        width=width,
        height=height,
        turn_limit=None,
        move_fn=default_move_fn,
        objective_fn=default_objective_fn,
        seed=None,
        render_texture_map=DEFAULT_TEXTURE_MAP,
        grid_tokens=snapshot,
    )


def _ensure_working_grid(width: int, height: int) -> WorkingGrid:
    """Create or resize the working grid in session state (floor-only baseline)."""
    key = "editor_working_grid"
    if key not in st.session_state:
        st.session_state[key] = [
            [[EditorToken(type="floor", params={"cost": 1})] for _ in range(width)]
            for _ in range(height)
        ]
    grid = cast(WorkingGrid, st.session_state[key])

    # Resize preserving content
    if len(grid) != height or len(grid[0]) != width:
        new_grid: WorkingGrid = [
            [[EditorToken(type="floor", params={"cost": 1})] for _ in range(width)]
            for _ in range(height)
        ]
        for yy in range(min(height, len(grid))):
            for xx in range(min(width, len(grid[0]))):
                # Deep copy tokens (immutable dataclass, but we copy list container)
                new_grid[yy][xx] = list(grid[yy][xx])
        st.session_state[key] = new_grid
        grid = new_grid
    return grid


def _ensure_floor(cell: WorkingCell) -> EditorToken:
    """Ensure the cell has a floor token; return that token."""
    for t in cell:
        if t.type == "floor":
            return t
    floor = EditorToken(type="floor", params={"cost": 1})
    cell.insert(0, floor)
    return floor


def _tool_icon(palette: Mapping[str, ToolSpec], key: str, default: str = "") -> str:
    spec = palette.get(key)
    return spec.icon if spec is not None else default


def place_tool(
    tool_key: str,
    x: int,
    y: int,
    grid: WorkingGrid,
    params: EditorParams | None = None,
) -> None:
    """Place the selected tool into the grid cell."""
    cell = grid[y][x]
    p = params or {}

    if tool_key == "erase":
        # Reset to floor-only (preserve existing floor if present)
        floor = _ensure_floor(cell)
        grid[y][x] = [floor]
        return

    if tool_key == "floor":
        floor = _ensure_floor(cell)
        # Update cost parameter (if provided)
        if "cost" in p:
            new_cost = int(p["cost"])
            # replace the floor token with updated params
            updated = EditorToken(type="floor", params={"cost": new_cost})
            # maintain floor at index 0
            grid[y][x] = [updated] + [t for t in cell if t.type != "floor"]
        else:
            # Keep floor and any non-floor tokens
            grid[y][x] = [floor] + [t for t in cell if t.type != "floor"]
        return

    # Non-floor: ensure floor then set to [floor, tool]
    floor = _ensure_floor(cell)
    tool = EditorToken(type=tool_key, params=dict(p))
    grid[y][x] = [floor, tool]


def _pair_portals(grid: WorkingGrid) -> None:
    """Compute portal pairs sequentially and store in session state."""
    portals: list[tuple[int, int]] = []
    for yy, row in enumerate(grid):
        for xx, cell in enumerate(row):
            if any(t.type == "portal" for t in cell):
                portals.append((xx, yy))
    st.session_state["editor_portal_pairs"] = [
        (portals[i], portals[i + 1]) for i in range(0, len(portals) - 1, 2)
    ]


def snapshot_grid(grid: WorkingGrid) -> TokenGridSnapshot:
    """Produce an immutable snapshot of the working grid."""
    return tuple(
        tuple(
            tuple(EditorToken(type=t.type, params=dict(t.params)) for t in cell)
            for cell in row
        )
        for row in grid
    )


# ------------ Level build from tokens ------------


def build_level_from_tokens(
    cfg: EditorConfig,
    palette: Mapping[str, ToolSpec],
) -> Level:
    """Convert an EditorConfig grid snapshot into a mutable Level."""
    level = Level(
        width=cfg.width,
        height=cfg.height,
        move_fn=cfg.move_fn,
        objective_fn=cfg.objective_fn,
        seed=cfg.seed,
        turn_limit=cfg.turn_limit,
    )

    # Track portals to wire pairs after placement
    portal_specs: dict[tuple[int, int], BaseEntity] = {}

    for y in range(cfg.height):
        for x in range(cfg.width):
            for token in cfg.grid_tokens[y][x]:
                ttype = token.type
                if ttype == "erase":
                    continue
                tspec = palette.get(ttype)
                if tspec is None:
                    continue
                try:
                    ent = tspec.builder(token.params)
                except Exception:
                    # Defensive fallback: try with empty params
                    ent = tspec.builder({})
                level.add((x, y), ent)
                if ttype == "portal":
                    portal_specs[(x, y)] = ent

    # Pair portals using stored pairs else sequential
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if "editor_portal_pairs" in st.session_state:
        pairs = cast(
            list[tuple[tuple[int, int], tuple[int, int]]],
            st.session_state["editor_portal_pairs"],
        )
    else:
        ordered = list(portal_specs.keys())
        pairs = [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)]

    for a_pos, b_pos in pairs:
        a = portal_specs.get(a_pos)
        b = portal_specs.get(b_pos)
        if a is None or b is None or a is b:
            continue
        # Mirror factory pairing semantics if attribute exists
        try:
            setattr(a, "portal_pair_ref", b)
            if getattr(b, "portal_pair_ref", None) is None:
                setattr(b, "portal_pair_ref", a)
        except Exception:
            pass

    return level


# ------------ Streamlit UI builder ------------


def build_editor_config(
    current: object,
    *,
    name: str,
    palette: Mapping[str, ToolSpec],
    texture_maps: list[TextureMap] | None = None,
    move_fn_registry: Mapping[str, MoveFn] | None = None,
    objective_fn_registry: Mapping[str, ObjectiveFn] | None = None,
    asset_root_resolver: Callable[[TextureMap], str] | None = None,
) -> EditorConfig:
    """Render the editor UI and return an updated EditorConfig.

    Args:
        name: Display name for namespacing widget keys.
        palette: ToolSpec mapping.
        texture_maps: Optional list of texture maps to offer (defaults to registry).
        move_fn_registry: Optional registry of move functions (defaults to MOVE_FN_REGISTRY).
        objective_fn_registry: Optional registry of objective functions (defaults to OBJECTIVE_FN_REGISTRY).

    Returns:
        An EditorConfig reflecting the current UI state.
    """
    base: EditorConfig = (
        current if isinstance(current, EditorConfig) else _default_editor_config()
    )
    st.info(name, icon="🛠️")

    # Resolve registries (configurable)
    move_fns: Mapping[str, MoveFn] = move_fn_registry or MOVE_FN_REGISTRY
    objectives: Mapping[str, ObjectiveFn] = (
        objective_fn_registry or OBJECTIVE_FN_REGISTRY
    )

    # Top config row
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        width = int(st.number_input("Width", 3, 50, base.width, key=f"{name}_width"))
    with c2:
        height = int(
            st.number_input("Height", 3, 50, base.height, key=f"{name}_height")
        )
    with c3:
        seed_val = int(base.seed or 0)
        seed = int(st.number_input("Seed", 0, None, seed_val, key=f"{name}_seed"))
    with c4:
        tl_in = int(
            st.number_input(
                "Turn limit (0=∞)",
                0,
                9999,
                int(base.turn_limit or 0),
                key=f"{name}_turn_limit",
            )
        )
        turn_limit = tl_in if tl_in > 0 else None

    # Movement and objective selection (use configurable registries)
    move_names = list(move_fns.keys())
    obj_names = list(objectives.keys())
    mc1, mc2 = st.columns([1, 1])
    with mc1:
        # find current key for base.move_fn
        try:
            current_move_key = next(k for k, v in move_fns.items() if v is base.move_fn)
        except StopIteration:
            current_move_key = move_names[0]
        move_label = st.selectbox(
            "Movement rule",
            move_names,
            index=move_names.index(current_move_key),
            key=f"{name}_move_fn",
        )
        move_fn = move_fns[move_label]
    with mc2:
        try:
            current_obj_key = next(
                k for k, v in objectives.items() if v is base.objective_fn
            )
        except StopIteration:
            current_obj_key = obj_names[0]
        obj_label = st.selectbox(
            "Objective",
            obj_names,
            index=obj_names.index(current_obj_key),
            key=f"{name}_objective_fn",
        )
        objective_fn = objectives[obj_label]

    # Texture map selection: mirror level_selection behavior
    offered_maps: list[TextureMap] = (
        list(TEXTURE_MAP_REGISTRY.values())
        if texture_maps is None
        else list(texture_maps)
    )
    if not offered_maps:
        offered_maps = list(TEXTURE_MAP_REGISTRY.values())
    texture_map = texture_map_section(
        base, key=f"{name}_texture_map", options=offered_maps
    )

    # Working grid and palette UI
    grid = _ensure_working_grid(width, height)
    palette_keys: list[str] = list(palette.keys())
    palette_labels: list[str] = [
        f"{palette[k].icon} {palette[k].label}".strip() for k in palette_keys
    ]

    left, middle, right = st.columns([1, 2, 2])

    # Palette selection + params
    with left:
        st.subheader("Palette")
        idx = st.radio(
            "Tool",
            options=list(range(len(palette_keys))),
            format_func=lambda i: palette_labels[int(i)],
            key=f"{name}_palette",
        )
        tool_key = palette_keys[idx]
        tspec = palette[tool_key]
        params: EditorParams = {}
        if tspec.param_ui is not None:
            st.markdown("Parameters")
            try:
                params = tspec.param_ui() or {}
            except Exception:
                params = {}
        if tspec.description:
            st.caption(tspec.description)

    # Grid editing
    with middle:
        st.subheader("Grid")
        # Determine default floor icon safely
        floor_icon = _tool_icon(palette, "floor", "⬜")
        for yy in range(height):
            cols = st.columns(width)
            for xx in range(width):
                cell = grid[yy][xx]
                icons = "".join(
                    _tool_icon(palette, t.type) if t.type != "floor" else ""
                    for t in cell
                )
                label = icons if icons else floor_icon
                if cols[xx].button(label, key=f"{name}_cell_{xx}_{yy}"):
                    place_tool(tool_key, xx, yy, grid, params)
                    if tool_key == "portal":
                        _pair_portals(grid)
                    st.rerun()

    # Resolve asset root for preview based on selected texture map
    preview_asset_root = (
        asset_root_resolver(texture_map)
        if asset_root_resolver is not None
        else DEFAULT_ASSET_ROOT
    )

    # Live preview
    with right:
        st.subheader("Preview")
        cfg_preview = EditorConfig(
            width=width,
            height=height,
            turn_limit=turn_limit,
            move_fn=move_fn,
            objective_fn=objective_fn,
            seed=seed,
            render_texture_map=texture_map,
            grid_tokens=snapshot_grid(grid),
        )
        try:
            lvl = build_level_from_tokens(cfg_preview, palette)
            state_preview = to_state(lvl)
            img = TextureRenderer(
                texture_map=texture_map,
                asset_root=preview_asset_root,
            ).render(state_preview)
            st.image(img, width="stretch")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    # Final immutable snapshot
    return EditorConfig(
        width=width,
        height=height,
        turn_limit=turn_limit,
        move_fn=move_fn,
        objective_fn=objective_fn,
        seed=seed,
        render_texture_map=texture_map,
        grid_tokens=snapshot_grid(grid),
    )


# ------------ Environment factory ------------


def make_env(
    cfg: EditorConfig,
    palette: Mapping[str, ToolSpec],
    env_factory: EnvFactory | None = None,
) -> GridUniverseEnv:
    """Create a GridUniverseEnv for the current editor config."""

    def _initial_state_fn(**_: Any) -> State:
        level = build_level_from_tokens(cfg, palette)
        return to_state(level)

    state0 = _initial_state_fn()
    # Defensive: level must contain at least one agent
    if not state0.agent:
        raise ValueError(
            "Level must contain an Agent. Use the 'Agent' tool in the palette to place one before starting."
        )

    if env_factory is not None:
        return env_factory(_initial_state_fn, cfg.render_texture_map)

    return GridUniverseEnv(
        render_mode="rgb_array",
        initial_state_fn=_initial_state_fn,
        width=state0.width,
        height=state0.height,
        render_texture_map=cfg.render_texture_map,
    )


# ------------ LevelSource factory ------------

# (initial_state_fn, texture_map) -> Env instance
EnvFactory = Callable[[Callable[..., State], TextureMap], GridUniverseEnv]


def make_level_editor_source(
    *,
    name: str,
    palette: Mapping[str, ToolSpec],
    texture_maps: list[TextureMap] | None = None,
    env_factory: EnvFactory | None = None,
    move_fn_registry: Mapping[str, MoveFn] | None = None,
    objective_fn_registry: Mapping[str, ObjectiveFn] | None = None,
    asset_root_resolver: Callable[[TextureMap], str] | None = None,
) -> LevelSource:
    """Create a LevelSource for a level editor with the given parameters.

    Args:
        name: Display name in the UI.
        palette: Mapping of tool_key -> ToolSpec.
        texture_maps: Optional list of texture maps to offer; defaults to registry values.
            If a single map is provided, the picker is hidden and that map is used.
        env_factory: Optional factory to build a custom environment (e.g., GridAdventureEnv).
            Signature: env_factory(initial_state_fn, texture_map) -> Env.
            If omitted, a default GridUniverseEnv will be used.
        move_fn_registry: Optional custom registry of move functions for the editor's dropdown.
        objective_fn_registry: Optional custom registry of objective functions for the editor's dropdown.
        asset_root_resolver: Optional function to resolve asset root paths based on texture map.

    Returns:
        A LevelSource instance for the level editor.
    """

    def _initial_config() -> EditorConfig:
        return _default_editor_config()

    def _build_config(current: object) -> EditorConfig:
        return build_editor_config(
            current,
            name=name,
            palette=palette,
            texture_maps=texture_maps,
            move_fn_registry=move_fn_registry,
            objective_fn_registry=objective_fn_registry,
            asset_root_resolver=asset_root_resolver,
        )

    def _make_env(cfg: BaseConfig) -> GridUniverseEnv:
        assert isinstance(cfg, EditorConfig)
        return make_env(cfg, palette, env_factory)

    return LevelSource(
        name=name,
        config_type=EditorConfig,
        initial_config=_initial_config,
        build_config=_build_config,
        make_env=_make_env,
    )
