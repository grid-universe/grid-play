from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from inspect import getmodule
from typing import Any, Callable, Mapping

import streamlit as st

from grid_universe.gym_env import GridUniverseEnv
from grid_universe.levels.convert import to_state
from grid_universe.levels.entity import BaseEntity
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.types import MoveFn, ObjectiveFn
from grid_universe.renderer.texture import TextureMap, DEFAULT_TEXTURE_MAP

from grid_play.config.sources.base import BaseConfig, LevelSource
from grid_play.config.shared_ui import texture_map_section


# -------- Editor models --------

EditorParams = dict[str, Any]


@dataclass(frozen=True)
class EditorToken:
    type: str
    params: EditorParams


WorkingCell = list[EditorToken]
WorkingRow = list[WorkingCell]
WorkingGrid = list[WorkingRow]
TokenGridSnapshot = tuple[tuple[tuple[EditorToken, ...], ...], ...]


@dataclass(frozen=True)
class EditorConfig(BaseConfig):
    width: int
    height: int
    turn_limit: int | None
    move_fn: MoveFn
    objective_fn: ObjectiveFn
    render_texture_map: TextureMap
    grid_tokens: TokenGridSnapshot


ParamUI = Callable[[], EditorParams]


@dataclass(frozen=True)
class ToolSpec:
    label: str
    icon: str
    factory_fn: Callable[..., BaseEntity]
    param_map: Callable[[dict[str, Any]], dict[str, Any]]
    param_ui: ParamUI | None = None
    description: str | None = None


def _default_editor_config() -> EditorConfig:
    width, height = 9, 7
    base_row: tuple[tuple[EditorToken, ...], ...] = tuple(
        (EditorToken(type="floor", params={"cost": 1}),) for _ in range(width)
    )
    snapshot: TokenGridSnapshot = tuple(base_row for _ in range(height))
    # Placeholders; caller must provide registries and texture maps via build_editor_config
    return EditorConfig(
        width=width,
        height=height,
        turn_limit=None,
        move_fn=lambda s, e, a: [],
        objective_fn=lambda s, e: False,
        seed=None,
        render_texture_map=DEFAULT_TEXTURE_MAP,
        grid_tokens=snapshot,
    )


def _ensure_working_grid(width: int, height: int) -> WorkingGrid:
    key = "editor_working_grid"
    if key not in st.session_state:
        st.session_state[key] = [
            [[EditorToken(type="floor", params={"cost": 1})] for _ in range(width)]
            for _ in range(height)
        ]
    grid: WorkingGrid = st.session_state[key]
    if len(grid) != height or len(grid[0]) != width:
        new_grid: WorkingGrid = [
            [[EditorToken(type="floor", params={"cost": 1})] for _ in range(width)]
            for _ in range(height)
        ]
        for yy in range(min(height, len(grid))):
            for xx in range(min(width, len(grid[0]))):
                new_grid[yy][xx] = list(grid[yy][xx])
        st.session_state[key] = new_grid
        grid = new_grid
    return grid


def _ensure_floor(cell: WorkingCell) -> EditorToken:
    for t in cell:
        if t.type == "floor":
            return t
    floor = EditorToken(type="floor", params={"cost": 1})
    cell.insert(0, floor)
    return floor


def place_tool(
    tool_key: str,
    x: int,
    y: int,
    grid: WorkingGrid,
    params: EditorParams | None = None,
) -> None:
    cell = grid[y][x]
    p = params or {}

    if tool_key == "erase":
        floor = _ensure_floor(cell)
        grid[y][x] = [floor]
        return

    if tool_key == "floor":
        floor = _ensure_floor(cell)
        if "cost" in p:
            new_cost = int(p["cost"])
            updated = EditorToken(type="floor", params={"cost": new_cost})
            grid[y][x] = [updated] + [t for t in cell if t.type != "floor"]
        else:
            grid[y][x] = [floor] + [t for t in cell if t.type != "floor"]
        return

    floor = _ensure_floor(cell)
    tool = EditorToken(type=tool_key, params=dict(p))
    grid[y][x] = [floor, tool]


def _pair_portals(grid: WorkingGrid) -> None:
    portals: list[tuple[int, int]] = []
    for yy, row in enumerate(grid):
        for xx, cell in enumerate(row):
            if any(t.type == "portal" for t in cell):
                portals.append((xx, yy))
    st.session_state["editor_portal_pairs"] = [
        (portals[i], portals[i + 1]) for i in range(0, len(portals) - 1, 2)
    ]


def snapshot_grid(grid: WorkingGrid) -> TokenGridSnapshot:
    return tuple(
        tuple(
            tuple(EditorToken(type=t.type, params=dict(t.params)) for t in cell)
            for cell in row
        )
        for row in grid
    )


def build_level_from_tokens(
    cfg: EditorConfig, palette: Mapping[str, ToolSpec]
) -> Level:
    level = Level(
        width=cfg.width,
        height=cfg.height,
        move_fn=cfg.move_fn,
        objective_fn=cfg.objective_fn,
        seed=cfg.seed,
        turn_limit=cfg.turn_limit,
    )
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
                kwargs = tspec.param_map(token.params)
                ent = tspec.factory_fn(**kwargs)
                level.add((x, y), ent)
                if ttype == "portal":
                    portal_specs[(x, y)] = ent

    # Pair portals using stored pairs else sequential
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if "editor_portal_pairs" in st.session_state:
        pairs = st.session_state["editor_portal_pairs"]
    else:
        ordered = list(portal_specs.keys())
        pairs = [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)]

    # Runtime wiring via mutual refs (kept for compatibility)
    for a_pos, b_pos in pairs:
        a = portal_specs.get(a_pos)
        b = portal_specs.get(b_pos)
        if a is None or b is None or a is b:
            continue
        try:
            setattr(a, "portal_pair_ref", b)
            if getattr(b, "portal_pair_ref", None) is None:
                setattr(b, "portal_pair_ref", a)
        except Exception:
            pass

    return level


def _tool_icon(palette: Mapping[str, ToolSpec], key: str, default: str = "") -> str:
    spec = palette.get(key)
    return spec.icon if spec is not None else default


def _registry_key_by_value(reg: Mapping[str, Any], value: Any, default_key: str) -> str:
    for k, v in reg.items():
        if v is value:
            return k
    return default_key


def _factory_import_line(fn: Callable[..., BaseEntity]) -> str | None:
    mod = getmodule(fn)
    if mod is None or not hasattr(fn, "__name__"):
        return None
    return f"from {mod.__name__} import {fn.__name__}"


def _render_arg_value(v: Any, extra_imports: list[str]) -> str:
    # Enum -> QualName and add import for class
    if isinstance(v, Enum):
        cls = v.__class__
        mod = cls.__module__
        name = cls.__name__
        qual = f"{name}.{v.name}"
        imp = f"from {mod} import {name}"
        if imp not in extra_imports:
            extra_imports.append(imp)
        return qual
    return repr(v)


def generate_code_from_palette(
    cfg: EditorConfig,
    palette: Mapping[str, ToolSpec],
    *,
    env_class: type,
    move_fn_registry: Mapping[str, MoveFn],
    objective_fn_registry: Mapping[str, ObjectiveFn],
) -> str:
    move_key = _registry_key_by_value(
        move_fn_registry, cfg.move_fn, next(iter(move_fn_registry))
    )
    obj_key = _registry_key_by_value(
        objective_fn_registry, cfg.objective_fn, next(iter(objective_fn_registry))
    )

    placements: dict[str, list[tuple[int, int]]] = {}
    used_imports: list[str] = []
    portal_positions: list[tuple[int, int]] = []

    # Collect tool placements
    for y in range(cfg.height):
        for x in range(cfg.width):
            tokens = [t for t in cfg.grid_tokens[y][x] if t.type != "erase"]
            if not tokens:
                continue

            # Floor first (explicit or default)
            floor_tok = next((t for t in tokens if t.type == "floor"), None)
            floor_spec = palette.get("floor")
            if floor_spec:
                kwargs = floor_spec.param_map(
                    floor_tok.params if floor_tok else {"cost": 1}
                )
                imp = _factory_import_line(floor_spec.factory_fn)
                if imp:
                    used_imports.append(imp)
                args = ", ".join(
                    f"{k}={_render_arg_value(v, used_imports)}"
                    for k, v in kwargs.items()
                )
                expr = (
                    f"{floor_spec.factory_fn.__name__}({args})"
                    if args
                    else f"{floor_spec.factory_fn.__name__}()"
                )
                placements.setdefault(expr, []).append((x, y))

            # Other tokens
            for tok in tokens:
                if tok.type == "floor":
                    continue
                if tok.type == "portal":
                    portal_positions.append((x, y))
                    continue
                spec = palette.get(tok.type)
                if spec is None:
                    placements.setdefault(
                        f"# Unsupported tool '{tok.type}'", []
                    ).append((x, y))
                    continue
                kwargs = spec.param_map(tok.params)
                imp = _factory_import_line(spec.factory_fn)
                if imp:
                    used_imports.append(imp)
                args = ", ".join(
                    f"{k}={_render_arg_value(v, used_imports)}"
                    for k, v in kwargs.items()
                )
                expr = (
                    f"{spec.factory_fn.__name__}({args})"
                    if args
                    else f"{spec.factory_fn.__name__}()"
                )
                placements.setdefault(expr, []).append((x, y))

    # Portal pairing & unpaired (standardized: fn(pair=...))
    portal_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = [
        (portal_positions[i], portal_positions[i + 1])
        for i in range(0, len(portal_positions) - 1, 2)
    ]
    unpaired = portal_positions[len(portal_pairs) * 2 :]

    portal_lines: list[str] = []
    portal_spec = palette.get("portal")
    if portal_spec:
        if len(portal_pairs) > 0 or len(unpaired) > 0:
            imp = _factory_import_line(portal_spec.factory_fn)
            if imp:
                used_imports.append(imp)
        fn_name = portal_spec.factory_fn.__name__
        for (ax, ay), (bx, by) in portal_pairs:
            portal_lines += [
                f"    p1 = {fn_name}()",
                f"    p2 = {fn_name}(pair=p1)",
                f"    level.add(({ax}, {ay}), p1)",
                f"    level.add(({bx}, {by}), p2)",
            ]
        for ux, uy in unpaired:
            portal_lines += [f"    level.add(({ux}, {uy}), {fn_name}())"]

    # Deduplicate imports preserving order (tools only)
    dedup_imports: list[str] = []
    seen: set[str] = set()
    for imp in used_imports:
        if imp and imp not in seen:
            seen.add(imp)
            dedup_imports.append(imp)

    # Derive env import automatically from env_class
    env_mod = getmodule(env_class)
    if env_mod is None or not hasattr(env_class, "__name__"):
        raise ValueError("env_class must be a normal class with resolvable module/name")
    env_import_line = f"from {env_mod.__name__} import {env_class.__name__}"
    env_class_name = env_class.__name__

    # Emit script
    lines: list[str] = []
    a = lines.append
    a("# Auto-generated by Grid Play Level Editor")
    a("from grid_universe.levels.grid import Level")
    a("from grid_universe.levels.convert import to_state")
    a("from grid_universe.moves import MOVE_FN_REGISTRY")
    a("from grid_universe.objectives import OBJECTIVE_FN_REGISTRY")
    a(env_import_line)
    for imp in dedup_imports:
        a(imp)
    a("")
    a("def build_level() -> Level:")
    tl_arg = f", turn_limit={int(cfg.turn_limit)}" if cfg.turn_limit is not None else ""
    a(
        f"    level = Level(width={cfg.width}, height={cfg.height}, "
        f"move_fn=MOVE_FN_REGISTRY[{repr(move_key)}], "
        f"objective_fn=OBJECTIVE_FN_REGISTRY[{repr(obj_key)}], "
        f"seed={repr(cfg.seed)}{tl_arg})"
    )
    a("")
    for expr, positions in sorted(placements.items(), key=lambda kv: kv[0]):
        if expr.startswith("# Unsupported"):
            for x, y in positions:
                a(f"    {expr} at ({x}, {y})")
            continue
        if len(positions) == 1:
            x, y = positions[0]
            a(f"    level.add(({x}, {y}), {expr})")
        else:
            pos_list = ", ".join([f"({x}, {y})" for (x, y) in positions])
            a(f"    for (x, y) in [{pos_list}]:")
            a(f"        level.add((x, y), {expr})")
    for ln in portal_lines:
        a(ln)
    a("")
    a("    return level")
    a("")
    a(f"def build_env() -> {env_class_name}:")
    a("    def _initial_state_fn(**_: object):")
    a("        return to_state(build_level())")
    a("    state = _initial_state_fn()")
    a(
        f"    return {env_class_name}("
        "render_mode='rgb_array', initial_state_fn=_initial_state_fn, "
        "width=state.width, height=state.height)"
    )
    a("")
    a("if __name__ == '__main__':")
    a("    env = build_env()")
    a("    img = env.render(mode='rgb_array')")
    a("    if img is not None: img.show()")
    return "\n".join(lines)


# ------------ Streamlit UI builder ------------


def build_editor_config(
    current: object,
    *,
    name: str,
    palette: Mapping[str, ToolSpec],
    texture_maps: list[TextureMap],
    move_fn_registry: Mapping[str, MoveFn],
    objective_fn_registry: Mapping[str, ObjectiveFn],
    asset_root_resolver: Callable[[TextureMap], str],
    env_class: type,
) -> EditorConfig:
    base: EditorConfig = (
        current if isinstance(current, EditorConfig) else _default_editor_config()
    )
    st.info(name, icon="🛠️")

    # Top config
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

    # Movement / Objective
    move_names = list(move_fn_registry.keys())
    obj_names = list(objective_fn_registry.keys())
    mc1, mc2 = st.columns([1, 1])
    with mc1:
        try:
            current_move_key = next(
                k for k, v in move_fn_registry.items() if v is base.move_fn
            )
        except StopIteration:
            current_move_key = move_names[0]
        move_label = st.selectbox(
            "Movement rule",
            move_names,
            index=move_names.index(current_move_key),
            key=f"{name}_move_fn",
        )
        move_fn = move_fn_registry[move_label]
    with mc2:
        try:
            current_obj_key = next(
                k for k, v in objective_fn_registry.items() if v is base.objective_fn
            )
        except StopIteration:
            current_obj_key = obj_names[0]
        obj_label = st.selectbox(
            "Objective",
            obj_names,
            index=obj_names.index(current_obj_key),
            key=f"{name}_objective_fn",
        )
        objective_fn = objective_fn_registry[obj_label]

    # Texture maps (strictly from caller)
    if not texture_maps:
        st.error("No texture maps provided.")
        st.stop()
    texture_map = texture_map_section(
        base, key=f"{name}_texture_map", options=texture_maps
    )

    # Working grid
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
            params = tspec.param_ui() or {}
        if tspec.description:
            st.caption(tspec.description or "")

    # Grid editing
    with middle:
        st.subheader("Grid")
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

    # Preview + Export
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
            preview_asset_root = asset_root_resolver(texture_map)
            from grid_universe.renderer.texture import TextureRenderer  # local import

            img = TextureRenderer(
                texture_map=texture_map, asset_root=preview_asset_root
            ).render(state_preview)
            st.image(img, width="stretch")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    with st.expander("Export as Python", expanded=False):
        try:
            code_str = generate_code_from_palette(
                cfg_preview,
                palette,
                env_class=env_class,
                move_fn_registry=move_fn_registry,
                objective_fn_registry=objective_fn_registry,
            )
            st.code(code_str, language="python")
            st.download_button(
                "Download generated_level.py",
                data=code_str,
                file_name="generated_level.py",
                mime="text/x-python",
                width="stretch",
            )
        except Exception as e:
            st.error(f"Code generation failed: {e}")

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


def _default_env_factory(
    initial_state_fn: Callable[..., State], texture_map: TextureMap
) -> GridUniverseEnv:
    sample_state = initial_state_fn()
    return GridUniverseEnv(
        render_mode="rgb_array",
        initial_state_fn=initial_state_fn,
        width=sample_state.width,
        height=sample_state.height,
        render_texture_map=texture_map,
    )


# ------------ LevelSource factory ------------

EnvFactory = Callable[[Callable[..., State], TextureMap], GridUniverseEnv]


def make_level_editor_source(
    *,
    name: str,
    palette: Mapping[str, ToolSpec],
    texture_maps: list[TextureMap],
    env_factory: EnvFactory | None,
    move_fn_registry: Mapping[str, MoveFn],
    objective_fn_registry: Mapping[str, ObjectiveFn],
    asset_root_resolver: Callable[[TextureMap], str],
    env_class: type,
) -> LevelSource:
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
            env_class=env_class,
        )

    def _make_env(cfg: BaseConfig) -> GridUniverseEnv:
        assert isinstance(cfg, EditorConfig)
        # Use custom env factory if provided; else default GU env
        factory = env_factory or _default_env_factory

        def initial_state_fn(**_: Any) -> State:
            lvl = build_level_from_tokens(cfg, palette)
            return to_state(lvl)

        sample_state = initial_state_fn()
        if not sample_state.agent:
            raise ValueError(
                "Level must contain an Agent. Use the 'Agent' tool in the palette to place one before starting."
            )

        return factory(initial_state_fn, cfg.render_texture_map)

    return LevelSource(
        name=name,
        config_type=EditorConfig,
        initial_config=_initial_config,
        build_config=_build_config,
        make_env=_make_env,
    )
