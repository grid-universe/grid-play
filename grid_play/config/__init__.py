import streamlit as st
from .env_factory import make_env_and_reset
from .sources.base import all_level_sources, find_level_source_by_config, LevelSource

from .types import AppConfig

__all__ = [
    "all_level_sources",
    "find_level_source_by_config",
    "make_env_and_reset",
    "set_default_config",
    "get_config_from_widgets",
    "LevelSource",
    "AppConfig",
]


def _initial_config() -> AppConfig:
    if not all_level_sources():
        raise RuntimeError("No level sources registered; cannot create initial config.")
    src = all_level_sources()[-1]
    return src.initial_config()


def set_default_config() -> None:
    if "config" not in st.session_state:
        st.session_state["config"] = _initial_config()
        st.session_state["seed_counter"] = 0


def get_config_from_widgets() -> AppConfig:
    current: AppConfig = st.session_state["config"]
    st.subheader("Level Source")
    sources = all_level_sources()
    source_names = [s.name for s in sources]
    # Determine default index based on current config's source
    current_source = find_level_source_by_config(current)
    default_idx = source_names.index(current_source.name) if current_source else 0
    selected_name = st.selectbox(
        "Source Type",
        source_names,
        index=default_idx,
        help="Select level family (extensible via plug-ins).",
        key="source_mode_select",
    )
    chosen = next(s for s in sources if s.name == selected_name)
    return chosen.build_config(current)
