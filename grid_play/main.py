from __future__ import annotations

import os
import streamlit as st

from dataclasses import replace
from pyrsistent import thaw

# --------- Main App Imports ---------

from grid_play.config import (
    AppConfig,
    set_default_config,
    get_config_from_widgets,
    make_env_and_reset,
)
from grid_play.components import (
    display_powerup_status,
    display_inventory,
    get_keyboard_action,
    do_action,
)
from grid_play.ai import (
    show_ai_agent_dialog,
    run_ai_agent_step,
    get_ai_agent_vision,
    get_ai_agent_info,
)
from grid_play.plugins import import_plugins, import_plugin_files
from grid_universe.gym_env import GridUniverseEnv, Observation, Action

# --------- Built-in Level Sources ---------

BUILT_IN_SOURCES = [
    "grid_play.config.sources.examples.maze",
    "grid_play.config.sources.examples.gameplay",
    "grid_play.config.sources.examples.cipher",
    "grid_play.config.sources.examples.editor",
]

# --------- Plugin Import ---------

plugin_modules_env = os.getenv("GRID_PLAY_PLUGINS", "")
plugin_files_env = os.getenv("GRID_PLAY_PLUGIN_FILES", "")

plugin_modules = [
    name.strip() for name in plugin_modules_env.split(",") if name.strip()
]
plugin_files = [path.strip() for path in plugin_files_env.split(",") if path.strip()]

if not plugin_modules and not plugin_files:
    plugin_modules = (
        BUILT_IN_SOURCES  # Default to built-in sources if no plugins specified
    )

import_plugins(plugin_modules)
import_plugin_files(plugin_files)

# --------- Streamlit App Setup ---------

SCRIPT_DIR: str = os.path.dirname(os.path.realpath(__file__))

st.set_page_config(layout="wide", page_title="Grid Play")

with open(os.path.join(SCRIPT_DIR, "styles.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------- Main App ---------

set_default_config()
tab_game, tab_config, tab_state = st.tabs(["Game", "Config", "State"])

with tab_config:
    config: AppConfig = get_config_from_widgets()
    st.session_state["config"] = config

    if st.button("Save", key="save_config_btn", width="stretch"):
        st.session_state["seed_counter"] = 0
        base_seed = config.seed if config.seed is not None else 0
        st.session_state["config"] = replace(config, seed=base_seed)
        make_env_and_reset(st.session_state["config"])
    st.divider()

with tab_game:
    if "env" not in st.session_state or "obs" not in st.session_state:
        make_env_and_reset(st.session_state["config"])

    left_col, middle_col, right_col = st.columns([0.25, 0.5, 0.25])

    with right_col:
        current_cfg: AppConfig = st.session_state["config"]
        if st.button("🔁 New Level", key="generate_btn", width="stretch"):
            st.session_state["seed_counter"] += 1
            base_seed = current_cfg.seed if current_cfg.seed is not None else 0
            new_seed = base_seed + st.session_state["seed_counter"]
            st.session_state["config"] = replace(current_cfg, seed=new_seed)
            make_env_and_reset(st.session_state["config"])

        # Need to put after generate maze
        env: GridUniverseEnv = st.session_state["env"]
        obs: Observation = st.session_state["obs"]
        info: dict[str, object] = st.session_state["info"]

        if env.state:
            maze_rule = env.state.movement.description
            st.info(f"{maze_rule}", icon="🚶")

            objective = env.state.objective.description
            message = env.state.message

            st.info(f"{objective}", icon="🎯")
            if message:
                st.info(f"{message}", icon="💬")
            if env.state.turn_limit is not None:
                st.info(f"Turn: {env.state.turn} / {env.state.turn_limit}", icon="⏳")

        tab_human, tab_ai_agent = st.tabs(["Human", "AI Agent"])

        with tab_human:
            _, up_col, _ = st.columns([1, 1, 1])
            with up_col:
                if st.button("⬆️", key="up_btn", width="stretch"):
                    do_action(env, Action.UP)
            left_btn, down_btn, right_btn = st.columns([1, 1, 1])
            with left_btn:
                if st.button("⬅️", key="left_btn", width="stretch"):
                    do_action(env, Action.LEFT)
            with down_btn:
                if st.button("⬇️", key="down_btn", width="stretch"):
                    do_action(env, Action.DOWN)
            with right_btn:
                if st.button("➡️", key="right_btn", width="stretch"):
                    do_action(env, Action.RIGHT)

            pickup_btn, usekey_btn, wait_btn = st.columns([1, 1, 1])
            with pickup_btn:
                if st.button("🤲 Pickup", key="pickup_btn", width="stretch"):
                    do_action(env, Action.PICK_UP)
            with usekey_btn:
                if st.button("🔑 Use", key="usekey_btn", width="stretch"):
                    do_action(env, Action.USE_KEY)
            with wait_btn:
                if st.button("⏳ Wait", key="wait_btn", width="stretch"):
                    do_action(env, Action.WAIT)

            action: Action | None = get_keyboard_action()

        with tab_ai_agent:
            ai_agent_settings_btn, ai_agent_step_btn = st.columns([1, 1])
            ai_agent_step: bool = False

            ai_agent: object | None = st.session_state.get("ai_agent")
            if ai_agent is None:
                st.warning(
                    "No AI agent loaded. Please load an AI agent in the Settings dialog.",
                    icon="⚠️",
                )

            with ai_agent_settings_btn:
                if st.button(
                    "Settings", key="ai_open_dialog", width="stretch", icon="⚙️"
                ):
                    show_ai_agent_dialog(env)

            with ai_agent_step_btn:
                if st.button(
                    "Step", key="ai_step_btn", width="stretch", icon="✨"
                ):
                    ai_agent_step = True

            if ai_agent_step:
                try:
                    action = run_ai_agent_step(env)
                    st.info(f"AI Agent chose action: {action}", icon="✨")
                except Exception as e:
                    st.error(e)

            # Execute action from keyboard or AI agent
            if action is not None:
                do_action(env, action)

            vision_img = get_ai_agent_vision(env)
            if vision_img is not None:
                st.text("Vision:")
                st.image(vision_img, width="stretch")
            ai_agent_info = get_ai_agent_info()
            if ai_agent_info:
                st.text("Info:")
                st.json(ai_agent_info)

    with left_col:
        state = env.state
        if state is not None:
            st.info(f"**Total Reward:** {st.session_state['total_reward']}", icon="🏅")

            agent_id = env.agent_id

            if agent_id is not None:
                health = state.health[agent_id]
                st.info(
                    f"**Health Point:** {health.current_health} / {health.max_health}",
                    icon="❤️",
                )
                prev_health = st.session_state["prev_health"]
                if health.current_health < prev_health:
                    st.toast(
                        f"Taking {health.current_health - prev_health} damage!",
                        icon="🔥",
                    )
                    st.session_state["prev_health"] = health.current_health

                display_powerup_status(state, state.status[agent_id])
                display_inventory(state, state.inventory[agent_id])

    with middle_col:
        if env.state and env.state.win:
            st.success("🎉 **Goal reached!** 🎉")
            st.balloons()
        if env.state and env.state.lose:
            st.error("💀 **You lose!** 💀")
        img = env.render(mode="rgb_array")
        if img is not None:
            img_compressed = img.convert("P")  # Converts to 8-bit palette mode
            st.image(img_compressed, width="stretch")
        if obs:
            # Re-fetch current observation, otherwise info may be stale
            current_obs = env._get_obs()
            if isinstance(current_obs, dict) and "info" in current_obs:
                st.json(current_obs["info"], expanded=1)

with tab_state:
    if env.state:
        st.json(thaw(env.state.description), expanded=1)
