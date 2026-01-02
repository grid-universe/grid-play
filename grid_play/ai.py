import types
import traceback
from typing import Any, Protocol, Type, Literal, TypeAlias

import streamlit as st

from grid_universe.levels.grid import Level
from grid_universe.levels.convert import from_state as level_from_state
from grid_universe.gym_env import GridUniverseEnv, Observation, Action
from grid_universe.state import State

CODE_EDITOR_HEIGHT = 400

AGENT_CODE_TEMPLATE = """from typing import Any
from grid_universe.gym_env import Action
from grid_universe.levels.grid import Level
from grid_universe.gym_env import Observation

class Agent:
    def __init__(self, *args: Any, **kwargs: Any):
        pass
    
    def step(self, obs: Level | Observation) -> Action:
        import random
        return random.choice(list(Action))
"""


class AgentProtocol(Protocol):
    def __init__(self, *args: Any, **kwargs: Any): ...
    def step(self, obs: Level | Observation) -> Action: ...


AgentObsMode: TypeAlias = Literal["Observation (rgb_image)", "Level"]


def load_agent_from_source(source: str) -> tuple[object | None, str | None]:
    """
    Compile user source containing class Agent with signature:
        class Agent:
            def __init__(self, *args: Any, **kwargs: Any): ...
            def step(self, obs: Level | Observation) -> Action: ...
    Returns (agent_instance, error_message).
    """
    mod = types.ModuleType("ai_agent_user_module")
    try:
        exec(compile(source, "<ai_agent>", "exec"), mod.__dict__, mod.__dict__)
    except Exception:
        return None, f"Compile error:\n{traceback.format_exc()}"

    AgentClass: Type[AgentProtocol] | None = getattr(mod, "Agent", None)
    if AgentClass is None or not callable(AgentClass):
        return None, "No class Agent found in the provided code."

    try:
        agent = AgentClass()
    except Exception:
        return None, f"Failed to instantiate Agent:\n{traceback.format_exc()}"

    step_fn = getattr(agent, "step", None)
    if not callable(step_fn):
        return None, "Agent.step(obs) is not callable."

    return agent, None


def agent_observation(env: GridUniverseEnv, mode: AgentObsMode) -> Level | Observation:
    """
    Return observation for agent, independent of env's observation_type.
    """
    if mode == "Level":
        if env.state is None:
            raise ValueError(
                "No environment state available to produce Level observation."
            )
        # Type-narrowing branch ensures env.state is State for type checker
        state: State = env.state
        return level_from_state(state)

    return env._get_obs()


def run_agent_step(env: GridUniverseEnv) -> Action | None:
    """
    Compute one action from the loaded agent using the current observation mode.
    """
    agent: AgentProtocol | None = st.session_state.get("ai_agent")
    if agent is None:
        raise ValueError(
            "No agent loaded. Please load an agent in the Agent Settings dialog."
        )
    try:
        mode: AgentObsMode = (
            st.session_state.get("ai_obs_mode") or "Observation (rgb_image)"
        )
        obs_for_agent = agent_observation(env, mode)
        return agent.step(obs_for_agent)
    except Exception as e:
        raise RuntimeError(f"Error during agent step: {e}") from e


@st.dialog("Agent Settings", width="large")
def show_agent_dialog(env: GridUniverseEnv) -> None:
    """
    Dialog for specifying Agent code, observation mode, and loading the Agent.
    """
    # Defaults
    st.session_state.setdefault("ai_code", AGENT_CODE_TEMPLATE)
    st.session_state.setdefault("ai_obs_mode", "Observation (rgb_image)")
    st.session_state.setdefault("ai_agent", None)
    st.session_state.setdefault("ai_error", None)

    # Observation mode
    st.selectbox(
        "Agent observation type",
        options=["Observation (rgb_image)", "Level"],
        index=0 if st.session_state["ai_obs_mode"] == "Observation (rgb_image)" else 1,
        key="ai_obs_mode",
        help="Choose what your Agent.step(obs) receives.",
    )

    # Agent code input
    code = st.text_area(
        "Agent code",
        value=st.session_state["ai_code"],
        height=CODE_EDITOR_HEIGHT,
        key="ai_code",
    )

    # Load button only
    if st.button("Load Agent", use_container_width=True, key="ai_dialog_load"):
        st.session_state["ai_agent"], st.session_state["ai_error"] = (
            load_agent_from_source(code)
        )
        if st.session_state["ai_error"]:
            st.error(st.session_state["ai_error"])
        elif st.session_state["ai_agent"] is not None:
            st.success("Agent loaded.")
