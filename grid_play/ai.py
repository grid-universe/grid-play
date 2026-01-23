import types
import traceback
from typing import Any, Protocol, Type, Literal, TypeAlias

import streamlit as st

from grid_universe.grid.gridstate import GridState
from grid_universe.grid.convert import from_state as gridstate_from_state, to_state
from grid_universe.env import GridUniverseEnv, ImageObservation, Action
from grid_universe.state import State
from PIL.Image import Image as PILImage

CODE_EDITOR_HEIGHT = 400

AGENT_CODE_TEMPLATE = """from typing import Any
from grid_universe.env import Action
from grid_universe.grid.gridstate import GridState
from grid_universe.env import ImageObservation

class Agent:
    def __init__(self, *args: Any, **kwargs: Any):
        pass
    
    def step(self, obs: GridState | ImageObservation) -> Action:
        import random
        return random.choice(list(Action))

    def parse(self, obs: ImageObservation) -> GridState:
        # Optional: implement if you want to see agent vision
        from grid_universe.grid.gridstate import GridState
        from grid_universe.movements import CardinalMovement
        from grid_universe.objectives import ExitObjective
        return GridState(width=5, height=5, movement=CardinalMovement(), objective=ExitObjective())  # Dummy implementation

    def info(self) -> dict[str, Any]:
        # Optional: return info about the agent
        return {"name": "Random AI Agent"}
"""


class AgentProtocol(Protocol):
    def __init__(self, *args: Any, **kwargs: Any): ...
    def step(self, obs: GridState | ImageObservation) -> Action: ...
    def parse(self, obs: ImageObservation) -> GridState: ...
    def info(self) -> dict[str, Any]: ...


AgentObsMode: TypeAlias = Literal["ImageObservation", "GridState"]


def load_ai_agent_from_source(source: str) -> tuple[object | None, str | None]:
    """
    Compile user source containing class Agent with signature:
        class Agent:
            def __init__(self, *args: Any, **kwargs: Any): ...
            def step(self, obs: GridState | ImageObservation) -> Action: ...
            def parse(self, obs: ImageObservation) -> GridState: ...
            def info(self) -> dict[str, Any]: ...
    Returns (ai_agent_instance, error_message).
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


def ai_agent_observation(
    env: GridUniverseEnv, mode: AgentObsMode
) -> GridState | ImageObservation:
    """
    Return observation for agent, independent of env's observation_type.
    """
    if mode == "GridState":
        if env.state is None:
            raise ValueError(
                "No environment state available to produce GridState observation."
            )
        # Type-narrowing branch ensures env.state is State for type checker
        state: State = env.state
        return gridstate_from_state(state)

    return env._get_obs()


def get_ai_agent_vision(env: GridUniverseEnv) -> PILImage | None:
    """
    Run agent.parse(obs) -> GridState -> State -> Rendered Image.
    Returns None if agent not loaded, parse not implemented, or errors occur.
    """
    agent: AgentProtocol | None = st.session_state.get("ai_agent")
    if agent is None:
        return None

    parse_fn = getattr(agent, "parse", None)
    if not callable(parse_fn):
        return None

    try:
        mode: AgentObsMode = st.session_state.get("ai_obs_mode") or "ImageObservation"
        obs: ImageObservation | GridState = ai_agent_observation(env, mode)
        gridstate_result: Any
        if isinstance(obs, GridState):
            st.warning("Agent ImageObservation is GridState.", icon="🧩")
            gridstate_result = obs
        else:
            gridstate_result = parse_fn(obs)

        if not isinstance(gridstate_result, GridState):
            st.error("Agent.parse(obs) did not return a GridState instance.", icon="⚠️")
            return None

        state = to_state(gridstate_result)
        env._setup_renderer()
        assert env._image_renderer is not None
        return env._image_renderer.render(state)
    except Exception:
        st.error(f"Error during agent vision parsing:\n{traceback.format_exc()}")
        pass

    return None


def get_ai_agent_info() -> dict[str, Any]:
    """
    Get agent info string from agent.info() if available.
    """
    agent: AgentProtocol | None = st.session_state.get("ai_agent")
    if agent is None:
        return {}

    info_fn = getattr(agent, "info", None)
    if not callable(info_fn):
        return {}

    try:
        info_result: Any = info_fn()
        if not isinstance(info_result, dict):
            st.error("Agent.info() did not return a dict.", icon="⚠️")
            return {}
        return info_result
    except Exception:
        pass
    return {}


def run_ai_agent_step(env: GridUniverseEnv) -> Action | None:
    """
    Compute one action from the loaded agent using the current ImageObservation mode.
    """
    agent: AgentProtocol | None = st.session_state.get("ai_agent")
    if agent is None:
        raise ValueError(
            "No agent loaded. Please load an agent in the Agent Settings dialog."
        )
    try:
        mode: AgentObsMode = st.session_state.get("ai_obs_mode") or "ImageObservation"
        obs_for_agent = ai_agent_observation(env, mode)
        return agent.step(obs_for_agent)
    except Exception as e:
        raise RuntimeError(f"Error during agent step: {e}") from e


@st.dialog("AI Agent Settings", width="large")
def show_ai_agent_dialog(env: GridUniverseEnv) -> None:
    """
    Dialog for specifying AI Agent code, ImageObservation mode, and loading the AI Agent.
    """
    # Defaults
    st.session_state.setdefault("ai_code", AGENT_CODE_TEMPLATE)
    st.session_state.setdefault("ai_obs_mode", "ImageObservation")
    st.session_state.setdefault("ai_agent", None)
    st.session_state.setdefault("ai_error", None)

    # Observation
    obs_mode = st.selectbox(
        "Observation",
        options=["ImageObservation", "GridState"],
        index=0 if st.session_state["ai_obs_mode"] == "ImageObservation" else 1,
        key="ai_obs_mode_widget",
        help="Choose what your Agent.step(obs) receives.",
    )

    # Code
    code = st.text_area(
        "Code",
        value=st.session_state["ai_code"],
        height=CODE_EDITOR_HEIGHT,
        key="ai_code_widget",
    )

    # Load button only
    if st.button("Load", use_container_width=True, key="ai_dialog_load"):
        st.session_state["ai_obs_mode"] = obs_mode
        st.session_state["ai_code"] = code
        st.session_state["ai_agent"], st.session_state["ai_error"] = (
            load_ai_agent_from_source(code)
        )
        if st.session_state["ai_error"]:
            st.error(st.session_state["ai_error"])
        elif st.session_state["ai_agent"] is not None:
            st.success("AI Agent loaded.")
            st.rerun()
