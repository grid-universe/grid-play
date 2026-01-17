# Grid Play

A web-based playground for building, configuring, and playing Grid Universe games. Built with Streamlit, Grid Play provides an interactive browser-based interface for playing gridworld games, testing levels, and experimenting with AI agents.

## Install

```bash
pip install -e .
```

Requires Python 3.11+.

## Usage

**Run with built-in sources:**
```bash
grid-play
```

**Load plugin from installed module:**
```bash
grid-play --plugin <plugin_module>
```

Examples:
- `grid_adventure.play.intro` - Pre-built adventure levels (requires `grid-adventure` installed)
- `grid_adventure.play.editor` - Level editor mode (requires `grid-adventure` installed)

**Load plugin from Python file:**
```bash
grid-play --plugin-file path/to/plugin.py
```

**Load multiple plugins:**
```bash
grid-play --plugin module1 --plugin module2 --plugin-file custom.py
```

Opens a Streamlit web interface with:
- **Play:** Interactive game view with keyboard controls (WASD/Arrow keys)
- **Config:** Level selection and game settings (dropdown menus and sliders)
- **AI:** Interactive agent testing (step-by-step execution and visualization)

## Features

- **Visual Rendering:** Real-time game visualization
- **Interactive Controls:** Keyboard-based player movement with action feedback
- **Configuration UI:** Dynamic level selection, parameter adjustment, and environment settings
- **Plugin System:** Load custom games via Python modules or file paths
- **AI Integration:** Test agents with step-by-step execution and state inspection

## Development

```bash
pip install -e ".[dev]"
```

Includes development tools: pytest, mypy, ruff, and type stubs.

## License

MIT License - see [LICENSE](LICENSE) file for details.