import os
import sys
import argparse
from streamlit.web import cli as stcli


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="grid-play",
        description="Grid Play: Interactive environment for Grid Universe games.",
    )
    parser.add_argument(
        "-p",
        "--plugin",
        dest="plugins",
        action="append",
        default=[],
        help="Plugin module to import (repeatable), e.g. grid_adventure.sources.intro",
    )
    parser.add_argument(
        "-f",
        "--plugin-file",
        dest="plugin_files",
        action="append",
        default=[],
        help="Plugin Python file to import (repeatable), e.g. ./plugins/my_source.py",
    )
    parser.add_argument(
        "--built-in-sources",
        action="store_true",
        help="Use built-in level sources. Default is True if no plugins are specified, False otherwise.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Set environment variable to signal grid_play to import plugins
    os.environ["GRID_PLAY_PLUGINS"] = ",".join(args.plugins)
    os.environ["GRID_PLAY_PLUGIN_FILES"] = ",".join(args.plugin_files)

    # Hand off to Streamlit
    sys.argv = ["streamlit", "run", os.path.join(SCRIPT_PATH, "main.py")]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
