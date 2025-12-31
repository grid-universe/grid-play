import os
import sys
import importlib
import importlib.util
import logging

logging.basicConfig(level=logging.INFO)


def import_plugins(plugin_modules: list[str]) -> None:
    """Import plugins by module name; each should self-register."""
    for name in plugin_modules:
        try:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)
            logging.info(f"Successfully imported plugin module: {name}")
        except Exception as e:
            logging.error(f"Plugin import failed for module '{name}': {e}")


def import_plugin_files(plugin_files: list[str]) -> None:
    """Import plugins from arbitrary Python file paths; each should self-register."""
    for path in plugin_files:
        file_path = os.path.abspath(path)
        if not os.path.isfile(file_path):
            logging.error(f"Plugin file not found: {file_path}")
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        module_name = f"grid_play_plugin_{base_name}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"spec_from_file_location failed for {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logging.info(f"Successfully imported plugin file: {file_path}")
        except Exception as e:
            logging.error(f"Plugin import failed for file '{file_path}': {e}")
