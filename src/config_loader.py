"""
Loads config.yaml and provides clean access to its values.
Use ConfigLoader in any pipeline script that needs config values.

"""

from pathlib import Path
from typing import Optional
import yaml

class ConfigLoader:
    """
    Loads config.yaml and gives easy access to its values.
    """

    def __init__(self, config_file: Path):
        """
        Parameters
        ----------
        config_file : Path
            Path to config.yaml.
        """
        self.config_path = Path(config_file)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys: str, default=None):
        """
        Accesses a value from the config dict, including nested keys.

        Parameters
        ----------
        *keys : str
            Keys in order of depth.
            e.g. cfg.get("params", "star", "threads")

        default
            Returned if any key in the chain is missing.

        Returns
        -------
        The value at the specified key path, or default.
        """
        value = self.config

        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)

        return value

    def get_path(self, *keys: str, base_path: Optional[Path] = None, must_exist: bool = False):
        """
        Returns a Path object for a value stored in the config.

        Parameters
        ----------
        *keys : str
            Keys to look up (same as get()).

        base_path : Path, optional
            If provided, the retrieved path is joined onto this base.

        must_exist : bool
            If True, raises FileNotFoundError when the path does not exist
            on disk.  Use this for inputs that must already be present.
            Leave False for output directories the pipeline will create.

        Returns
        -------
            Path

        Raises
        ------
        KeyError
            If the key is missing from the config entirely.
        FileNotFoundError
            If must_exist=True and the path does not exist on disk.
        """
        raw = self.get(*keys)

        if raw is None:
            raise KeyError(
                f"Key {' -> '.join(keys)!r} not found in config. "
                f"Have you run the config GUI to set it?"
            )

        p = Path(raw)

        if base_path is not None:
            p = base_path / p

        if must_exist and not p.exists():
            raise FileNotFoundError(f"Path {p} not found (keys: {keys})")

        return p

    def check_bools(self):
        """
        Validates that all fields listed in bool_fields are proper Python
        booleans (True/False), not strings like 'true' or 'yes'.

        Raises ValueError if any are mis-formatted.
        """
        errors     = []
        bool_fields = {}  # populate with field names that must be boolean

        def recurse(value, path=""):
            for k, v in value.items():
                current_path = f"{path}.{k}" if path else k
                if isinstance(v, dict):
                    recurse(v, current_path)
                elif k in bool_fields and not isinstance(v, bool):
                    errors.append(current_path)

        recurse(self.config)

        if errors:
            raise ValueError(
                "Invalid boolean fields in config.yaml — use True/False not strings:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
        else:
            print("All boolean fields valid, continuing pipeline")
