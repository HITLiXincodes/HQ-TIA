import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
PATH_KEYS = {
    "dataset_dir",
    "training_output_dir",
    "face_parsing_dir",
    "face_parsing_checkpoint",
    "face_system_dir",
    "adaface_checkpoint",
    "checkpoint",
    "real_output_dir",
    "fake_output_dir",
    "input_root",
    "target_root",
    "fake_embedding_output_dir",
    "real_embedding_output_dir",
    "ref_dir",
    "gen_dir",
    "output_json",
    "output_txt",
    "real_embedding_root",
    "fake_embedding_root",
    "output_csv",
}


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values with dot paths, e.g. --set train.batch_size=32.",
    )


def load_config(config_path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config = copy.deepcopy(config)
    for override in overrides or []:
        _apply_override(config, override)
    _resolve_paths(config, PROJECT_ROOT)
    return config


def require_config_path(config: Dict[str, Any], dotted_path: str) -> str:
    current: Any = config
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config value: {dotted_path}")
        current = current[key]
    if current in (None, ""):
        raise ValueError(f"Config value '{dotted_path}' must be set before running this script.")
    return str(current)


def resolve_device(device: str):
    import torch
    requested = str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device was requested but is not available: {device}")
    if resolved.type == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was requested but is not available.")
    return resolved


def _apply_override(config: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")

    dotted_path, raw_value = override.split("=", 1)
    keys = [key for key in dotted_path.strip().split(".") if key]
    if not keys:
        raise ValueError(f"Invalid override path in '{override}'.")

    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = _parse_value(raw_value)


def _parse_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _resolve_paths(value: Any, root: Path, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            value[child_key] = _resolve_paths(child_value, root, child_key)
        return value

    if key not in PATH_KEYS or value in (None, ""):
        return value

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())
