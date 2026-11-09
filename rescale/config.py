from pathlib import Path
import yaml

DATASETS = ('CVC-ClinicDB', 'ISIC2018', 'BUSI', 'PAPILA', 'PTX498', 'FUSeg2021')

def load_config(config_path: str, dataset: str=None) -> dict:
    """Load the shared config and resolve paths for the selected dataset."""
    path = Path(config_path).expanduser().resolve()
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'Config file must contain a mapping at the top level: {path}')
    default_dataset = cfg.pop('dataset', 'CVC-ClinicDB')
    selected = dataset if dataset is not None else default_dataset
    if selected not in DATASETS:
        raise ValueError(f'Unknown dataset {selected!r}; choose from {", ".join(DATASETS)}')
    for section, key in (('data', 'manifest_json'),
                         ('data', 'splits_json'), ('output', 'save_dir')):
        cfg[section][key] = cfg[section][key].format(dataset=selected)
    return cfg

def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply dot-notation key=value overrides to a nested config dict.

    Args:
        cfg: Base configuration dict.
        overrides: List of strings in the form "section.key=value".

    Returns:
        Modified config dict (in-place mutation, also returned for convenience).
    """
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Invalid override format (expected key=value): {item}')
        (key_path, value_str) = item.split('=', 1)
        keys = key_path.strip().split('.')
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        value = _cast_value(value_str)
        node[keys[-1]] = value
        print(f'[Override] {key_path} = {value!r}')
    return cfg

def _cast_value(s: str):
    """Attempt to cast a string to int, float, or bool; fall back to str."""
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    if s.lower() in {'null', 'none'}:
        return None
    if s.startswith(('[', '{')):
        return yaml.safe_load(s)
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
