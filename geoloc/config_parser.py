import os
import yaml
import importlib
from types import SimpleNamespace


def class_from_config(config, instantiate=True, *args, **kwargs):
    def process_value(value):
        if isinstance(value, SimpleNamespace) and hasattr(value, "class_path"):
            init_args = getattr(value, "init_args", None)
            if init_args:
                return class_from_config(value)
            else:
                # Instantiate class with no args
                module_name, class_name = value.class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                return cls()
        return value

    module_name, class_name = config.class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    _cls = getattr(module, class_name)

    init_args = {}
    if hasattr(config, "init_args"):
        for key, value in vars(config.init_args).items():
            init_args[key] = process_value(value)

    init_args.update(kwargs)
    if not instantiate:
        return _cls, init_args
    return _cls(*args, **init_args)


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, (list, tuple)):
        return [dict_to_namespace(i) for i in d]
    return d


def namespace_to_dict(n):
    if isinstance(n, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in n.__dict__.items()}
    elif isinstance(n, dict):
        return {k: namespace_to_dict(v) for k, v in n.items()}
    elif isinstance(n, (list, tuple)):
        return [namespace_to_dict(v) for v in n]
    else:
        return n


def load_config(filepath):
    with open(filepath, "r") as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)


def save_config(config, savedir, prefix=""):
    config_name = "config.yaml" if prefix == "" else f"{prefix}_config.yaml"
    with open(os.path.join(savedir, config_name), "w") as f:
        yaml.safe_dump(namespace_to_dict(config), f)
