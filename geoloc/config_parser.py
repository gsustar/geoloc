import os
import yaml
import json
import importlib
from types import SimpleNamespace
from typing import Any, IO

from functools import partial

class SimpleNamespaceWithPop(SimpleNamespace):

    def get(self, key, default=None):
        return getattr(self, key, default)

    def pop(self, key, default=None):
        if hasattr(self, key):
            value = getattr(self, key)
            delattr(self, key)
            return value
        return default


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""
    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')
    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def _load_checkpoint_helper(value, checkpoint_path, nested_key=None):
    dirpath = os.path.dirname(checkpoint_path)
    if os.path.exists(os.path.join(dirpath, "train_config.yaml")):
        train_config_path = os.path.join(dirpath, "train_config.yaml")
        value = load_config(train_config_path).model
    model_cls, init_args = class_from_config(value, instantiate=False, from_load_checkpoint=True, key=nested_key)
    retcls = model_cls.load_from_checkpoint(
        checkpoint_path, strict=False, weights_only=False,
        **init_args
    )
    return retcls

def class_from_config(config, instantiate=True, key=None, from_load_checkpoint=False, *args, **kwargs):
    if config is None:
        return None

    def process_value(value, nested_key=None):
        if isinstance(value, SimpleNamespace) and hasattr(value, "class_path"):
            init_args = getattr(value, "init_args", None)
            checkpoint_path = getattr(value, "load_checkpoint", None)
            if checkpoint_path:
                retcls = _load_checkpoint_helper(value, checkpoint_path, nested_key=nested_key)
            elif init_args:
                retcls = class_from_config(value, key=nested_key)
            else:
                # Instantiate class with no args
                module_name, class_name = value.class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                retcls = cls()
            retcls.my_config = value
            return retcls
        if isinstance(value, list):
            return [process_value(v) for v in value]
        return value

    module_name, class_name = config.class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    _cls = getattr(module, class_name)

    if hasattr(config, "load_checkpoint") and not from_load_checkpoint:
        checkpoint_path = config.load_checkpoint
        ret_cls = _load_checkpoint_helper(config, checkpoint_path, nested_key=key)
        return ret_cls
    
    init_args = {}
    if hasattr(config, "init_args"):
        for ikey, ivalue in vars(config.init_args).items():
            init_args[ikey] = process_value(ivalue, nested_key=ikey)
    init_args.update(kwargs)

    if key in ('optimizer', 'scheduler'):
        return partial(_cls, **init_args)
    
    if not instantiate:
        return _cls, init_args

    ret_cls = _cls(*args, **init_args)
    ret_cls.my_config = config
    return ret_cls


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespaceWithPop(**{k: dict_to_namespace(v) for k, v in d.items()})
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

yaml.add_constructor('!include', construct_include, Loader)
def load_config(filepath):
    with open(filepath, "r") as f:
        config_dict = yaml.load(f, Loader)
    return dict_to_namespace(config_dict)


def save_config(config, savedir, prefix=""):
    config_name = "config.yaml" if prefix == "" else f"{prefix}_config.yaml"
    with open(os.path.join(savedir, config_name), "w") as f:
        yaml.safe_dump(namespace_to_dict(config), f)
