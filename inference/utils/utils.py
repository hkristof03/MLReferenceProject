from pathlib import Path
import logging
import inspect
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import yaml


def get_logger() -> logging.Logger:
    caller = inspect.stack()[1][3]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(caller)


def read_config() -> DictConfig:
    """Reads, parses and returns a yaml file at a given path.
    """
    path_config = Path().resolve().joinpath('config', 'config.yaml')

    with open(path_config, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    config = OmegaConf.create(config)

    return config
