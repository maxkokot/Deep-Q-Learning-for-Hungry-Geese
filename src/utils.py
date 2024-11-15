import yaml


def load_config(config_path):
    """Loads yaml config
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def add_n_dims(config):
    """Calculates number of dimension
    for state representation
    given number of enemies
    """
    config['n_dims'] = (config['n_enemies'] + 1) * 4 + 1
    return config
