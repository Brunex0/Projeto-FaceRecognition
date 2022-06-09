import yaml

def load_yaml(path):
    with open(path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded