import yaml

def load_yaml(path):
    """
    Load content from congig file .yml
    :param path: the path to the file
    :return: the data of the file
    """
    with open(path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded