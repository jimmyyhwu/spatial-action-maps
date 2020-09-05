import configparser
from multiprocessing.dummy import Pool
from pathlib import Path
import anki_vector

def get_config_path():
    return Path.home() / '.anki_vector/sdk_config.ini'

def get_config():
    config = configparser.ConfigParser()
    config.read(get_config_path())
    return config

def write_config(config):
    with open(get_config_path(), 'w') as f:
        config.write(f)

def get_robot_names():
    config = get_config()
    return sorted(config[serial]['name'] for serial in config.sections())

def get_robot_serials():
    config = get_config()
    return {config[serial]['name']: serial for serial in config.sections()}

def get_robot_indices():
    config = get_config()
    return {config[serial]['name']: i for i, serial in enumerate(config.sections())}

def get_available_robots(num_robots=10):
    def ping(args):
        name, serial = args
        try:
            with anki_vector.Robot(serial=serial, default_logging=False) as _:
                return name
        except:
            return None

    robot_serials = get_robot_serials()
    available_names = []
    with Pool(len(robot_serials)) as p:
        it = p.imap_unordered(ping, robot_serials.items())
        for name in it:
            if name is not None:
                available_names.append(name)
            if len(available_names) > num_robots:
                return available_names
        return available_names

def get_first_available_robot():
    names = get_available_robots(num_robots=1)
    if len(names) > 0:
        return names[0]
    return None
