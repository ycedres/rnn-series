
import os

from datetime import datetime

def create_expFolder(config):
    """
    Create a experiment folder.
    ConfigParser's object with sections of config.ini and
    the key and values of the sections.
    :param config: ConfigParser's object
    :return: create a experiment folder.
    """

    exp_path = config.get('path_exp', 'exp_path')
    exp_name = config.get('path_exp', 'exp_name')
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # -- create exp name
    exp_name = '{0}_{1}'.format(exp_name, now)

    # -- create exp folder
    exp_path = os.path.join(exp_path, exp_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    return exp_path