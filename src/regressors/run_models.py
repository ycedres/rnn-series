
import sys
import socket

from configparser import ConfigParser
from select_models import SelectModels


if socket.gethostname() == 'ml-models':
    sys.path.append('/home/tmorales/regressors')
else:
    sys.path.join('/Users/tmorales/tmp/python-google-cloud')


def run_models(config):
    pass

if __name__ == "__main__":
    config = ConfigParser()
    if socket.gethostname() == 'ml-models':
        config_file = 'config_google.ini'
    else:
        config_file = 'config.ini'

    config.read(config_file)

    # class SelectModels
    models = SelectModels(config)
    # run all models selected in config file
    predicted = models.run()

    print(predicted)


