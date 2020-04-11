import logging
import logging.config
import os
from datetime import datetime

NOW_FMT = '%a__%b_%d_%Y__%H_%M_%S'
NOW = datetime.utcnow().strftime(NOW_FMT)
DIR = os.path.expanduser(f'~/logs/{NOW}')
os.makedirs(DIR, exist_ok=True)
ALL_FILE = os.path.join(DIR, 'all.log')
ERR_FILE = os.path.join(DIR, 'err.log')
FMT = '%Y.%m.%d %H:%M:%S'

CONFIG = {
    'version': 1,
    'formatters': {
        'detailed': {
            'class': 'logging.Formatter',
            'format': '[%(asctime)s %(msecs)03d][%(name).15s][%(levelname).8s][%(processName)-17s] %(message)s',
            'datefmt': FMT
        },
        'short': {
            'class': 'logging.Formatter',
            'format': '[%(asctime)s][%(levelname).1s] %(message)s',
            'datefmt': FMT
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'short'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': ALL_FILE,
            'mode': 'w',
            'formatter': 'detailed'
        },
        'errors': {
            'class': 'logging.FileHandler',
            'filename': ERR_FILE,
            'mode': 'w',
            'level': 'ERROR',
            'formatter': 'detailed',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file', 'errors']
    }
}


def configure_global_logging(config: dict = None):
    config = config or CONFIG
    logging.config.dictConfig(config)


def get_logger(name: str = None, level: int = logging.DEBUG):
    log = logging.getLogger(name)
    log.setLevel(level)
    return log

