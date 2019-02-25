import argparse

from config import read_config
from run_dmrg import run_dmrg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='location of config file, for example ./config/parity.yaml')
    args = parser.parse_args()
    config_path = args.config
    config = read_config(config_path)
    run_dmrg(config)

