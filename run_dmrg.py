import numpy as np

from config import get_logger
from data import data_split, encodings, prepare_numeric, process_text, save_object
from dmrg import dmrg_sweep
from logger import Logger
from sampling import generate_samples
from util import mps_stats, log_sweep, random_gauged_mps


def run_dmrg(config):
    np.random.seed(config['random_seed'])
    log_directory = config['log_directory']
    data_path = config['data_path']
    test_fraction = config['test_fraction']
    max_sweeps = config['max_sweeps']
    patience = config['patience']
    num_sites = config['num_sites']
    bond_dimension = config['bond_dimension']
    ix_to_char = config.get('ix_to_char')
    logger, save_name = get_logger()
    logger.info(config)
    tf_logger = Logger('tensorboard/{}'.format(save_name))

    text = process_text(data_path, lower=True, remove_punctuation=False)
    char_to_ix, ix_to_char = encodings(text, ix_to_char)
    site_dimension = len(char_to_ix)
    numeric = prepare_numeric(text, char_to_ix)
    logger.info("Data has {} characters, {} unique.".format(len(text), len(char_to_ix)))
    train_batch, cv_batch, test_batch = data_split(numeric, num_sites, test_fraction)

    mps = random_gauged_mps(num_sites, site_dimension, bond_dimension)

    context = {'config': config, 'step': 0}

    stats_history = [mps_stats(mps, train_batch, cv_batch, test_batch)]
    sweep, cv_bumps = 1, 0

    while sweep <= max_sweeps and cv_bumps <= patience:
        dmrg_sweep(mps, train_batch, context)
        stats = mps_stats(mps, train_batch, cv_batch, test_batch)
        stats_history.append(stats)
        log_sweep(logger, tf_logger, stats, sweep)
        save_path = '{}/{}/mps-after-step-{}.pickle'.format(log_directory, save_name, sweep)
        data_to_save = {'config': config, 'mps': mps, 'ix_to_char': ix_to_char, 'save_name': save_name, 'sweep': sweep}
        save_object(data_to_save, save_path)
        logger.info("saved mps to: {}".format(data_path))

        if config['generate_samples']:
            samples_per_sweep = config['samples_per_sweep']
            samples_txt = list(generate_samples(mps, ix_to_char, samples_per_sweep))

            for phrase in samples_txt:
                logger.info("sample phrase: {}".format(phrase))

        sweep += 1
        cv_bumps = update_cv_bumps(cv_bumps, stats_history)

    return stats


def update_cv_bumps(cv_bumps, stats_history):
    stats, stats_last = stats_history[-1], stats_history[-2]
    fidelity_change = stats['cv_fidelity'] - stats_last['cv_fidelity']
    return 0 if fidelity_change > 0 else cv_bumps + 1
