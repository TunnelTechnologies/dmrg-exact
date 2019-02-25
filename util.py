import numpy as np
import scipy as sp

from itertools import chain
from functools import reduce


def contract(*args):
    return reduce(lambda x, y: np.tensordot(x, y, axes=[-1, 0]), args)


def zip_outer(*args):
    return [reduce(np.multiply.outer, _) for _ in zip(*args)]


def tensor_rank(tensor):
    return len(tensor.shape)


def get_shape(mps, site_index):
    if site_index == 0:
        return 'VB'
    elif site_index > 0 and site_index < len(mps) - 1:
        return 'BVB'
    elif site_index == len(mps) - 1:
        return 'BV'
    else:
        raise ValueError("got bad index", site_index)


def factor_local_tensor(local_tensor, shape='BVB', direction='R'):
    # shape should be VB, BVB, BV
    # direction should be L, R

    if shape in {'VB', 'BV'}:
        u, s, v = np.linalg.svd(local_tensor, full_matrices=False)

        if direction == 'L':
            return np.matmul(u, np.diag(s)), v
        elif direction == 'R':
            return u, np.matmul(np.diag(s), v)
        else:
            raise ValueError("got bad direction", direction)
    elif shape == 'BVB':
        a, b, c = local_tensor.shape
        if direction == 'L':
            reshaped = local_tensor.reshape(a, b * c)
            u, s, v = sp.linalg.svd(reshaped, full_matrices=False)
            left_tensor = np.matmul(u, np.diag(s))
            bond_dim = left_tensor.shape[-1]
            return left_tensor, v.reshape(bond_dim, b, c)
        elif direction == 'R':
            reshaped = local_tensor.reshape(a * b, c)
            u, s, v = np.linalg.svd(reshaped, full_matrices=False)
            right_tensor = np.matmul(np.diag(s), v)
            bond_dim = right_tensor.shape[0]
            return u.reshape(a, b, bond_dim), right_tensor
        else:
            raise ValueError("got bad direction", direction)
    else:
        raise ValueError("got bad shape", shape)


def random_gauged_mps(num_sites, site_dim, bond_dim):
    # builds a random MPS with given length, site_dim, and bond_dim
    left_shape = (site_dim, bond_dim)
    middle_shape = (bond_dim, site_dim, bond_dim)
    right_shape = (bond_dim, site_dim)

    shapes = [left_shape] + (num_sites - 2) * [middle_shape] + [right_shape]

    mps = [np.random.normal(size=shape) for shape in shapes]
    return prepare_right_gauge(mps)


def prepare_right_gauge(mps):
    last_index = len(mps) - 1
    for site_index in range(last_index, 0, -1):
        p, q = factor_local_tensor(mps[site_index], shape=get_shape(mps, site_index), direction='L')
        mps[site_index] = q
        mps[site_index - 1] = np.matmul(mps[site_index - 1], p)
    mps[0] = mps[0] / np.linalg.norm(mps[0])
    return mps


def shape_of_contraction(mps, loc):
    # shape-string for contraction of mps[loc] and mps[loc + 1]
    last_loc = len(mps) - 2
    if loc < 0:
        raise IndexError
    elif loc == 0:
        return 'VVB'
    elif loc < last_loc:
        return 'BVVB'
    elif loc == last_loc:
        return 'BVV'
    else:
        raise IndexError


def make_one_hot(idx, size):
    tmp = np.zeros(size)
    tmp[idx] = 1.0
    return tmp


def average_bond_dimension(mps):
    bonds = [_.shape[-1] for _ in mps[:-1]]
    return np.average(bonds)


def fat_contract(x, y):
    # X_[...]ij, Y_ij[...] => sum_ij X_[...]ij Y_ij[...]
    return np.tensordot(x, y, axes=[[-2, -1], [0, 1]])


def zip_contract(*args):
    return [contract(*tup) for tup in zip(*args)]


def just(x):
    return (x,)


def mps_inner_product(mps1, mps2):
    # computes inner product of pair of mps of the same length, from left

    if len(mps1) != len(mps2):
        raise ValueError("got mps of mismatched length: {}, {}".format(len(mps1), len(mps2)))

    left_piece = np.einsum('ij, ik -> jk', mps1[0], mps2[0])

    middle = (np.einsum('ijk, ljm -> ilkm', x, y) for x, y in zip(mps1[1:-1], mps2[1:-1]))

    right_piece = np.einsum('ij, kj -> ik', mps1[-1], mps2[-1])

    vertical_pairs = chain(just(left_piece), middle, just(right_piece))

    return reduce(fat_contract, vertical_pairs)


def mps_squared_norm(mps):
    return mps_inner_product(mps, mps)


def mps_norm(mps):
    return np.sqrt(mps_squared_norm(mps))


def mps_squared_distance(mps1, mps2):
    # || mps1 - mps2 ||^2 = <mps1 - mps2, mps1 - mps2>
    # = <mps1, mps1> + <mps2, mps2> - 2 <mps1, mps2>
    # but we can compute all of that!
    sqnorm1 = mps_squared_norm(mps1)
    sqnorm2 = mps_squared_norm(mps2)
    inner = mps_inner_product(mps1, mps2)
    return sqnorm1 + sqnorm2 - 2 * inner


def mps_real_distance(mps1, mps2):
    return np.sqrt(mps_squared_distance(mps1, mps2))


def mps_angle(mps1, mps2):
    norm1 = mps_norm(mps1)
    norm2 = mps_norm(mps2)
    inner = mps_inner_product(mps1, mps2)
    cos = inner / (norm1 * norm2)
    angle = np.arccos(cos)

    return angle


def mps_comparison_stats(mps1, mps2):
    stats = dict()

    sqnorm1 = mps_inner_product(mps1, mps1)
    norm1 = np.sqrt(sqnorm1)
    sqnorm2 = mps_inner_product(mps2, mps2)
    norm2 = np.sqrt(sqnorm2)
    inner = mps_inner_product(mps1, mps2)
    cos = inner / (norm1 * norm2)
    angle = np.arccos(cos)
    log_complementary_angle = np.log((np.pi / 2) - angle)

    stats['inner_product'] = inner
    stats['cosine_of_angle'] = cos
    stats['angle'] = angle
    stats['log_complementary_angle'] = log_complementary_angle

    return stats


def mps_stats(mps, train_batch, cv_batch, test_batch):
    stats = dict()
    stats['average_bond_dimension'] = average_bond_dimension(mps)
    stats['train_log_loss'] = global_log_loss(mps, train_batch)
    stats['test_log_loss'] = global_log_loss(mps, test_batch)
    stats['cv_log_loss'] = global_log_loss(mps, cv_batch)
    stats['train_fidelity'] = global_fidelity(mps, train_batch)
    stats['test_fidelity'] = global_fidelity(mps, test_batch)
    stats['cv_fidelity'] = global_fidelity(mps, cv_batch)
    return stats


def log_sweep(logger, tf_logger, stats_dict, sweep_num):
    logger.info('sweep {}'.format(sweep_num))
    for key, value in stats_dict.items():
        logger.info('{}: {}'.format(key, value))
        tf_logger.log_scalar(key, value, sweep_num)


def pair_with_one_hot(mps, indices):
    # Pair MPS with list of one-hot vectors, moving from left
    # Equivalent to pair_with_vectors(mps, [one_hot(ind) for ind in indices])

    left = mps[0][indices[0], :]
    middle_indices, middle_tensors = indices[1:-1], mps[1:-1]
    middle = (tens[:, idx, :] for tens, idx in zip(middle_tensors, middle_indices))
    right = mps[-1][:, indices[-1]]

    slices = chain(just(left), middle, just(right))

    return reduce(contract, slices)


def global_log_loss(mps, batch):
    # get the inner products
    num_items = batch.shape[0]
    inner_products = [pair_with_one_hot(mps, batch[k, :]) for k in range(num_items)]

    squared_norm = mps_inner_product(mps, mps)

    # now compute the loss
    log_likelihood = 0.0

    for inner_product in inner_products:
        log_likelihood += np.log2(np.square(inner_product)) - np.log2(squared_norm)
    return - log_likelihood / len(inner_products)


def global_fidelity(mps, batch):
    # get the inner products
    num_items = batch.shape[0]
    inner_products = [pair_with_one_hot(mps, batch[k, :]) for k in range(num_items)]

    squared_norm = mps_inner_product(mps, mps)

    # now compute the loss
    loss_accumulator = 0.0
    for inner_product in inner_products:
        loss_accumulator += inner_product / np.sqrt(squared_norm)
    return loss_accumulator / np.sqrt(len(inner_products))