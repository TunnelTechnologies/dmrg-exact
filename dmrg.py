import numpy as np


from util import zip_outer, make_one_hot, factor_local_tensor, get_shape


def fresh_local_tensor(effective_data):
    zipped = zip_outer(*effective_data)
    local_sum = sum(zipped)
    return local_sum / np.linalg.norm(local_sum)


def update_right_combs(mps, data, site_index, context):
    key = 'combs_r'
    combs = context.get(key)

    if not combs:
        combs = [None] * len(mps)
        context[key] = combs

    # walk to first available comb or off the edge
    loc = site_index
    while not combs[loc]:
        loc += 1
        if loc == len(mps):
            break

    # fill in combs from what you found
    while loc > site_index:
            focus = loc - 1
            if focus > 0 and focus < len(mps) - 1:
                # middle site
                combs[focus] = [np.matmul(mps[focus][:, idx, :], comb) for idx, comb in zip(data[:, focus], combs[loc])]
            elif focus == 0:
                # left site
                combs[focus] = [np.matmul(mps[focus][idx, :], comb) for idx, comb in zip(data[:, focus], combs[loc])]
            else:
                # right site
                combs[focus] = [mps[focus][:, idx] for idx in data[:, focus]]
            loc = focus

    if loc != site_index:
        raise IndexError("we didn't make it back!")


def update_left_combs(mps, data, site_index, context):
    key = 'combs_l'
    combs = context.get(key)

    if not combs:
        combs = [None] * len(mps)
        context[key] = combs

    loc = site_index
    while not combs[loc]:
        loc -= 1
        if loc == -1:
            break

    while loc < site_index:
        focus = loc + 1

        # cases: left, then middle, then right
        if focus == 0:
            combs[focus] = [mps[focus][idx, :] for idx in data[:, focus]]
        elif focus < len(mps) - 1:
            combs[focus] = [np.matmul(comb, mps[focus][:, idx, :]) for idx, comb in zip(data[:, focus], combs[loc])]
        else:
            combs[focus] = [np.matmul(comb, mps[focus][:, idx]) for idx, comb in zip(data[:, focus], combs[loc])]

        loc = focus

    if loc != site_index:
        raise IndexError("we didn't make it back!")


def prepare_effective_data(mps, data, site_index, context):
    # data is 2d array data[i, j] = i-th datapoint, j-th position
    # think of data as sparse representation of data_full[i,j,k]
    # where the k index is a vector which happens to be one-hot
    # so we just record the index

    if site_index == 0:
        physical_dimension, _ = mps[site_index].shape

    elif site_index > 0 and site_index < len(mps) - 1:
        _, physical_dimension, _ = mps[site_index].shape

    elif site_index == len(mps) - 1:
        _, physical_dimension = mps[site_index].shape
    else:
        raise ValueError("got bad site_index")

    physical_effective_data = [make_one_hot(idx, physical_dimension) for idx in data[:, site_index]]

    if site_index == 0:
        update_right_combs(mps, data, site_index, context)
        combs_r = context['combs_r'][site_index + 1]
        effective_data = physical_effective_data, combs_r

    elif site_index < len(mps) - 1:
        update_left_combs(mps, data, site_index, context)
        update_right_combs(mps, data, site_index, context)
        combs_r = context['combs_r'][site_index + 1]
        combs_l = context['combs_l'][site_index - 1]
        effective_data = combs_l, physical_effective_data, combs_r

    else:
        update_left_combs(mps, data, site_index, context)
        combs_l = context['combs_l'][site_index - 1]
        effective_data = combs_l, physical_effective_data

    return effective_data, context


# assumes mps is in right gauge to begin with
def right_sweep(mps, data, context):
    # do all but the rightmost site
    for loc in range(len(mps) - 1):
        effective_data, context = prepare_effective_data(mps, data, loc, context)

        local_tensor = fresh_local_tensor(effective_data)
        p, q = factor_local_tensor(local_tensor, shape=get_shape(mps, loc), direction='R')

        mps[loc] = p
        mps[loc + 1] = np.tensordot(q, mps[loc + 1], axes=[-1, 0])

        if context.get('combs_l'):  # not there on the first sweep
            context['combs_l'][loc] = None
        context['combs_r'][loc] = None
        context['step'] += 1

    return mps, context


# assumes mps is in left gauge to begin with
def left_sweep(mps, data, context):
    last_index = len(mps) - 1
    for offset in range(len(mps) - 1):
        loc = last_index - offset

        effective_data, context = prepare_effective_data(mps, data, loc, context)

        local_tensor = fresh_local_tensor(effective_data)
        p, q = factor_local_tensor(local_tensor, shape=get_shape(mps, loc), direction='L')

        mps[loc - 1] = np.matmul(mps[loc - 1], p)
        mps[loc] = q

        context['combs_l'][loc] = None
        context['combs_r'][loc] = None
        context['step'] += 1

    return mps, context


def dmrg_sweep(mps, data, context):
    mps, context = right_sweep(mps, data, context)
    mps, context = left_sweep(mps, data, context)
