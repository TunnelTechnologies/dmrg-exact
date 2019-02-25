import numpy as np

from util import contract


def unnormalized_multinomial(weights):
    eps = 1e-8
    adjusted_weights = np.array([max(w, eps) for w in weights])
    p_vals = adjusted_weights / sum(adjusted_weights)

    return np.random.choice(range(len(weights)), p=p_vals)


def born_sample(matrix):
    return unnormalized_multinomial(matrix.diagonal())


def generate_samples(mps, ix_to_char, num_samples=20):
    samples_ix = (sample_from_right_normalized(mps) for _ in range(num_samples))
    samples_txt = (''.join(ix_to_char[ix] for ix in sample_ix) for sample_ix in samples_ix)
    return samples_txt


def sample_from_right_normalized(mps):
    def samples_generator():
        first_tensor = mps[0]
        B = np.einsum('ix, jx -> ij', first_tensor, first_tensor)
        draw = born_sample(B)
        yield draw
        left_comb = first_tensor[draw, :]

        for tensor in mps[1:-1]:
            X = contract(left_comb, tensor)
            B = np.einsum('ix, jx -> ij', X, X)
            draw = born_sample(B)
            yield draw
            left_comb = contract(left_comb, tensor[:, draw, :])

        last_tensor = mps[-1]
        X = contract(left_comb, last_tensor)
        B = np.outer(X, X)
        draw = born_sample(B)
        yield draw
    return list(samples_generator())
