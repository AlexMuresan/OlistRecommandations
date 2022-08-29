from itertools import combinations

import numpy as np


def triangle_prime(user_item_matrix, items=False):
    user_item = user_item_matrix.T if items else user_item_matrix
    nr_users = user_item.shape[0]

    sum_1 = np.array(
        [
            np.sqrt(np.sum(np.square(np.diff(comb, axis=0))))
            for comb in combinations(user_item, 2)
        ]
    )
    user_sums = np.sqrt(np.sum(np.square(user_item), axis=1))
    sum_2 = np.array([np.sum(i) for i in combinations(user_sums, 2)])

    sum_3 = 1 - (sum_1 / sum_2)

    tri = np.zeros((nr_users, nr_users))
    tri[np.triu_indices(nr_users, 1)] = sum_3

    return tri


def urp(user_item_matrix, items=False):
    user_item_masked = np.ma.array(user_item_matrix.T if items else user_item_matrix)

    nr_users = user_item_masked.shape[0]
    user_item_masked = np.ma.masked_where(user_item_masked == 0, user_item_masked)

    means = user_item_masked.mean(axis=1).data
    stds = user_item_masked.std(axis=1).data

    mean_diffs = np.array([np.abs(np.diff(mean)) for mean in combinations(means, 2)])
    std_diffs = np.array([np.abs(np.diff(std)) for std in combinations(stds, 2)])

    tri_upper = (1 - 1 / (1 + np.exp(-1 * mean_diffs * std_diffs))).squeeze()

    tri = np.zeros((nr_users, nr_users))
    tri[np.triu_indices(nr_users, 1)] = tri_upper

    return tri


def itr(user_item_matrix, items=False, full_matrix=False):
    triangle_res = triangle_prime(user_item_matrix, items=items)
    urp_res = urp(user_item_matrix, items=items)
    itr_res = triangle_res * urp_res

    if full_matrix:
        itr_res += itr_res.T

    return itr_res
