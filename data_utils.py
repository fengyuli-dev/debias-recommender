import numpy as np
import pandas as pd
from random import random
import reclab


def generate_ground_truth_matrix(dimensions, environment='random'):
    """
    Generates a ground truth matrix.

    Parameters:
        quantization_method:
        distribution:
        dimensions: a tuple containing the dimensions of the matrix.

    Returns:
        a mxn matrix containing the ground truth for each pair of m users and n items.
    """
    m, n = dimensions

    if environment == 'random':
        return np.random.rand(dimensions)
    else:
        env = reclab.make(environment, num_users=m, num_items=n, noise=0)
        env.reset()
        return env._get_dense_ratings()


def ground_truth_matrix_to_dataset(matrix, sample_prob=0.1, bias=False, shuffle=True):
    """
    Converts a ground truth matrix to a recommender dataset. First simulate the observations on the ground truth
    matrix, then convert them to a recommender dataset.
    Bias can be introduced in the dataset.

    Parameters:
        matrix: a mxn matrix that contains the ground truth for each pair of m users and n items.
        sample_prob: the probability of sampling a chosen pair of users and items.
        bias: the type of sampling bias.
        shuffle: whether to shuffle the dataset by users.

    Returns:
        Same as reclab.data_utils.read_dataset()
    """
    m, n = matrix.shape

    if shuffle:
        np.random.shuffle(matrix)

    if not bias:
        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(matrix[i, j], sample_prob)
        users, items = generate_users_items(ratings, m, n)
        return users, items, ratings

    else:
        # TODO: Introduce bias
        return


def generate_users_items(ratings, m, n):
    """
    Helper function that takes in a dictionary of ratings and outputs two disctionarys of users and items.
    """
    users = {}
    items = {}
    for i in range(m):
        users[i] = [None] * n
    for i in range(n):
        items[i] = [None] * m

    for (user_id, item_id), rating in ratings.items():
        users[user_id][item_id] = rating
        items[item_id][user_id] = rating
    return users, items


def sample(value, sample_prob=0.1):
    """
    Helper function that set the value to zero with probability (1 - sample_prob).
    """
    if random() > sample_prob:
        return None
    else:
        return value


if __name__ == '__main__':
    truth = generate_ground_truth_matrix((1000, 1000), environment='latent-static-v1')
    print(truth)
    # users, items, ratings = ground_truth_matrix_to_dataset(truth)
    # print(users)
    # print(items)
    # print(ratings)
