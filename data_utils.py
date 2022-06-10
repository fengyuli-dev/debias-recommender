import numpy as np
from scipy.stats import pearsonr, spearmanr
from random import random
from math import exp
import reclab

EPSILON = 1e-8


def generate_ground_truth_matrix(dimensions, environment='random'):
    """
    Generates a ground truth matrix.

    Parameters:
        environment: the behavoral model that establishes the ground truth matrix. A list of predefined, supported
        environments is available at https://github.com/berkeley-reclab/RecLab/blob/master/reclab/environments/registry.py.
        dimensions: a tuple containing the dimensions of the matrix.

    Returns:
        a mxn matrix containing the (continuously distributed) ground truth for each pair of m users and n items.
    """
    m, n = dimensions

    if environment == 'random':
        return np.random.rand(dimensions)
    else:
        env = reclab.make(environment, num_users=m, num_items=n, noise=0)
        env.reset()
        return np.array(env._get_dense_ratings())


def ground_truth_matrix_to_dataset(matrix, quantization, sample_prob=0.1, bias=None, beta=1, noise=0.05):
    """
    Converts a ground truth matrix to a recommender dataset. First simulate the observations on the ground truth
    matrix, then convert them to a recommender dataset.
    Bias can be introduced in the dataset.

    Parameters:
        matrix: a mxn matrix that contains the ground truth for each pair of m users and n items.
        quantization: the way to quantize / discretize the ratings.
        sample_prob: the expected probability of sampling a chosen pair of users and items.
        bias: the type of sampling bias.
        beta: hyperparameter that controls the degress of the bias.
        noise: hyperparameter that controls the degree of noise.

    Returns:
        users: a dictionary whose ith entry is a list of observed ratings for the ith user.
        itmes: a dictionary whose ith entry is a list of observed ratings for the ith item.
        ratings: a dictionary whose entry with key (i, j) is the observed rating for user i on item j.
        P: P[i, j] is the probability of observing rating i on item j.
        R: R is a quantized version of the input matrix. Noise is also added.
    """
    m, n = matrix.shape
    np.random.shuffle(matrix)

    R = matrix.copy()

    # Normalize the ground truth and add Gaussian noise
    R = (R - R.min()) / (R.max() - R.min())
    R = R + np.random.normal(0, noise, R.shape)

    # Quantize the matrix
    if quantization == 'binary':
        R[R <= 0.5] = 0
        R[R > 0.5] = 1
    elif quantization == 'onetofive':
        R[R <= 0.2] = 1
        R[R <= 0.4] = 2
        R[R <= 0.6] = 3
        R[R <= 0.8] = 4
        R[R <= 1] = 5
    else:
        raise ValueError('Quantization scale not supported.')

    print(f'Ground truth after normalization and quantization: {R}')

    if bias is None:
        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(R[i, j], sample_prob)
        users, items = generate_users_items(ratings, m, n)
        return users, items, ratings, P, R

    elif bias == 'popularity':
        average_ratings = np.mean(matrix, axis=0)
        exps = np.exp(beta * average_ratings)
        softmax_result = exps / np.sum(exps)
        # Scale mean to sample_prob
        softmax_result /= np.mean(softmax_result)
        softmax_result *= sample_prob
        P = np.tile(softmax_result, (m, 1))
        assert abs(P.mean()) - sample_prob < EPSILON
        assert P.shape == matrix.shape

        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(R[i, j], P[i, j])
        users, items = generate_users_items(ratings, m, n)
        return users, items, ratings, P, R

    elif bias == 'active user':
        average_ratings = np.mean(matrix, axis=1)
        exps = np.exp(beta * average_ratings)
        softmax_result = exps / np.sum(exps)
        # Scale mean to sample_prob
        softmax_result /= np.mean(softmax_result)
        softmax_result *= sample_prob
        P = np.tile(softmax_result.reshape(1, m), (n, 1))
        assert abs(P.mean()) - sample_prob < EPSILON
        assert P.shape == matrix.shape

        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(R[i, j], P[i, j])
        users, items = generate_users_items(ratings, m, n)
        return users, items, ratings, P, R

    elif bias == 'full underlying':
        P = np.exp(beta * matrix)
        P /= np.sum(P)
        P /= np.mean(P)
        P *= sample_prob
        assert abs(P.mean()) - sample_prob < EPSILON

        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(R[i, j], P[i, j])
        users, items = generate_users_items(ratings, m, n)
        return users, items, ratings, P, R

    else:
        raise ValueError('Bias method not supported.')


def generate_users_items(ratings, m, n):
    """
    Helper function that takes as input a dictionary of ratings and outputs two dictionaries of users and items.
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


def correlation(P, matrix, correlation='pearson'):
    """
    Helper function that computes the correlation between two matrices.
    """
    if correlation == 'pearson':
        return pearsonr(P.flatten(), matrix.flatten())[0]
    elif correlation == 'spearman':
        return spearmanr(P.flatten(), matrix.flatten())[0]


if __name__ == '__main__':
    truth = generate_ground_truth_matrix(
        (1000, 1000), environment='latent-dynamic-v1')
    assert truth.shape == (1000, 1000)
    users, items, ratings, P, R = ground_truth_matrix_to_dataset(
        truth, quantization='onetofive', bias='popularity')

    # count = 0
    # for rating in ratings.values():
    #     if rating is not None:
    #         count += 1
    # print(count / (1000 * 1000))

    print(correlation(P, truth))