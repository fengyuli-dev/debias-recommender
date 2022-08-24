from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import reclab
from math import exp
from random import random
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from copy import deepcopy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


EPSILON = 1e-8
device = torch.device('cpu')


def generate_ground_truth_matrix(dimensions, environment='random'):
    m, n = dimensions
    if environment == 'random':
        return np.random.rand(dimensions)
    elif environment == 'ml-100k-v1':
        env = reclab.make(environment)
        env.reset()
        return np.array(env._get_dense_ratings())
    else:
        env = reclab.make(environment, num_users=m, num_items=n, noise=0)
        env.reset()
        return np.array(env._get_dense_ratings())


def ground_truth_matrix_to_dataset(matrix, quantization, sample_prob=0.1, bias=None, beta=1, noise=0.05):
    m, n = matrix.shape
    np.random.shuffle(matrix)
    R = matrix.copy()

    # Normalize the ground truth and add Gaussian noise
    R_no_noise = (R - R.min()) / (R.max() - R.min())
    R = R_no_noise + np.random.normal(0, noise, R.shape)
    R[R > 1] = 1
    R[R <= 0] = 1e-7
    R_no_noise[R_no_noise <= 0] = 1e-7

    # Quantize the matrix
    if quantization == 'binary':
        bins = np.linspace(0, 1, 3)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    elif quantization == 'onetofive':
        bins = np.linspace(0, 1, 6)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    elif quantization == 'onetothree':
        bins = np.linspace(0, 1, 4)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    elif quantization == 'onetoten':
        bins = np.linspace(0, 1, 11)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    else:
        raise ValueError('Quantization scale not supported.')

    if bias is None or beta == 0:
        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(R[i, j], sample_prob)
        users, items = generate_users_items(ratings, m, n)
        P = np.ones(matrix.shape) * sample_prob
        return users, items, ratings, P, R, R_no_noise

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
        return users, items, ratings, P, R, R_no_noise

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
        return users, items, ratings, P, R, R_no_noise

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
        return users, items, ratings, P, R, R_no_noise

    else:
        raise ValueError('Bias method not supported.')


def mixing_mf_dataset(matrix, quantization, sample_prob=0.1, bias=None, beta=1, noise=0.05):
    # Bias must be 'popularity', quantization must be 'onetofive'
    m, n = matrix.shape
    np.random.shuffle(matrix)
    R = matrix.copy()

    # Normalize the ground truth and add Gaussian noise
    R_no_noise = (R - R.min()) / (R.max() - R.min())
    R = R_no_noise + np.random.normal(0, noise, R.shape)
    R[R > 1] = 1
    R[R <= 0] = 1e-7
    R_no_noise[R_no_noise <= 0] = 1e-7

    # Quantize the matrix
    if quantization == 'binary':
        bins = np.linspace(0, 1, 3)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    elif quantization == 'onetofive':
        bins = np.linspace(0, 1, 6)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    elif quantization == 'onetothree':
        bins = np.linspace(0, 1, 4)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    elif quantization == 'onetoten':
        bins = np.linspace(0, 1, 11)
        indexes = np.digitize(R, bins)
        indexes_no_noise = np.digitize(R_no_noise, bins, right=True)
        bins = np.append(bins, 1)
        for i in range(m):
            for j in range(n):
                R[i][j] = bins[indexes[i][j]]
                R_no_noise[i][j] = bins[indexes_no_noise[i][j]]
    else:
        raise ValueError('Quantization scale not supported.')

    if bias is None or beta == 0:
        ratings = {}
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = sample(R[i, j], sample_prob)
        users, items = generate_users_items(ratings, m, n)
        P = np.ones(matrix.shape) * sample_prob
        return users, items, ratings, P, R, R_no_noise

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

        # HERE!
        for i in range(m):
            for j in range(n):
                if R[m][n] is None:
                    R[m][n] = 0.2 if matrix[m][n] <= 0.5 else 1
                    ratings[(m, n)] = R[m][n]
        return users, items, ratings, P, R, R_no_noise

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
        return users, items, ratings, P, R, R_no_noise

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
        return users, items, ratings, P, R, R_no_noise

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


def to_dataframe(ratings):
    ratings_dict = {
        'itemID': [],
        'userID': [],
        'rating': []
    }
    for key, value in ratings.items():
        if value is not None:
            userID, itemID = key
            ratings_dict['itemID'].append(itemID)
            ratings_dict['userID'].append(userID)
            ratings_dict['rating'].append(value)
    return pd.DataFrame(ratings_dict)


def generate_test_dataframe(R):
    test_dict = {'itemID': [],
                 'userID': [],
                 'rating': []}
    m, n = R.shape
    for i in range(m):
        for j in range(n):
            if random() > 0.99:
                test_dict['itemID'].append(j)
                test_dict['userID'].append(i)
                test_dict['rating'].append(R[i][j])
    return pd.DataFrame(test_dict)


def naive_propensity_estimation(ratings, shape, quantization='onetofive'):
    assert (quantization == 'onetofive'), 'Other quantizations not supported'
    proportion = np.zeros(5)
    count = 0
    for value in ratings.values():
        if value is not None:
            proportion[int(value / 0.2) - 1] += 1
            count += 1
    proportion /= proportion.sum()
    m, n = shape
    size = m * n
    result = np.zeros(shape) + count / size
    return result / result.mean()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(y)
        return y


def mlp_propensity_estimation(ratings):
    '''
    Not working yet.
    '''
    X = []
    y = []
    for (user, item), value in ratings.items():
        X.append([user, item])
        y.append(0 if value is None else 1)
    X = torch.tensor(X)
    y = torch.tensor(y)
    dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = MLP().to(device)
    learning_rate = 1e-3
    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for batch_no, (X_sample, y_sample) in enumerate(train_dataloader):
        pred = model(X_sample)
        loss = loss_fn(pred, y_sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_no == epochs:
            break
    return np.array(model.forward(X))


def masked_nb_propensity_estimation(truth, ratings, shape, beta=0):
    proportion = np.zeros(5)
    count = 0
    for value in ratings.values():
        if value is not None:
            proportion[int(value / 0.2) - 1] += 1
            count += 1
    proportion /= proportion.sum()
    m, n = shape
    size = m * n
    numerator = np.zeros(shape)
    for key, value in ratings.items():
        if value is not None:
            user, item = key
            numerator[user][item] = proportion[int(value / 0.2) - 1]
    numerator = numerator * count / size

    _, _, ratings_less_biased, P, R, R_no_noise = ground_truth_matrix_to_dataset(
        truth, quantization='binary', bias='full underlying', beta=beta, sample_prob=1)
    ratings_copy = deepcopy(ratings)

    for key, value in ratings.items():
        if value is not None and ratings_less_biased[key] is None:
            ratings_copy[key] = None

    proportion = np.zeros(5)
    for value in ratings_copy.values():
        if value is not None:
            proportion[int(value / 0.2) - 1] += 1
    proportion /= proportion.sum()

    # just check
    # _, _, ratings_ml100k, _, _, _ = ground_truth_matrix_to_dataset(
    #     truth, quantization='onetofive', bias=None)
    # proportion_truth = np.zeros(5)
    # for value in ratings_ml100k.values():
    #     if value is not None:
    #         proportion_truth[int(value / 0.2) - 1] += 1
    #         count += 1
    # proportion_truth /= proportion_truth.sum()
    # print(proportion_truth)
    # print(proportion)

    denominator = np.ones(shape)
    for (user, item), value in ratings.items():
        if value is not None:
            denominator[user][item] = proportion[int(value / 0.2) - 1]
    result = np.divide(numerator, denominator)
    result[result == 0] = 0.1
    return result


if __name__ == '__main__':
    truth = generate_ground_truth_matrix(
        (1000, 1000), environment='ml-100k-v1')
    users, items, ratings, P, R, R_no_noise = ground_truth_matrix_to_dataset(
        truth, quantization='onetofive', bias='full underlying', beta=5)
    df = to_dataframe(ratings)
    propensity = masked_nb_propensity_estimation(truth, ratings, P.shape)
    print(propensity)
    print(propensity.mean())
    print(propensity.max())
    print(propensity.min())
