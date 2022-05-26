import numpy as np
import pandas as pd
from random import random
import reclab


def generate_ground_truth_matrix(dimensions, distribution='uniform', quantization_method='binary'):
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

    if distribution == 'uniform':
        if quantization_method == 'binary':
            return np.random.randint(2, size=dimensions)

    elif distribution == 'normal':
        if quantization_method == 'binary':
            raise ValueError
    pass


def ground_truth_matrix_to_dataset(matrix, sample_prob=0.1, bias=None, shuffle=True):
    """
    Converts a ground truth matrix to a recommender dataset.

    Parameters:
        matrix: a mxn  matrix that contains the ground truth for each pair of m users and n items.

    Returns:

    """
    m, n = matrix.shape

    if shuffle:
        np.random.shuffle(matrix)

    for i in range(m):
        for j in range(n):
            if random() > sample_prob:
                matrix[i, j] = None

    if bias is None:
        users = {}
        items = {}
        ratings = {}
        for i in range(m):
            users[i] = matrix[i]
        for i in range(n):
            items[n] = matrix[:, i]
        for i in range(m):
            for j in range(n):
                ratings[(i, j)] = matrix[i, j]
    pass


if __name__ == '__main__':
    print(generate_ground_truth_matrix((10, 10)))
