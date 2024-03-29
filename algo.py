from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
                        
from data_utils import generate_ground_truth_matrix, ground_truth_matrix_to_dataset, to_dataframe, generate_test_dataframe, masked_nb_propensity_estimation, naive_propensity_estimation, mixing_mf_dataset
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.prediction_algorithms import AlgoBase
from surprise.prediction_algorithms import SVD as surprise_SVD
from surprise.prediction_algorithms import PredictionImpossible
from surprise.utils import get_rng
from copy import deepcopy

class SVD(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state

        AlgoBase.__init__(self)

    def fit(self, trainset, verbose=False):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset, verbose=verbose)

        return self

    def sgd(self, trainset, verbose=False):

        u, i, f = 0, 0, 0
        r, err, dot, puf, qif = 0., 0., 0., 0., 0.
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

            self.bu = bu
            self.bi = bi
            self.pu = pu
            self.qi = qi

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est


class PropensitySVD(AlgoBase):

    def __init__(self, p, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None):

        if p is None:
            assert False, 'Please use standard SVD'
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.p = p

        AlgoBase.__init__(self)

    def fit(self, trainset, verbose=False):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset, verbose=verbose)

        return self

    def sgd(self, trainset, verbose=False):

        u, i, f = 0, 0, 0
        r, err, dot, puf, qif = 0., 0., 0., 0., 0.
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u]) / self.p[u][i]
                    bi[i] += lr_bi * (err - reg_bi * bi[i]) / self.p[u][i]

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf) / self.p[u][i]
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif) / self.p[u][i]
 
            self.bu = bu
            self.bi = bi
            self.pu = pu
            self.qi = qi  

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est



if __name__ == '__main__':
    truth = generate_ground_truth_matrix(
        (1000, 1000), environment='ml-100k-v1')
    # users, items, ratings, P, R, R_no_noise = ground_truth_matrix_to_dataset(
    #     truth, quantization='onetofive', bias='full underlying', beta=1)
    users, items, ratings, P, R, R_no_noise = mixing_mf_dataset(
    truth, quantization='onetofive', bias='full underlying', beta=1)
    df = to_dataframe(ratings)
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(
        df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    experiment = 2

    if experiment == 1:
        p = masked_nb_propensity_estimation(truth, ratings, P.shape, beta=3)
        p_naive = naive_propensity_estimation(ratings, P.shape)
        algo_better = PropensitySVD(p, n_epochs=10)
        algo = PropensitySVD(p_naive, n_epochs=10)
        algo_worst = SVD(n_epochs=10)
        algo_better.fit(trainset, verbose=True)
        # algo.fit(trainset, verbose=True)
        # algo_worst.fit(trainset, verbose=True)
        test_df = generate_test_dataframe(R_no_noise)
        testset = Dataset.load_from_df(
            test_df[['userID', 'itemID', 'rating']], reader).build_full_trainset().build_testset()

        predictions = algo_better.test(testset)
        print(accuracy.rmse(predictions))
        print(accuracy.mae(predictions))

        # predictions = algo.test(testset)
        # print(accuracy.rmse(predictions))
        # print(accuracy.mae(predictions))

        # predictions = algo_worst.test(testset)
        # print(accuracy.rmse(predictions))
        # print(accuracy.mae(predictions))

    elif experiment == 2:
        algo = surprise_SVD(n_epochs=10, verbose=True)    
        algo.fit(trainset)
        test_df = generate_test_dataframe(R_no_noise)
        testset = Dataset.load_from_df(
            test_df[['userID', 'itemID', 'rating']], reader).build_full_trainset().build_testset()
        predictions = algo.test(testset)
        print(accuracy.rmse(predictions))
        print(accuracy.mae(predictions))    
        
