from functools import partial, reduce
import itertools
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
import numpy as np
import operator
import os
import pandas as pd
import random
import scipy.sparse as sp
from scipy.stats import rankdata
import time


class ParameterGrid(object):
    """Grid of parameters with a discrete number of values for each.
    Can be used to iterate over parameter value combinations with the Python built-in function iter.
    Adapted from scikit-learn
    ----------
    param_grid : dict of string to sequence
        The parameter grid to explore, as a dictionary mapping estimator parameters to sequences of allowed values.
        An empty dict signifies default parameters.

    """

    def __init__(self, param_grid):

        self.param_grid = [param_grid]

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in itertools.product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)


class LFM:
    """Latent Factor Model (LFM) class, based on LightFM
    It must be initalized either with a set of interactions in sparse COO format, or with the path to the csv file with
    the ratings matrix (user x items)
    Interactions and weights are then processed and loaded as properties in accordance with LightFM expected structure
    """

    def __init__(self,
                 interactions=None,
                 weights=None,
                 path=None,
                 category=None,
                 use_weights=False,
                 **kwargs):
        """
        :param interactions: sparse COO matrix containing every user-item interaction
        :param weights: sparse COO matrix containing the weights of each user-item interaction
        :param path:(string) - path to the csv file to be processed - must be a user x item ratings matrix or a user:
        item interactions list
        :param type:(string) "interactions" or "ratings_matrix" to guide the data pre-processing
        :param kwargs: legacy, originally used to initialize Connector class - to be updated when dealing with db
        """
        # Check that the user has supplied some input, implicit or explicit
        if interactions is None and path is None:
            raise ValueError("The user must provide a path to a file or some interactions to instantiate the class")
        # Check that if a path has been supplied the variable type is valid
        if path is not None and (category != "interactions" and category != "ratings_matrix"):
            raise ValueError("The parameter 'category' must be provided and can only take one of two values: "
                             "'interactions' or 'ratings_matrix'")
        # Make sure the interactions and weights, if supplied, are in the correct format
        if interactions is not None and type(interactions) != sp.coo.coo_matrix:
            raise TypeError("The interactions object must be a scipy sparse COO matrix!")
        if weights is not None and type(weights) != sp.coo.coo_matrix:
            raise TypeError("The weights object must be a scipy sparse COO matrix!")

        self.best_model = None
        self.best_params = None
        self.best_score = 0
        self._category = category
        self._interactions = interactions
        self.mapping = None
        self.model = None
        self.path = path
        self._use_weights = use_weights
        self._weights = weights

    @property
    def interactions(self):
        # If interactions have not been supplied, process the file provided in source
        # N.B. This property also sets weights, which is probably not a best practice
        if self._interactions is None:

            if self._category == 'ratings_matrix':
                rm_df = pd.read_csv(self.path)
                ids = rm_df['sub']
                rm_df = rm_df.set_index(keys='sub')
                if 'Unnamed: 0' in rm_df.columns:
                    rm_df.drop('Unnamed: 0', axis=1, inplace=True)
                dataset = Dataset()
                dataset.fit(list(ids),
                            list(rm_df.columns))
                self.mapping = dataset.mapping()

                interactions = []

                for item in rm_df.columns.tolist():
                    users = rm_df.index[rm_df[item] >= 1].tolist()
                    counts = rm_df[item][rm_df[item] >= 1]
                    interactions.extend(zip(users, itertools.repeat(item, len(users)), counts))

                (self._interactions, self._weights) = dataset.build_interactions(interactions)

            else:
                int_df = pd.read_csv(self.path)
                if 'Unnamed: 0' in int_df.columns:
                    int_df.drop('Unnamed: 0', axis=1, inplace=True)
                int_df = int_df.groupby(['subscriber_id', 'ddi_block_id']).size().reset_index()\
                    .rename(columns={0:'count'})
                dataset = Dataset()
                ids = int_df['subscriber_id'].unique()
                items = int_df['ddi_block_id'].unique()
                dataset.fit(list(ids),
                            list(items))
                self.mapping = dataset.mapping()

                if self._use_weights:
                    interactions = zip(int_df['subscriber_id'], int_df['ddi_block_id'], int_df['count'])
                else:
                    interactions = zip(int_df['subscriber_id'], int_df['ddi_block_id'])
                (self._interactions, self._weights) = dataset.build_interactions(interactions)

        else:
            return self._interactions

    @interactions.setter
    def interactions(self, value):
        self._interactions = value

    @property
    def weights(self):
        if self._weights is None:
            pass # to be implemented
        else:
            return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def init_model(self,
                   no_components=10,
                   k=5,
                   n=10,
                   learning_schedule='adagrad',
                   loss='logistic',
                   learning_rate=0.05,
                   rho=0.95,
                   epsilon=1e-06,
                   item_alpha=0.0,
                   user_alpha=0.0,
                   max_sampled=10,
                   random_state=None):
        """
        Initialize model to be evaluated.
        :param no_components:(int, optional) – the dimensionality of the feature latent embeddings.
        :param k:(int, optional) – for k-OS training, the k-th positive example will be selected from the
               n positive examples sampled for every user.
        :param n:(int, optional) – for k-OS training, maximum number of positives sampled for each update.
        :param learning_schedule:(string, optional) – one of (‘adagrad’, ‘adadelta’).
        :param loss:(string, optional) – one of (‘logistic’, ‘bpr’, ‘warp’, ‘warp-kos’): the loss function.
        :param learning_rate:(float, optional) – initial learning rate for the adagrad learning schedule.
        :param rho:(float, optional) – moving average coefficient for the adadelta learning schedule.
        :param epsilon:(float, optional) – conditioning parameter for the adadelta learning schedule.
        :param item_alpha:(float, optional) – L2 penalty on item features.
        :param user_alpha:(float, optional) – L2 penalty on user features.
        :param max_sampled:(int, optional) – maximum number of negative samples used during WARP fitting.
        :param random_state:(int seed, RandomState instance, or None)
        """
        self.model = LightFM(no_components=no_components,
                             k=k,
                             n=n,
                             learning_schedule=learning_schedule,
                             loss=loss,
                             learning_rate=learning_rate,
                             rho=rho,
                             epsilon=epsilon,
                             item_alpha=item_alpha,
                             user_alpha=user_alpha,
                             max_sampled=max_sampled,
                             random_state=random_state)

    def train(self,
              interactions,
              partial=False,
              user_features=None,
              item_features=None,
              sample_weight=None,
              epochs=1,
              verbose=False):
        """
        Train the model in self.model either partially (i.e. for one epoch, starting from the last trained parameters)
        or till the end, for a maximum number of epochs
        :param interactions:(COO matrix, required) - set of training user-item interactions
        :param partial:(bool, optional) - fit the model partially if True, to completion otherwise
        :param user_features:(CSR matrix of shape [n_users, n_user_features], optional) - set of user features
        :param item_features:(CSR matrix of shape [n_items, n_item_features], optional) - set of item features
        :param sample_weight:(COO matrix, optional) - matrix with entries expressing weights of individual interactions
        :param epochs:(int, optional) - number of epochs for the training, only used if partial==False
        :param verbose:(bool, optional) – whether to print progress messages
        """
        if partial:
            self.model.fit_partial(interactions=interactions,
                                   user_features=user_features,
                                   item_features=item_features,
                                   sample_weight=sample_weight,
                                   epochs=1,
                                   verbose=verbose)
        else:
            self.model.fit(interactions=interactions,
                           user_features=user_features,
                           item_features=item_features,
                           sample_weight=sample_weight,
                           epochs=epochs,
                           verbose=verbose)

    @staticmethod
    def evaluate_model(model, metric, test, train):
        """
        Evaluate trained model on the test set, using one of the three available accuracy metrics
            AUC: the probability that a randomly chosen positive example has a higher score than a randomly chosen
            negative example.
            Precision: the fraction of known positives in the first k positions of the ranked list of results.
            Recall: the number of positive items in the first k positions of the ranked list of results divided by the
            number of positive items in the test period.
        :param model:(LightFM, required) - model to be evaluated
        :param metric:(string, required) - accuracy metric to be used, one of ['auc', 'precision', 'recall']
        :param test:(COO matrix, required) - known positives used to test the model
        :param train:(COO matrix, required) - training set; these interactions will be omitted from the score
               calculations to avoid re-recommending known positives.
        :return: test_score (float) - score computed on the test set
        """
        try:
            # make sure the metric is correct
            assert metric in ['auc', 'precision', 'recall']
            if metric == 'auc':
                test_score = auc_score(model, test, train).mean()
            elif metric == 'precision':
                test_score = precision_at_k(model, test, train, k=5).mean()
            else:
                test_score = recall_at_k(model, test, train, k=5).mean()
            return test_score
        except AssertionError:
            print('The metric provided is not correct or available!')

    def grid_search(self,
                    params,
                    metric='auc',
                    max_iterations=None,
                    max_epochs=50,
                    early_stopping=False,
                    use_weights=False):
        """
        Standard grid search method to select the hyper-parameters that result in the highest score on the test set.
        Uses ParameterGrid class from scikit-learn in order to create an iterable of all possible hyper-parameter
        combinations.
        The user can supply a max_iterations value that will stop the search once said number of combinations has been
        reached. Furthermore, early_stopping can be set to True to stop the training of a particular model when the test
        score has stopped improving, which is particularly useful when overfitting.
        :param params:(dict, required) - dictionary of parameters to test, {parameter: [list of values to try]}
        :param metric:(string, optional) - metric to use to pick the best model
        :param max_iterations:(int, optional) - if provided, the hyper-parameter optimization will stop after this many
               tests, irrespective of len(ParameterGrid(params))
        :param max_epochs:(int, optional) - max number of epochs to train each model
        :param early_stopping:(bool, optional) - if True, the training of a model will be partial and will stop after 5
               epochs of non-improvement on the test score; the model will then be re-trained using the optimal number
               of epochs
        :param use_weights:(bool, optional) - if True, the training procedure will use weights to value repeated
               interactions more
        """
        # Raise an error if any of the parameters supplied is not one of the arguments used by self.init_model
        valid_params = self.init_model.__code__.co_varnames
        if any([x not in valid_params for x in params.keys()]):
            raise ValueError("One of the hyper-parameters supplied is invalid. Please make sure there are no typos.")
        # Reset best values
        self.best_model = None
        self.best_params = None
        self.best_score = 0

        # Create train and test datasets
        (train_set, test_set) = random_train_test_split(self._interactions, test_percentage=0.2)
        # Since we cannot provide the same seed to random_train_test_split, using it on self._weights would generate a
        # set of weights that doesn't match train_set; we are thus forced to use the following convoluted procedure
        if use_weights and self._weights is not None:
            weights_csr = self._weights.tocsr()
            data = [weights_csr[u, i] for u, i in zip(train_set.row, train_set.col)]

            train_weights = sp.coo_matrix((data,
                                           (train_set.row,
                                            train_set.col)),
                                          shape=self._weights.shape,
                                          dtype=self._weights.dtype)
        else:
            train_weights = None

        # Create ParameterGrid instance to be iterated
        grid = ParameterGrid(params)
        # If max_iterations has not been provided then test all parameter combinations
        if not max_iterations:
            max_iterations = len(grid)
        # Turn grid from iterable to iterator
        grid = iter(grid)
        test_params = next(grid)
        test_params_idx = 1

        start_time = time.time()

        while test_params and test_params_idx <= max_iterations:
            # Initialize model with current combination of hyper-parameters to be tested
            self.init_model(**test_params)

            if early_stopping:
                best_iter = 0
                best_score = 0
                iters_no_improvement = 0
                # Train the model for max_epochs, evaluating it at each step
                for i in range(max_epochs):
                    self.train(train_set, sample_weight=train_weights, partial=True)
                    test_score = self.evaluate_model(self.model, metric, test_set, train_set)
                    if test_score > best_score:
                        best_iter = i+1
                        best_score = test_score
                        iters_no_improvement = 0
                    else:
                        iters_no_improvement += 1
                        # If the test score has not improved in the last 5 epochs stop the training
                        if iters_no_improvement == 5:
                            break
                # If the last epoch did not result in the highest test score, re-train the model for the optimal number
                # of epochs
                if best_iter != max_epochs:
                    self.init_model(**test_params)
                    self.train(train_set, sample_weight=train_weights, epochs=best_iter)
                    test_score = self.evaluate_model(self.model, metric, test_set, train_set)

            else:
                self.train(train_set, sample_weight=train_weights, epochs=max_epochs)
                test_score = self.evaluate_model(self.model, metric, test_set, train_set)

            # If the test score achieved by this model was the highest so far, set the class variables accordingly
            if test_score > self.best_score:
                self.best_model = self.model
                self.best_params = test_params
                self.best_score = test_score

            elapsed_time = (time.time() - start_time)/60

            print('Hyperparameters tested: {}/{}; {} score: {}; total time: {:.2f} minutes'.format(test_params_idx,
                                                                                                   max_iterations,
                                                                                                   metric,
                                                                                                   test_score,
                                                                                                   elapsed_time))

            test_params = next(grid)
            test_params_idx += 1

        print('The best model achieved a {} score of {} on the test set, with parameters {}'.format(metric,
                                                                                                    self.best_score,
                                                                                                    self.best_params))

    def randomized_search(self,
                          params,
                          metric='auc',
                          max_iterations=None,
                          max_epochs=50,
                          early_stopping=False,
                          use_weights=False):
        """
        Standard randomized search method to select the hyper-parameters that result in the highest score on the test
        set. Each iteration will sample one of the possible combinations of hyper-parameters.
        Uses ParameterGrid class from scikit-learn in order to create an iterable of all possible hyper-parameter
        combinations.
        The user can supply a max_iterations value that will stop the search once said number of combinations has been
        reached. Furthermore, early_stopping can be set to True to stop the training of a particular model when the test
        score has stopped improving, which is particularly useful when overfitting.
        :param params:(dict, required) - dictionary of parameters to test, {parameter: [list of values to try]}
        :param metric:(string, optional) - metric to use to pick the best model
        :param max_iterations:(int, optional) - if provided, the hyper-parameter optimization will stop after this many
               tests, irrespective of len(ParameterGrid(params))
        :param max_epochs:(int, optional) - max number of epochs to train each model
        :param early_stopping:(bool, optional) - if True, the training of a model will be partial and will stop after 5
               epochs of non-improvement on the test score; the model will then be re-trained using the optimal number
               of epochs
        :param use_weights:(bool, optional) - if True, the training procedure will use weights to value repeated
               interactions more
        """
        # Raise an error if any of the parameters supplied is not one of the arguments used by self.init_model
        valid_params = self.init_model.__code__.co_varnames
        if any([x not in valid_params for x in params.keys()]):
            raise ValueError("One of the hyper-parameters supplied is invalid. Please make sure there are no typos.")

        # Reset best values
        self.best_model = None
        self.best_params = None
        self.best_score = 0

        # create train and test datasets
        (train_set, test_set) = random_train_test_split(self._interactions, test_percentage=0.2)
        if use_weights and self._weights is not None:
            weights_csr = self._weights.tocsr()
            data = [weights_csr[u, i] for u, i in zip(train_set.row, train_set.col)]

            train_weights = sp.coo_matrix((data,
                                          (train_set.row,
                                           train_set.col)),
                                          shape=self._weights.shape,
                                          dtype=self._weights.dtype)
        else:
            train_weights = None

        # Create ParameterGrid instance to be iterated and cast it to list
        grid = list(ParameterGrid(params))
        # If max_iterations has not been provided then test all parameter combinations
        if not max_iterations:
            max_iterations = len(grid)
        # Shuffle the list and pop out and remove the last element
        random.shuffle(grid)
        test_params = grid.pop()
        test_params_idx = 1

        start_time = time.time()

        while test_params and test_params_idx <= max_iterations:
            # Initialize model with current combination of hyper-parameters to be tested
            self.init_model(**test_params)

            if early_stopping:
                best_iter = 0
                best_score = 0
                iters_no_improvement = 0
                # Train the model for max_epochs, evaluating it at each step
                for i in range(max_epochs):
                    self.train(train_set, sample_weight=train_weights, partial=True)
                    test_score = self.evaluate_model(self.model, metric, test_set, train_set)
                    if test_score > best_score:
                        best_iter = i+1
                        best_score = test_score
                        iters_no_improvement = 0
                    else:
                        iters_no_improvement += 1
                        # If the test score has not improved in the last 5 epochs stop the training
                        if iters_no_improvement == 5:
                            break

                # If the last epoch did not result in the highest test score, re-train the model for the optimal number
                # of epochs
                if best_iter != max_epochs:
                    self.init_model(**test_params)
                    self.train(train_set, sample_weight=train_weights, epochs=best_iter)
                    test_score = self.evaluate_model(self.model, metric, test_set, train_set)

            else:
                self.train(train_set, sample_weight=train_weights, epochs=max_epochs)
                test_score = self.evaluate_model(self.model, metric, test_set, train_set)

            # If the test score achieved by this model was the highest so far, set the class variables accordingly
            if test_score > self.best_score:
                self.best_model = self.model
                self.best_params = test_params
                self.best_score = test_score

            random.shuffle(grid)
            if grid:
                test_params = grid.pop()
            else:
                test_params = None

            elapsed_time = (time.time() - start_time)/60

            print('Hyperparameters tested: {}/{}; {} score: {}; total time: {:.2f} minutes'.format(test_params_idx,
                                                                                                   max_iterations,
                                                                                                   metric,
                                                                                                   test_score,
                                                                                                   elapsed_time))
            test_params_idx += 1

        print('The best model achieved a {} score of {} on the test set, with parameters {}'.format(metric,
                                                                                                    self.best_score,
                                                                                                    self.best_params))

    def predict(self,
                users,
                items,
                what='scores',
                item_features=None,
                user_features=None):
        """
        Prediction method: once the best hyper-parameters have been selected and the resulting model has been trained,
        this can be used the get predictions for a (sub)set of users. The predictions can be in the form of absolute
        scores on in terms of ranks, 1 being the highest.
        :param users:(np.int32 array of shape [n_pairs,], required) - user ids for whom we want predictions
        :param items:(np.int32 array of shape [n_pairs,], required) - item ids for which we want predictions
        :param what:(string, optional) - must be 'scores' or 'ranks'
        :param user_features:(CSR matrix of shape [n_users, n_user_features], optional) - set of user features
        :param item_features:(CSR matrix of shape [n_items, n_item_features], optional) - set of item features
        :return:(np.float32 array of shape [n_pairs,]): Numpy array containing the recommendation scores for pairs
                defined by the inputs.
        """
        # check the training has been done and we have picked a best model
        if self.best_model:
            scores = np.empty(shape=(len(users), len(items)))
            for i, user in enumerate(users):
                scores[i] = self.best_model.predict(user_ids=user,
                                                    item_ids=items,
                                                    user_features=user_features,
                                                    item_features=item_features)
            if what == 'scores':
                return scores
            elif what == 'ranks':
                ranks = np.empty_like(scores)
                for i, score in enumerate(scores):
                    ranks[i] = len(score) + 1 - rankdata(score).astype(int)
                return ranks
            else:
                print("The parameter 'how' can only be 'scores' or 'ranks'")
        else:
            print('The model has not been trained yet!')
