import datetime
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd

import hyperopt
import torch
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from spotlight.cross_validation import (random_train_test_split,
                                        user_based_train_test_split)
from spotlight.evaluation import precision_recall_score, sequence_mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.torch_utils import set_seed

if __name__ == "__main__":
    CUDA = torch.cuda.is_available()
    FILE_PATH = '../../data/'


def hyperparameter_space():
    """Define hyperopt hyperparameter space"""

    space = {
        'batch_size': hp.quniform('batch_size', 128, 384, 16),
        'learn_rate': hp.loguniform('learn_rate', -6, -3),
        'l2': hp.loguniform('l2', -25, -9),
        'n_iter': hp.quniform('n_iter', 5, 10, 1),
        'loss': hp.choice('loss', ['adaptive_hinge', 'pointwise', 'bpr', 'hinge',]),
        'embedding_dim': hp.quniform('embedding_dim', 16, 128, 8),
        'representation': hp.choice('representation', ['cnn', 'lstm',])
    }

    return space


def get_objective(train, valid, test, random_state=None):

    def objective(space):
        """Objective function for Spotlight ImplicitFactorizationModel"""

        batch_size = int(space['batch_size'])
        embedding_dim = int(space['embedding_dim'])
        l2 = space['l2']
        learn_rate = space['learn_rate']
        loss = space['loss']
        n_iter = int(space['n_iter'])
        representation = space['representation']

        model = ImplicitSequenceModel(
            loss=loss,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            representation=representation,
            learning_rate=learn_rate,
            n_iter=n_iter,
            l2=l2,
            use_cuda=CUDA)

        start = time.clock()

        try:
            model.fit(train, verbose=True)
        except ValueError:
            elapsed = time.clock() - start
            return {'loss': 0.0,
                    'status': STATUS_FAIL,
                    'validation_mrr': 0.0,
                    'test_mrr': 0.0,
                    'elapsed': elapsed,
                    'hyper': space}
        elapsed = time.clock() - start
        print(model)

        validation_mrr = sequence_mrr_score(model, valid).mean()
        test_mrr = sequence_mrr_score(model, test).mean()

        print('MRR {} {}'.format(validation_mrr, test_mrr))

        if np.isnan(validation_mrr):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        return {'loss': -validation_mrr,
                'status': status,
                'validation_mrr': validation_mrr,
                'test_mrr': test_mrr,
                'elapsed': elapsed,
                'hyper': space}
    return objective


def optimize(objective, space, trials_fname=None, max_evals=5):

    if trials_fname is not None and os.path.exists(trials_fname):
        with open(trials_fname, 'rb') as trials_file:
            trials = pickle.load(trials_file)
    else:
        trials = Trials()

    fmin(objective,
         space=space,
         algo=tpe.suggest,
         trials=trials,
         max_evals=max_evals)

    if trials_fname is not None:
        temporary = '{}.temp'.format(trials_fname)
        with open(temporary, 'wb') as trials_file:
            pickle.dump(trials, trials_file)
        shutil.move(temporary, trials_fname)

    return trials


def summarize_trials(trials):
    results = trials.trials

    results = sorted(results, key=lambda x: -x['result']['validation_mrr'])

    if results:
        print('Best: {}'.format(results[0]['result']))

    results = sorted(results, key=lambda x: -x['result']['test_mrr'])

    if results:
        print('Best test MRR: {}'.format(results[0]['result']))


def main(max_evals):
    status = 'available' if CUDA else 'not available'
    print("CUDA is {}!".format(status))

    # Fix random_state
    seed = 42
    set_seed(seed)
    random_state = np.random.RandomState(seed)

    max_sequence_length = 15
    min_sequence_length = 2
    step_size = 1

    df = pd.read_csv(FILE_PATH)
    if 'time_of_day' in df.columns:
        df = df.drop(columns=['time_of_day', 'time_of_year', 'is_content_block'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0', 'js_key'])

    sub_col = 'subscriber_id'
    block_col = 'ddi_id'
    time_col = 'entry_at'

    # preprocess dataframe
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(by=time_col, inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)

    # create idx mapping compatible with spotlight, map users and items
    sub_mapping = {k:v for v, k in enumerate(df[sub_col].unique())}
    block_mapping = {k:v for v, k in enumerate(df[block_col].unique(), 1)}
    df['user_id'] = df[sub_col].map(sub_mapping)
    df['item_id'] = df[block_col].map(block_mapping)

    # create dataset using interactions and timestamps
    dataset = Interactions(user_ids=np.array(df['user_id'], dtype='int32'),
                           item_ids=np.array(df['item_id'], dtype='int32'),
                           timestamps=df[time_col])

    # create training, validation and test sets using a 80/10/10 split
    train, rest = user_based_train_test_split(
        dataset,
        test_percentage=0.2,
        random_state=random_state)
    test, valid = user_based_train_test_split(
        rest,
        test_percentage=0.5,
        random_state=random_state)
    # convert to sequences
    train = train.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)
    test = test.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)
    valid = valid.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)

    print('data: {}'.format(train))

    dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = './experiment_{}.pickle'.format(dtime)
    objective = get_objective(train, valid, test, random_state)
    space = hyperparameter_space()

    trials = optimize(objective,
                      space,
                      trials_fname=fname,
                      max_evals=max_evals)

    summarize_trials(trials)

    return trials
