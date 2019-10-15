from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import torch
from sklearn.cluster import KMeans
from spotlight.cross_validation import (random_train_test_split,
                                        user_based_train_test_split)
from spotlight.evaluation import precision_recall_score, sequence_mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.torch_utils import set_seed


def create_dataframe():

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
    user_mapping = {k:v for v, k in enumerate(df[sub_col].unique())}
    item_mapping = {k:v for v, k in enumerate(df[block_col].unique(), 1)}
    df['user_id'] = df[sub_col].map(user_mapping)
    df['item_id'] = df[block_col].map(item_mapping)

    return (df, user_mapping, item_mapping)


def train_model(df, hyperparams):
    # Fix random_state
    seed = 42
    set_seed(seed)
    random_state = np.random.RandomState(seed)

    max_sequence_length = 15
    min_sequence_length = 2
    step_size = 1

    # create dataset using interactions dataframe and timestamps
    dataset = Interactions(user_ids=np.array(df['user_id'], dtype='int32'), 
                           item_ids=np.array(df['item_id'], dtype='int32'), 
                           timestamps=df['entry_at'])

    # create training and test sets using a 80/20 split
    train, test = user_based_train_test_split(
        dataset,
        test_percentage=0.2,
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

    print('data: {}'.format(train))

    # initialize and train model
    model = ImplicitSequenceModel(
            **hyperparams,
            use_cuda=CUDA,
            random_state=random_state)
    model.fit(train, verbose=True)

    # compute mrr score on test set
    test_mrr = sequence_mrr_score(model, test).mean()
    print('MRR score on test set: {}'.format(test_mrr))

    return model


def individual_predictions(df, model):
    num_users = len(df['user_id'].unique())
    num_items = len(df['item_id'].unique())
    predictions = np.zeros(shape=(num_users, num_items+1))

    dataset = Interactions(user_ids=np.array(df['user_id'], dtype='int32'),
                           item_ids=np.array(df['item_id'], dtype='int32'),
                           timestamps=df['entry_at'])
    sequences = dataset.to_sequence(max_sequence_length=15)

    user_id = 0

    for user, sequence in zip(sequences.user_ids, sequences.sequences):
        if user == user_id:
            predictions[user] = model.predict(sequence)
            user_id += 1

    return predictions


def group_predictions(ind_predictions, user_mapping, item_mapping):
    kmeans_params = {'n_clusters':50,
                 'n_init':5,
                 'max_iter':50,
                 'precompute_distances':True,
                 'random_state':42,
                 'verbose':1}

    kmeans = KMeans(**kmeans_params)
    print("Fitting {} K-Means models...".format(kmeans_params['n_init']))
    clusters = kmeans.fit_predict(ind_predictions)

    clusters_to_users = defaultdict(list)

    for user, cluster in enumerate(clusters):
        clusters_to_users[cluster].append(user)

    clusters_to_recommendations = {}
    n = 20

    for cluster in clusters_to_users.keys():
        cluster_predictions = ind_predictions[clusters_to_users[cluster], :]
        avg_predictions = np.median(cluster_predictions, axis=0)
        ranks = len(avg_predictions) + 1 - rankdata(avg_predictions).astype(int)
        recommendations = [np.where(ranks == k)[0][0] for k in range(1, n+1)]
        clusters_to_recommendations[cluster] = recommendations

    df_removed = pd.read_csv(EXCLUDE_PATH)
    removed_ddi_blocks = list(df_removed['0'])
    removed_ids = [item_mapping[x] for x in removed_ddi_blocks]

    k = 5
    for cluster, recommendations in clusters_to_recommendations.items():
        topk = []
        i = 0
        while len(topk) < k:
            if recommendations[i] not in removed_ids:
                topk.append(recommendations[i])
            i += 1
        clusters_to_recommendations[cluster] = topk

    return (clusters, clusters_to_recommendations)


if __name__ == "__main__":
    hyperparams = {'batch_size': 352,
               'embedding_dim': 128,
               'l2': 5.310207879958108e-07,
               'learning_rate': 0.0025011952345768613,
               'loss': 'adaptive_hinge',
               'n_iter': 5,
               'representation': 'lstm'}

    CUDA = torch.cuda.is_available()
    FILE_PATH = '../../data/.csv'
    EXCLUDE_PATH = '../../data/.csv'

    df, user_mapping, item_mapping = create_dataframe()
    model = train_model(df=df, hyperparams=hyperparams)
    individual_scores = individual_predictions(df=df, model=model)
    clusters, clusters_to_recommendations = group_predictions(ind_predictions=individual_scores,
                                                            user_mapping=user_mapping,
                                                            item_mapping=item_mapping)


    idx_to_userid = {v: k for k, v in user_mapping.items()}
    idx_to_block = {v: k for k, v in item_mapping.items()}

    # create user to cluster dataframe
    df_clusters = pd.DataFrame(clusters)
    df_clusters.reset_index(inplace=True)
    df_clusters.rename({'index':'User ID', 0:'Group ID'}, axis=1, inplace=True)
    df_clusters['User ID'] = df_clusters['User ID'].map(idx_to_userid)
    df_clusters.to_csv('./user_groups_20190813.csv', index=False)

    # create cluster to recommendations dataframe
    df_recommendations = pd.DataFrame.from_dict(clusters_to_recommendations, orient='index')
    df_recommendations.reset_index(inplace=True)
    df_recommendations.rename({'index':'Group ID'}, axis=1, inplace=True)
    df_recommendations.sort_values(by='Group ID', inplace=True)
    for col in range(5):
        df_recommendations[col] = df_recommendations[col].map(idx_to_block)
    df_recommendations.rename({0:'First', 1:'Second', 2:'Third', 3:'Fourth', 4:'Fifth'}, axis=1, inplace=True)
    df_recommendations.to_csv('./group_recommendations_20190813.csv', index=False)
