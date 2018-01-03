import multiprocessing
import os
import pickle as pkl
from multiprocessing import Pool

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from config import FEATURES_PATH, DATASETS_PATH, FOLDS_FILENAME

identity = lambda x: x


class SimpleTransform(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.transformer(X)


def generate_dataset(struct, dataset_name, test=False):
    """
    Generate the dataset from list of features and target variable
    :param struct: dict containing information of features and target variable
    :param dataset_name: name of the dataset
    :param test: if True then target values are not required in struct
    """
    features = struct['features']

    if test:
        dataset_name = 'test/%s' % dataset_name
        features = [('test/%s' % feature_name, feature_transform) for feature_name, feature_transform in features]

    # Processing features
    features_numpy = []
    for i, (feature, transformer) in enumerate(features):
        x_all = np.load(os.path.join(FEATURES_PATH, '%s.npy' % feature))
        x_all = transformer.fit_transform(x_all)

        features_numpy.append(x_all)

    dataset = np.hstack(features_numpy)

    dataset_filename = os.path.join(DATASETS_PATH, '%s_X.npy' % dataset_name)

    make_dir_if_not_exists(os.path.dirname(dataset_filename))
    dataset.dump(dataset_filename)

    if not test:
        target_feature_name, target_transformer = struct['target']
        # Processing the target variable
        target_values = np.load(os.path.join(FEATURES_PATH, '%s.npy' % target_feature_name))
        target_values = target_transformer.fit_transform(target_values)
        target_values.dump(os.path.join(DATASETS_PATH, '%s_y.npy' % dataset_name))


def load_dataset(dataset_name):
    """
    Loads the dataset from the dataset folds given the dataset name
    :param dataset_name
    :return: dataset_X, dataset_y
    """
    X = np.load(os.path.join(DATASETS_PATH, '%s_X.npy' % dataset_name))
    y = np.load(os.path.join(DATASETS_PATH, '%s_y.npy' % dataset_name))
    return X, y


def load_testdata(dataset_name):
    """
    Loads test dataset
    """
    return np.load(os.path.join(DATASETS_PATH, 'test/%s_X.npy' % dataset_name))


def save_features(data, features_name, test=False):
    """
    Save the features in the features folder
    :param data: numpy array or pandas dataframe with samples x features format
    :param features_name: name of the filename
    :param test: True if to save features in test folder
    """
    if not isinstance(data, np.ndarray):
        data = data.values
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if test:
        feature_filename = os.path.join(FEATURES_PATH, 'test/%s.npy' % features_name)
    else:
        feature_filename = os.path.join(FEATURES_PATH, '%s.npy' % features_name)
    make_dir_if_not_exists(os.path.dirname(feature_filename))
    data.dump(feature_filename)


def features_exists(feature_name, test=False):
    """
    Checks if feature/s exists with given name
    """
    if test:
        return os.path.exists(os.path.join(FEATURES_PATH, 'test', feature_name + '.npy'))
    else:
        return os.path.exists(os.path.join(FEATURES_PATH, feature_name + '.npy'))


def make_dir_if_not_exists(dir_name):
    """
    Makes directory if does not exists
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_folds():
    """
    Loads the k-folds split of the dataset for cross-validation
    :return: list of (train_split, test_split)
    """
    return pkl.load(open(FOLDS_FILENAME, 'r'))


def parallelize_row(data, func, n_jobs):
    """
    To speed up data preprocessing
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    if n_jobs > 1:
        data_split = np.array_split(data, n_jobs)
        pool = Pool(n_jobs)
        data = np.concatenate(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data
    else:
        return func(data)
