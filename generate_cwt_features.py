import os
import argparse

import numpy as np
from pywt import cwt

from config import FEATURES_PATH
from utils.processing_helper import parallelize_row, save_features


def generate_cwt_features(x_data):
    return np.array([cwt(series, 2, 'cmor')[0].flatten() for series in x_data])


def generate_cwt_features_parallel(data, n_jobs=-1):
    return parallelize_row(data, generate_cwt_features, n_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    if args.test:
        raw_mean_std_normalized = np.load(os.path.join(FEATURES_PATH, 'test', 'raw_mean_std_normalized.npy'))
    else:
        raw_mean_std_normalized = np.load(os.path.join(FEATURES_PATH, 'raw_mean_std_normalized.npy'))

    print ' - Processing CWT Features'
    cwt_scale_2 = generate_cwt_features(raw_mean_std_normalized)
    save_features(cwt_scale_2.real, 'cwt_features_scale2_real', args.test)
    save_features(cwt_scale_2.imag, 'cwt_features_scale2_imag', args.test)
