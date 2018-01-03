import pickle as pkl
import argparse
import os

import numpy as np
import peakutils
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis

from config import FEATURES_PATH
from utils.features_extraction_helper import peak_features_paper, dtw_distance_one_vs_all_parallel
from utils.processing_helper import save_features


def get_peaks_indexes(series):
    series = gaussian_filter(series, sigma=1)
    peaks = peakutils.indexes(-series, thres=.2, min_dist=80)
    return peaks


def get_peak_differences(peaks_indexes):
    peaks_indexes.sort()
    return peaks_indexes[1:] - peaks_indexes[:-1]


def get_stats(series):
    if len(series) == 0: return [0] * 8
    stats = [np.mean(series), np.std(series), kurtosis(series)]
    stats.extend(np.percentile(series, [10, 25, 50, 75, 90]))
    return stats


def get_series_peak_features(series, peaks):
    feats = []

    feats.extend(get_stats(peaks))
    feats.extend(get_stats(np.abs(series[peaks])))
    feats.extend(get_stats(get_peak_differences(peaks)))

    return feats


def peak_features(data):
    median = np.median(data, axis=1)
    std = np.std(data, axis=1)

    features = []
    for i, series in enumerate(data):
        feats = []
        peaks = get_peaks_indexes(series)

        # Removing peaks above median
        peaks = peaks[series[peaks] < median[i]]

        feats.extend(get_series_peak_features(series, peaks))

        # Stats of peaks above median-std
        peaks_above_std = peaks[series[peaks] > (median[i] - std[i])]
        feats.extend(get_series_peak_features(series, peaks_above_std))

        # Stats of peaks between median-std and median-2std
        peaks = peaks[series[peaks] <= median[i] - std[i]]
        peaks_between_std_2std = peaks[series[peaks] > median[i] - 2 * std[i]]
        feats.extend(get_series_peak_features(series, peaks_between_std_2std))

        # Stats of peaks between median-2std and median-3std
        peaks = peaks[series[peaks] <= median[i] - 2 * std[i]]
        peaks_between_2std_3std = peaks[series[peaks] > median[i] - 3 * std[i]]
        feats.extend(get_series_peak_features(series, peaks_between_2std_3std))

        # Stats of remaining peaks
        peaks = peaks[series[peaks] <= median[i] - 3 * std[i]]
        feats.extend(get_series_peak_features(series, peaks))

        features.append(feats)
        if i % 10 == 0: sys.stdout.write('.')
        sys.stdout.flush()
    return np.array(features)


def get_dtw_features(data):
    """
    Extracts the Dynamic Time Warping time distance between the base templates mentioned and all other row
    """
    features = []
    templates_data = pkl.load(open('data/templates.pkl', 'r'))

    for i, template in templates_data.iteritems():
        print(' - Processing with template index %d' % i)
        features.append(dtw_distance_one_vs_all_parallel(data, template))
    return np.hstack(features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="Generate test features if this is set", action='store_true')
    args = parser.parse_args()

    if args.test:
        detrend_median41 = np.load(os.path.join(FEATURES_PATH, 'test', 'detrend_median41.npy'))
    else:
        detrend_median41 = np.load(os.path.join(FEATURES_PATH, 'detrend_median41.npy'))

    print(' - Extracting peak features')
    detrend_features = peak_features(detrend_median41)
    save_features(detrend_features, 'peak_features', args.test)

    print(' - Extracting peak features from paper')
    detrend_features_paper = peak_features_paper(detrend_median41)
    save_features(detrend_features_paper, 'peak_features_paper', args.test)

    indexes = [0, 6, 10, 11, 17, 28]
    if not args.test:
        y_ = {i: detrend_median41[i] for i in indexes}
        pkl.dump(y_, open('data/templates.pkl', 'w'))

    print(' - Extracting DTW features')
    feats = get_dtw_features(detrend_median41)
    save_features(feats, 'dtw_features', args.test)
