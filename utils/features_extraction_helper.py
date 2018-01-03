import pickle as pkl
import numpy as np
import sys
from PyAstronomy import pyasl
from fastdtw import fastdtw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import PolynomialFeatures
from utils.processing_helper import parallelize_row


def remove_outlier(series, points=80, count=70, deg=3, stdlim=3, trend_stdlim=4):
    """
    Hybrid outlier removal algorithm
    """
    series = np.copy(series)
    n = series.shape[0]
    trend = gaussian_filter1d(series, sigma=100)
    std = series.std()

    outliers = pyasl.slidingPolyResOutlier(np.arange(n), series, points=points, count=count, deg=deg, stdlim=stdlim)[1]

    outliers_filtered = np.where(series[outliers] > trend[outliers] + trend_stdlim * std)[0]
    outliers = outliers[outliers_filtered]

    for outlier in outliers:
        series[outlier] = np.median(series[outlier - 20:outlier + 20])

    return series


def remove_outlier_all(x_values):
    """
    Remove outlier for each row in x_values
    """
    result = []
    for series in x_values:
        result.append(remove_outlier(series))
        sys.stdout.write('.')
        sys.stdout.flush()
    return np.array(result)


def remove_outlier_parallel(x_values, n_jobs=-1):
    """
    Remove outliers in parallel
    """
    return parallelize_row(x_values, remove_outlier_all, n_jobs=n_jobs)


def min_max_normalize(data):
    """
    Normalizes the vector with min-max and makes the changes inplace
    """
    data[:] = (data - data.min()) / (data.max() - data.min())
    return data


def dtw_distance_one_vs_all(data):
    """
    Calculates the Dynamic Time Warping distance from a base template example
    """
    global y
    n, _ = data.shape
    dist = np.zeros(shape=n)
    for i in xrange(n):
        dist[i], _ = fastdtw(data[i], y, dist=euclidean)
        sys.stdout.write('.')
        sys.stdout.flush()
    return dist.reshape(-1, 1)


def dtw_distance_one_vs_all_parallel(data, template_data):
    global y
    y = template_data
    data = parallelize_row(data, dtw_distance_one_vs_all, n_jobs=-1)
    print
    return data


def peak_features_paper(x_dataset):
    """
    Extracts the features mentioned in https://pdfs.semanticscholar.org/8c96/1f7357fbc3e22e0e5417744af24662222818.pdf
    """
    mean = np.mean(x_dataset, axis=1)
    std_dev = np.std(x_dataset, axis=1)

    mean_minus_std = (mean - std_dev).reshape(-1, 1)
    mean_minus_2std = (mean - 2 * std_dev).reshape(-1, 1)

    points_mean_std_between = np.logical_and(x_dataset < mean_minus_std, x_dataset > mean_minus_2std).sum(axis=1)
    points_mean_std_below = (x_dataset < mean_minus_2std).sum(axis=1)
    points_above_mean = (x_dataset > mean.reshape(-1, 1)).sum(axis=1)

    features = [mean, std_dev, points_mean_std_between, points_mean_std_below, points_above_mean]
    features = [min_max_normalize(feature) for feature in features]

    features_ = np.column_stack(features)

    # Create PolynomialFeatures object with interaction_only set to true
    features_dataset = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(features_)
    return features_dataset
