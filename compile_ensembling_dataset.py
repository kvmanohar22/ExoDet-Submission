import argparse

import numpy as np
from utils.processing_helper import generate_dataset, SimpleTransform
from sklearn.base import BaseEstimator, TransformerMixin


class ReshapeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, reshape_size=(-1, 1)):
        self.reshape_size = reshape_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(self.reshape_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    struct = {
        'features': [
            # ('probability_results/wavelet_db2_b_dataset_xgb', ReshapeTransform()),
            ('probability_results/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_tune', ReshapeTransform()),
            # ('probability_results/raw_normalized_smoothed_dataset_cnn_wrapper_2d_tune', ReshapeTransform()),
            ('probability_results/raw_normalized_smoothed_dataset_cnn_wrapper_window_slicing_size1000', ReshapeTransform()),
            # ('probability_results/detrend_gaussian10_dataset_edited_nn_pca_xgb_tune', ReshapeTransform()),
            ('probability_results/fft_smoothed10_dataset_xgb_tune', ReshapeTransform()),
            ('probability_results/fft_smoothed10_dataset_edited_nn_pca_xgb_tune', ReshapeTransform()),
            ('probability_results/fft_smoothed10_dataset_cnn_wrapper_1d_half', ReshapeTransform()),
            # ('probability_results/fft_normalized_dataset_onesided_pca_xgb_tune', ReshapeTransform()),
            # ('probability_results/fft_smoothed10_dataset_lle_xgb_tune', ReshapeTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'ensemble_dataset_dummy', test=args.test)
    
    struct = {
        'features': [
            ('probs/fft_smoothed10_ar100_dataset_edited_nn_xgb_fft10ar100', ReshapeTransform()),
            ('probs/fft_smoothed10_ar100_dataset_rfecv_xgb', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_1d_half_no_rolling_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_fft', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_edited_nn_pca_xgb_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_xgb_fft10', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_rng50', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_window_slicing_2500_rng50', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_xgb_rng50', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_wrapper_2d_rns', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_window_slicing_2d_rns', ReshapeTransform()),
            ('probs/raw_time_series_peak_dataset_rfecv_xgb', ReshapeTransform()),
            ('probs/wavelet_db2_b_dataset_xgb', ReshapeTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'ensembling_dataset_dummy', test=args.test)

    struct = {
        'features': [
            ('probs/fft_smoothed10_ar100_dataset_edited_nn_xgb_fft10ar100', ReshapeTransform()),
            ('probs/fft_smoothed10_ar100_dataset_rfecv_xgb', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_1d_half_no_rolling_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_fft', ReshapeTransform()),
            # ('probs/fft_smoothed10_dataset_edited_nn_pca_xgb_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_xgb_fft10', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_rng50', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_window_slicing_2500_rng50', ReshapeTransform()),
            # ('probs/raw_normalized_gaussian50_dataset_xgb_rng50', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_wrapper_2d_rns', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_window_slicing_2d_rns', ReshapeTransform()),
            ('probs/raw_time_series_peak_dataset_rfecv_xgb', ReshapeTransform()),
            ('probs/wavelet_db2_b_dataset_xgb', ReshapeTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'ensembling_dataset_final', test=args.test)

    struct = {
        'features': [
            ('probs/fft_smoothed10_ar100_dataset_edited_nn_xgb_fft10ar100', ReshapeTransform()),
            ('probs/fft_smoothed10_ar100_dataset_rfecv_xgb', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_1d_half_no_rolling_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_fft', ReshapeTransform()),
            # ('probs/fft_smoothed10_dataset_edited_nn_pca_xgb_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_xgb_fft10', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_rng50', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_window_slicing_2500_rng50', ReshapeTransform()),
            # ('probs/raw_normalized_gaussian50_dataset_xgb_rng50', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_wrapper_2d_rns', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_window_slicing_2d_rns', ReshapeTransform()),
            ('probs/raw_time_series_peak_dataset_rfecv_xgb', ReshapeTransform()),
            # ('probs/wavelet_db2_b_dataset_xgb', ReshapeTransform())
            ('probs/cwt_features_scale2_dataset_cnn_wrapper_2d_window_slicing2500_edits_cwt2', ReshapeTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'ensembling_dataset_final_sub', test=args.test)