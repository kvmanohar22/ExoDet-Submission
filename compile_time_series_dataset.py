import argparse

from preprocessing_features import generate_fft_features
from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    struct = {
        'features': [
            ('tsfeats/raw_ar_coeff100', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_ar_coeff100_dataset', test=args.test)

    struct = {
        'features': [
            ('tsfeats/raw_ar_coeff100', SimpleTransform()),
            ('raw_arima_5_5_1', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_ar_coeff100_arima_dataset', test=args.test)

    struct = {
        'features': [
            ('tsfeats/raw_ar_coeff500', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_ar_coeff500_dataset', test=args.test)

    struct = {
        'features': [
            ('raw_auto_correlation_all', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_auto_correlation_dataset', test=args.test)

    struct = {
        'features': [
            ('raw_mean_std_normalized', SimpleTransform()),
            ('raw_auto_correlation_all', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'orig_with_correlation_dataset', test=args.test)

    struct = {
        'features': [
            ('raw_auto_correlation_all', SimpleTransform(transformer=generate_fft_features))
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_fft_auto_correlation_dataset', test=args.test)

    struct = {
        'features': [
            ('fft_smoothed_sigma10', SimpleTransform()),
            ('tsfeats/raw_ar_coeff100', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed10_ar100_dataset', test=args.test)

    struct = {
        'features': [
            ('tsfeats/raw_mean', SimpleTransform()),
            ('tsfeats/raw_median', SimpleTransform()),
            ('tsfeats/raw_length', SimpleTransform()),
            ('tsfeats/raw_minimum', SimpleTransform()),
            ('tsfeats/raw_maximum', SimpleTransform()),
            ('tsfeats/raw_variance', SimpleTransform()),
            ('tsfeats/raw_skewness', SimpleTransform()),
            ('tsfeats/raw_kurtosis', SimpleTransform()),
            ('tsfeats/raw_sum_values', SimpleTransform()),
            ('tsfeats/raw_abs_energy', SimpleTransform()),
            ('tsfeats/raw_mean_change', SimpleTransform()),
            ('tsfeats/raw_mean_abs_change', SimpleTransform()),
            ('tsfeats/raw_count_below_mean', SimpleTransform()),
            ('tsfeats/raw_count_above_mean', SimpleTransform()),
            ('tsfeats/raw_has_duplicate_min', SimpleTransform()),
            ('tsfeats/raw_has_duplicate_max', SimpleTransform()),
            ('tsfeats/raw_standard_deviation', SimpleTransform()),
            ('tsfeats/raw_absolute_sum_of_changes', SimpleTransform()),
            ('tsfeats/raw_last_location_of_minimum', SimpleTransform()),
            ('tsfeats/raw_last_location_of_maximum', SimpleTransform()),
            ('tsfeats/raw_first_location_of_maximum', SimpleTransform()),
            ('tsfeats/raw_longest_strike_below_mean', SimpleTransform()),
            ('tsfeats/raw_longest_strike_above_mean', SimpleTransform()),
            ('tsfeats/raw_sum_of_reoccurring_values', SimpleTransform()),
            ('tsfeats/raw_first_location_of_minimum', SimpleTransform()),
            ('tsfeats/raw_sum_of_reoccurring_data_points', SimpleTransform()),
            ('tsfeats/raw_variance_larger_than_standard_deviation', SimpleTransform()),
            ('tsfeats/raw_ratio_value_number_to_time_series_length', SimpleTransform()),
            ('tsfeats/raw_percentage_of_reoccurring_values_to_all_values', SimpleTransform()),
            ('tsfeats/raw_binned_entropy_max300', SimpleTransform()),
            ('tsfeats/raw_binned_entropy_max400', SimpleTransform()),
            ('tsfeats/raw_cid_ce_true', SimpleTransform()),
            ('tsfeats/raw_cid_ce_false', SimpleTransform()),
            ('tsfeats/raw_percentage_of_reoccurring_datapoints_to_all_datapoints', SimpleTransform()),
            ('tsfeats/raw_ar_coeff100', SimpleTransform()),
            ('tsfeats/raw_ar_coeff500', SimpleTransform()),
            ('tsfeats/raw_agg50_mean_lin_trend', SimpleTransform()),
            ('tsfeats/raw_aug_dickey_fuler', SimpleTransform()),
            ('tsfeats/raw_energy_ratio_num10_focus5', SimpleTransform()),
            ('tsfeats/raw_fft_aggr_spectrum', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_real', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_imag', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_abs', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_angle', SimpleTransform()),
            ('tsfeats/raw_linear_trend', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_time_series_dataset', test=args.test)

    struct = {
        'features': [
            ('tsfeats/raw_mean', SimpleTransform()),
            ('tsfeats/raw_median', SimpleTransform()),
            ('tsfeats/raw_length', SimpleTransform()),
            ('tsfeats/raw_minimum', SimpleTransform()),
            ('tsfeats/raw_maximum', SimpleTransform()),
            ('tsfeats/raw_variance', SimpleTransform()),
            ('tsfeats/raw_skewness', SimpleTransform()),
            ('tsfeats/raw_kurtosis', SimpleTransform()),
            ('tsfeats/raw_sum_values', SimpleTransform()),
            ('tsfeats/raw_abs_energy', SimpleTransform()),
            ('tsfeats/raw_mean_change', SimpleTransform()),
            ('tsfeats/raw_mean_abs_change', SimpleTransform()),
            ('tsfeats/raw_count_below_mean', SimpleTransform()),
            ('tsfeats/raw_count_above_mean', SimpleTransform()),
            ('tsfeats/raw_has_duplicate_min', SimpleTransform()),
            ('tsfeats/raw_has_duplicate_max', SimpleTransform()),
            ('tsfeats/raw_standard_deviation', SimpleTransform()),
            ('tsfeats/raw_absolute_sum_of_changes', SimpleTransform()),
            ('tsfeats/raw_last_location_of_minimum', SimpleTransform()),
            ('tsfeats/raw_last_location_of_maximum', SimpleTransform()),
            ('tsfeats/raw_first_location_of_maximum', SimpleTransform()),
            ('tsfeats/raw_longest_strike_below_mean', SimpleTransform()),
            ('tsfeats/raw_longest_strike_above_mean', SimpleTransform()),
            ('tsfeats/raw_sum_of_reoccurring_values', SimpleTransform()),
            ('tsfeats/raw_first_location_of_minimum', SimpleTransform()),
            ('tsfeats/raw_sum_of_reoccurring_data_points', SimpleTransform()),
            ('tsfeats/raw_variance_larger_than_standard_deviation', SimpleTransform()),
            ('tsfeats/raw_ratio_value_number_to_time_series_length', SimpleTransform()),
            ('tsfeats/raw_percentage_of_reoccurring_values_to_all_values', SimpleTransform()),
            ('tsfeats/raw_binned_entropy_max300', SimpleTransform()),
            ('tsfeats/raw_binned_entropy_max400', SimpleTransform()),
            ('tsfeats/raw_cid_ce_true', SimpleTransform()),
            ('tsfeats/raw_cid_ce_false', SimpleTransform()),
            ('tsfeats/raw_percentage_of_reoccurring_datapoints_to_all_datapoints', SimpleTransform()),
            ('tsfeats/raw_ar_coeff100', SimpleTransform()),
            ('tsfeats/raw_ar_coeff500', SimpleTransform()),
            ('raw_arima_5_5_1', SimpleTransform()),
            ('tsfeats/raw_agg50_mean_lin_trend', SimpleTransform()),
            ('tsfeats/raw_aug_dickey_fuler', SimpleTransform()),
            ('tsfeats/raw_energy_ratio_num10_focus5', SimpleTransform()),
            ('tsfeats/raw_fft_aggr_spectrum', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_real', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_imag', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_abs', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_angle', SimpleTransform()),
            ('tsfeats/raw_linear_trend', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_time_series_complete_dataset', test=args.test)

    struct = {
        'features': [
            ('peak_features', SimpleTransform()),
            ('dtw_features', SimpleTransform()),
            ('peak_features_paper', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'peak_and_dtw_dataset', test=args.test)


    struct = {
        'features': [
            ('tsfeats/raw_mean', SimpleTransform()),
            ('tsfeats/raw_median', SimpleTransform()),
            ('tsfeats/raw_length', SimpleTransform()),
            ('tsfeats/raw_minimum', SimpleTransform()),
            ('tsfeats/raw_maximum', SimpleTransform()),
            ('tsfeats/raw_variance', SimpleTransform()),
            ('tsfeats/raw_skewness', SimpleTransform()),
            ('tsfeats/raw_kurtosis', SimpleTransform()),
            ('tsfeats/raw_sum_values', SimpleTransform()),
            ('tsfeats/raw_abs_energy', SimpleTransform()),
            ('tsfeats/raw_mean_change', SimpleTransform()),
            ('tsfeats/raw_mean_abs_change', SimpleTransform()),
            ('tsfeats/raw_count_below_mean', SimpleTransform()),
            ('tsfeats/raw_count_above_mean', SimpleTransform()),
            ('tsfeats/raw_has_duplicate_min', SimpleTransform()),
            ('tsfeats/raw_has_duplicate_max', SimpleTransform()),
            ('tsfeats/raw_standard_deviation', SimpleTransform()),
            ('tsfeats/raw_absolute_sum_of_changes', SimpleTransform()),
            ('tsfeats/raw_last_location_of_minimum', SimpleTransform()),
            ('tsfeats/raw_last_location_of_maximum', SimpleTransform()),
            ('tsfeats/raw_first_location_of_maximum', SimpleTransform()),
            ('tsfeats/raw_longest_strike_below_mean', SimpleTransform()),
            ('tsfeats/raw_longest_strike_above_mean', SimpleTransform()),
            ('tsfeats/raw_sum_of_reoccurring_values', SimpleTransform()),
            ('tsfeats/raw_first_location_of_minimum', SimpleTransform()),
            ('tsfeats/raw_sum_of_reoccurring_data_points', SimpleTransform()),
            ('tsfeats/raw_variance_larger_than_standard_deviation', SimpleTransform()),
            ('tsfeats/raw_ratio_value_number_to_time_series_length', SimpleTransform()),
            ('tsfeats/raw_percentage_of_reoccurring_values_to_all_values', SimpleTransform()),
            ('tsfeats/raw_binned_entropy_max300', SimpleTransform()),
            ('tsfeats/raw_binned_entropy_max400', SimpleTransform()),
            ('tsfeats/raw_cid_ce_true', SimpleTransform()),
            ('tsfeats/raw_cid_ce_false', SimpleTransform()),
            ('tsfeats/raw_percentage_of_reoccurring_datapoints_to_all_datapoints', SimpleTransform()),
            ('tsfeats/raw_ar_coeff100', SimpleTransform()),
            ('tsfeats/raw_ar_coeff500', SimpleTransform()),
            ('raw_arima_5_5_1', SimpleTransform()),
            ('tsfeats/raw_agg50_mean_lin_trend', SimpleTransform()),
            ('tsfeats/raw_aug_dickey_fuler', SimpleTransform()),
            ('tsfeats/raw_energy_ratio_num10_focus5', SimpleTransform()),
            ('tsfeats/raw_fft_aggr_spectrum', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_real', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_imag', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_abs', SimpleTransform()),
            ('tsfeats/raw_fft_coeff_angle', SimpleTransform()),
            ('tsfeats/raw_linear_trend', SimpleTransform()),
            ('peak_features', SimpleTransform()),
            ('dtw_features', SimpleTransform()),
            ('peak_features_paper', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_time_series_peak_dataset', test=args.test)
