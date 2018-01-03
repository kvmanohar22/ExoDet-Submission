#!/bin/bash

echo "Generating Fetures"
python preprocessing_features.py
python generate_time_series_features.py
python generate_peak_features.py
python generate_cwt_features.py

echo "Generating Folds"
python generate_folds.py

echo "Compiling Datasets"
python compile_dataset_basic.py
python compile_dataset_fft.py
python compile_dataset_wavelet.py

echo 'Training models'
python train_model.py fft_smoothed10_ar100_dataset edited_nn_xgb_fft10ar100 --both
python train_model.py fft_smoothed10_ar100_dataset rfecv_xgb --both
python train_model.py fft_smoothed10_dataset cnn_wrapper_1d_half_no_rolling_fft10 --both
python train_model.py fft_smoothed10_dataset cnn_wrapper_fft --both
python train_model.py fft_smoothed10_dataset xgb_fft10 --both
python train_model.py raw_normalized_gaussian50_dataset cnn_wrapper_2d_rng50 --both
python train_model.py raw_normalized_gaussian50_dataset cnn_wrapper_2d_window_slicing_2500_rng50 --both
python train_model.py raw_normalized_smoothed_dataset cnn_wrapper_2d_rns --both
python train_model.py raw_normalized_smoothed_dataset cnn_window_slicing_2d_rns --both
python train_model.py raw_time_series_peak_dataset rfecv_xgb --both
python train_model.py cwt_features_scale2_dataset cnn_wrapper_2d_window_slicing2500_edits_cwt2 --both

python compile_ensembling_dataset.py

python train_model.py ensembling_dataset_final_sub ensembling_submission --both
python train_model.py ensembling_dataset_final ensemble_dataset_11 --both
