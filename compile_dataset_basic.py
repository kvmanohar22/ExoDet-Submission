from utils.processing_helper import SimpleTransform, generate_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    struct = {
        'features': [
            ('raw_mean_std_normalized', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_normalized_dataset', args.test)

    struct = {
        'features': [
            ('raw_mean_std_normalized', SimpleTransform()),
            ('raw_mean_std_normalized_smoothed_uniform200', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_normalized_smoothed_dataset', args.test)

    # Raw data with gaussian smoothing 50
    struct = {
        'features': [
            ('raw_mean_std_normalized', SimpleTransform()),
            ('raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_normalized_gaussian50_dataset', args.test)

    # Only gaussian smoothing 50
    struct = {
        'features': [
            ('raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'smoothed_gaussian50_dataset', args.test)

    struct = {
        'features': [
            ('detrend_gaussian10', SimpleTransform()),
            ('raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian10_with_smoothed_dataset', args.test)

    struct = {
        'features': [
            ('detrend_gaussian10', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian10_dataset', args.test)

    struct = {
        'features': [
            ('detrend_gaussian5', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian5_dataset', args.test)

    struct = {
        'features': [
            ('detrend_gaussian15', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian15_dataset', args.test)

    struct = {
        'features': [
            ('detrend_median81', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian81_dataset', args.test)

    struct = {
        'features': [
            ('detrend_median41', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian41_dataset', args.test)
