from utils.processing_helper import SimpleTransform, generate_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    struct = {
        'features': [
            ('fft_smoothed_sigma10', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed10_dataset', args.test)

    struct = {
        'features': [
            ('fft_smoothed_sigma20', SimpleTransform()),
            # ('fft_half_normalized', SimpleTransform()),
            # ('fft_normalized', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }

    generate_dataset(struct, 'fft_smoothed20_dataset', args.test)

    struct = {
        'features': [
            ('fft_smoothed_median81', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed_median81_dataset', args.test)

    struct = {
        'features': [
            ('fft_smoothed_median41', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed_median41_dataset', args.test)

    struct = {
        'features': [
            ('fft_smoothed_median21', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed_median21_dataset', args.test)
