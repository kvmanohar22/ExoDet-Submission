from utils.processing_helper import SimpleTransform, generate_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    struct = {
        'features': [
            ('wavelet_db2_a', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'wavelet_db2_a_dataset', args.test)

    struct = {
        'features': [
            ('wavelet_db2_b', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'wavelet_db2_b_dataset', args.test)

    struct = {
        'features': [
            ('cwt_features_scale2_real', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'cwt_features_scale2_real_dataset', args.test)

    struct = {
        'features': [
            ('cwt_features_scale2_imag', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'cwt_features_scale2_imag_dataset', args.test)

    struct = {
        'features': [
            ('cwt_features_scale2_real', SimpleTransform()),
            ('cwt_features_scale2_imag', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'cwt_features_scale2_dataset', args.test)
