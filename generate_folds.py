import os
import pickle as pkl

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import *

if __name__ == '__main__':
    labels = np.load(os.path.join(FEATURES_PATH, 'labels.npy'))
    skf = StratifiedKFold(n_splits=k_fold, shuffle=False, random_state=random_seed)

    folds = list(skf.split(labels, labels))
    pkl.dump(folds, open(FOLDS_FILENAME, 'w'))
