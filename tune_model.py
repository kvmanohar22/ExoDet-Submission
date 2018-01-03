import argparse
import os
import pickle as pkl
from datetime import datetime

from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict

from config import RESULTS_PATH
from train_model import analyze_results
from utils.processing_helper import load_dataset, load_folds
from utils.python_utils import start_logging, print_dict, read_pickle_if_exists, prettyfloat


def get_foldwise_metric(y_true, y_pred, cv, metric):
    result = []
    for _, test_indexes in cv:
        result.append(prettyfloat(metric(y_true[test_indexes], y_pred[test_indexes])))
    return result


def tune_model(model_name, dataset_name, n_trials, args):
    module_file = __import__("models.%s" % model_name, globals(), locals(), ['model', 'params_space'])
    model = module_file.model
    params_space = module_file.params_space
    trials_file_path = os.path.join(RESULTS_PATH, "%s_%s_tuning_trials.pkl" % (model_name, dataset_name))

    X, y = load_dataset(dataset_name)
    folds = load_folds()

    def objective_fn(params):
        print "\n\nHyper-Parameters: "
        print_dict(params)

        model.set_params(**params)
        y_pred = cross_val_predict(model, X, y, cv=folds, method='predict_proba', verbose=2, n_jobs=args['n_jobs'])

        print "Fold Wise AUPRC ", get_foldwise_metric(y, y_pred[:, 1], folds, average_precision_score)
        # values = ['%0.2f' % val for val in ]
        results = analyze_results(y, y_pred[:, 1])

        max_f1_score = max(results, key=lambda x: x[1][3])[1][3]
        return {'loss': -max_f1_score, 'status': STATUS_OK}

    trials = read_pickle_if_exists(trials_file_path) or Trials()
    best_params = fmin(fn=objective_fn,
                       space=params_space,
                       algo=tpe.suggest,
                       max_evals=n_trials,
                       trials=trials
                       )
    best_params['z_score'] = -trials.best_trial['result']['loss']
    print "\n\nBest Parameters..."
    print_dict(best_params)

    # Save the trials
    pkl.dump(trials, open(trials_file_path, 'wb'))

    return best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to process")
    parser.add_argument('model', type=str, help="name of the python file")
    parser.add_argument('--trials', '-t', type=int, default=3, help="Number of trials to choose the perform the "
                                                                    "validation")
    parser.add_argument('--n_jobs', '-n', type=int, default=1, help="Number of threads to run in parallel for cross "
                                                                    "validation")
    args = parser.parse_args()

    # Log the output to file also
    current_timestring = datetime.now().strftime("%Y%m%d%H%M%S")
    start_logging(os.path.join(RESULTS_PATH, 'tune_%s_%s_%s.txt' % (current_timestring, args.dataset, args.model)))

    tune_model(args.model, args.dataset, n_trials=args.trials, args={
        'n_jobs': args.n_jobs
    })
