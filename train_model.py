import argparse
import os
import pprint
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, average_precision_score

from config import RESULTS_PATH
from utils.processing_helper import load_dataset, load_folds, save_features
from utils.python_utils import start_logging
from utils.model_utils import save_model

np.set_printoptions(precision=3)


def get_metrics(y_true, y_pred, threshold):
    """
    Computes the necessary metrics with probability threshold
    :return: confusion_matrix, precision, recall, f1_score, true_skill_score
    """
    cm = confusion_matrix(y_true, y_pred > threshold)
    skill_score = np.linalg.det(cm) / np.prod(np.sum(cm, axis=1))
    prec, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred > threshold)
    return cm, prec[1], recall[1], f1_score[1], skill_score


def analyze_results(y_true, y_pred):
    """
    Prints the analysis of the predicted results
    :param y_true: True values
    :param y_pred: Predicted values
    :return: returns the list of metrics for different threshold in (threshold, metrics) format
    """
    print "{:s} - {:8.4f}".format('AUC', roc_auc_score(y_true, y_pred)),
    print "\t\t{:s} - {:8.4f}".format('AUPRC', average_precision_score(y_true, y_pred)), "\n"
    formatter = "{:15.4f} {:15.4f} {:15.4f} {:15.4f} {:15.4f} {:10d} {:10d} {:10d} {:10d}"

    results = [(threshold, get_metrics(y_true, y_pred, threshold)) for threshold in np.linspace(0, 1, 1001)]
    max_f1_score_threshold = max(results, key=lambda x: x[1][3])[0]
    max_true_skill_threshold = max(results, key=lambda x: x[1][4])[0]

    print "{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>10s} {:>10s} {:>10s} {:>10s}" \
        .format("Threshold", "Precision", "Recall", "F1-Score", "Skill Score", "FP", "FN", "TP", "TN")
    for threshold in [max_f1_score_threshold, max_true_skill_threshold, 0.25, 0.5, 0.75]:
        cm, prec, recall, f1_score, skill_score = get_metrics(y_true, y_pred, threshold)
        print formatter.format(threshold, prec, recall, f1_score, skill_score, cm[0, 1], cm[1, 0], cm[1, 1],
                               cm[0, 0])
    print

    return results


def summarize_model(model_name, dataset_name, novalidate):
    """
    Trains the model with k-fold cross-validation and then generate the summary.
    Similar to sklearn.model_selection.cross_val_predict
    :param model_name: filename  in model package representing the pipeline or model object
    :param dataset_name: dataset name to load
    :param novalidate: If True, trains the data into whole dataset and save the model
    """
    model = __import__("models.%s" % model_name, globals(), locals(), ['model']).model

    X, y = load_dataset(dataset_name)
    y_complete_pred = np.zeros_like(y).astype('float')

    if not novalidate:
        folds = load_folds()
        for i, (train_index, val_index) in enumerate(folds):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            y_pred_train = model.predict_proba(X_train)[:, 1]

            # Copying the values to generate predictions of complete dataset
            y_complete_pred[val_index] = y_pred

            # Saving the model
            save_model(model, "%s_%s_fold%d" % (dataset_name, model_name, i))

            print "[Fold %d]: " % (i + 1)
            print "Fold Summary: ",
            print "Training AUPRC - %8.4f" % average_precision_score(y_train, y_pred_train)
            analyze_results(y_val, y_pred)
            print

        y_complete_pred.dump(os.path.join(RESULTS_PATH, '%s_%s.npy' % (dataset_name, model_name)))
        save_features(y_complete_pred, 'probs/%s_%s' % (dataset_name, model_name))

        print "Complete Summary: ",
        analyze_results(y, y_complete_pred)
        print "\nModel parameters: "
        pprint.pprint(model.get_params(), indent=4, depth=1)
        print
    else:
        model.fit(X, y)
        y_pred_train = model.predict_proba(X)[:, 1]
        print "Training AUPRC - %8.4f" % average_precision_score(y, y_pred_train)
        save_model(model, '%s_%s' % (dataset_name, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to process")
    parser.add_argument('model', type=str, help="name of the py")
    parser.add_argument('--both', help="Will cross_validate and train the model with complete dataset",
                        action='store_true')
    parser.add_argument('--novalidate', '-nv', help="Perform cross-validation or directly train model?",
                        action='store_true')
    args = parser.parse_args()

    # Log the output to file also
    current_timestring = datetime.now().strftime("%Y%m%d%H%M%S")
    start_logging(os.path.join(RESULTS_PATH, '%s_%s_%s.txt' % (current_timestring, args.dataset, args.model)))

    if args.both:
        summarize_model(args.model, args.dataset, False)
        summarize_model(args.model, args.dataset, True)
    else:
        summarize_model(args.model, args.dataset, args.novalidate)
