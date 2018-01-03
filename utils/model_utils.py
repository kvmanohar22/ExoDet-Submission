import cPickle
import os
import pickle

from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier

from config import MODELFILE_PATH
from utils.processing_helper import make_dir_if_not_exists


def save_model(model, model_filename):
    """
    Saves the model
    :param model: model object
    :param model_filename: File name of the model
    """
    print 'Saving the model...'
    if not isinstance(model, KerasClassifier):
        model_filename = os.path.join(MODELFILE_PATH, '%s.model' % model_filename)
        make_dir_if_not_exists(os.path.dirname(model_filename))
        with open(model_filename, 'wb') as fp:
            pickle.dump(model, fp)
    else:
        json_model = model.model.to_json()
        model_filename_archi = os.path.join(MODELFILE_PATH, '%s_archi.model' % model_filename)
        make_dir_if_not_exists(os.path.dirname(model_filename_archi))
        model_filename_weights = os.path.join(MODELFILE_PATH, '%s_weights.model' % model_filename)
        make_dir_if_not_exists(os.path.dirname(model_filename_weights))
        # Save the architecture
        with open(model_filename_archi, 'w') as f:
            f.write(json_model)
        # Save the weights
        model.model.save_weights(model_filename_weights, overwrite=True)

        model.model = None
        model_filename = os.path.join(MODELFILE_PATH, '%s.model' % model_filename)
        with open(model_filename, 'wb') as fp:
            cPickle.dump(model, fp)


def load_model(dataset_name, model_name):
    """
    Loads a model
    :param dataset_name: Name of the dataset to load
    :param model_filename: Name of the model to load
    """
    model_filename = os.path.join(MODELFILE_PATH, '%s_%s.model' % (dataset_name, model_name))
    with open(model_filename, 'rb') as fp:
        model_wrapper = cPickle.load(fp)
    if isinstance(model_wrapper, KerasClassifier):
        archi_file = os.path.join(MODELFILE_PATH, '%s_%s_archi.model' % (dataset_name, model_name))
        weights_file = os.path.join(MODELFILE_PATH, '%s_%s_weights.model' % (dataset_name, model_name))
        model = model_from_json(open(archi_file).read())
        model.load_weights(weights_file)

        model_wrapper.model = model
    return model_wrapper
