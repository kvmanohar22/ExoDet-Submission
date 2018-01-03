import copy
import random
import types

import numpy as np
import tensorflow as tf

from config import random_seed

np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)

from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
    BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

from hyperopt import hp

org_input_len = 3197
input_shape = [org_input_len, 2]
new_input_len = 2500
new_input_shape = [new_input_len, 2]
max_slice_start_idx = org_input_len - new_input_len

class KerasBatchClassifier(KerasClassifier):
    """
    Add fit_generator to KerasClassifier
    """

    def fit(self, X, y, **kwargs):
        self.classes_ = set(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)  # Load this function if required in future

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator._original_function))
        fit_args.update(kwargs)

        # Uncomment if required
        # early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=5, mode="auto")
        # model_checkpoint = ModelCheckpoint("results/best_weights.{epoch:02d}-{val_loss:.5f}.hdf5", monitor="val_loss",
        # verbose=5, save_best_only=True, mode="auto")
        # callbacks = [early_stopping, model_checkpoint]
        # fit_args.update({"callbacks": callbacks})

        return self.model.fit_generator(
            self.batch_generator(X, y, batch_size=self.sk_params["batch_size"]),
            steps_per_epoch=X.shape[0] // self.sk_params["batch_size"],
            **fit_args)

    def predict_proba(self, X, **kwargs):
        mid_cut = max_slice_start_idx//2
        X = X.reshape([-1] + input_shape)
        X = X[:, mid_cut:mid_cut+new_input_len]
        return super(KerasBatchClassifier, self).predict_proba(X, **kwargs)

    @staticmethod
    def batch_generator(x_train, y_train, batch_size=32):
        """
        Gives equal number of positive and negative samples, and rotates them randomly in time
        """
        half_batch = batch_size // 2
        x_batch = np.empty((batch_size, x_train.shape[1]), dtype='float32')
        y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

        x_batch_new = np.empty([batch_size]+new_input_shape)

        yes_idx = np.where(y_train[:, 0] == 1.)[0]
        non_idx = np.where(y_train[:, 0] == 0.)[0]

        while True:
            np.random.shuffle(yes_idx)
            np.random.shuffle(non_idx)

            x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
            x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
            y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
            y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

            for i in range(batch_size):
                temp_data = np.reshape(x_batch[i], input_shape)
                rand_start_idx = np.random.randint(max_slice_start_idx)
                x_batch_new[i] = temp_data[rand_start_idx:rand_start_idx+new_input_len, :]

            yield x_batch_new, y_batch

    @property
    def history(self):
        return self.__history


def create_model(learning_rate=50e-5, dropout_1=0.5, dropout_2=0.25):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=new_input_shape))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(Flatten())
    model.add(Dropout(dropout_1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate, decay=4e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasBatchClassifier(build_fn=create_model, epochs=100, batch_size=32, verbose=2, learning_rate=0.00162973933654, dropout_1=0.5, dropout_2=0.5)

params_space = {
    'learning_rate': hp.loguniform('learning_rate', -10, -4),
    'dropout_1': hp.quniform('dropout_1', 0.25, .75, 0.25),
    'dropout_2': hp.quniform('dropout_2', 0.25, .75, 0.25),
}
