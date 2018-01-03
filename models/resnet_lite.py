import copy
import random
import types

import numpy as np
import tensorflow as tf
from hyperopt import hp

from config import random_seed

np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)

from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
    BatchNormalization, Input, Activation, Add, AveragePooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model

input_shape = [3197, 2]


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
            steps_per_epoch=2 * X.shape[0] // self.sk_params["batch_size"],
            **fit_args)

    def predict_proba(self, x, **kwargs):
        x = x.reshape([-1] + input_shape)
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        probs = self.model.predict(x, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        else:
            raise NotImplementedError
        return probs

    @staticmethod
    def batch_generator(x_train, y_train, batch_size=32):
        """
        Gives equal number of positive and negative samples, and rotates them randomly in time
        """
        half_batch = batch_size // 2
        x_batch = np.empty([batch_size] + input_shape, dtype='float32')
        y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

        yes_idx = np.where(y_train[:, 0] == 1.)[0]
        non_idx = np.where(y_train[:, 0] == 0.)[0]

        while True:
            np.random.shuffle(yes_idx)
            np.random.shuffle(non_idx)

            x_batch[:half_batch] = np.reshape(x_train[yes_idx[:half_batch]], [-1] + input_shape)
            x_batch[half_batch:] = np.reshape(x_train[non_idx[half_batch:batch_size]], [-1] + input_shape)
            y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
            y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

            for i in range(batch_size):
                sz = np.random.randint(x_batch.shape[1])
                x_batch[i] = np.roll(x_batch[i], sz, axis=0)

            yield x_batch, y_batch

    @property
    def history(self):
        return self.__history


def create_model(learning_rate=50e-5, dropout1=0.5):
    input_data_reshape = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=11, strides=2)(input_data_reshape)
    x = BatchNormalization(axis=1, name='bn_1')(x)
    x = Activation('relu')(x)
    x = MaxPool1D(strides=4)(x)

    # Shortcut for the first residual block
    shortcut = Conv1D(filters=128, kernel_size=1, strides=2, padding='SAME')(x)
    shortcut = BatchNormalization(axis=1, name='bn_2')(shortcut)
    # 1st conv block
    x1 = Conv1D(filters=32, kernel_size=1, strides=2, padding='SAME')(x)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=32, kernel_size=3, strides=1, padding='SAME')(x1)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=128, kernel_size=1, strides=1, padding='SAME')(x1)
    x1 = BatchNormalization(axis=1)(x1)

    # Merge the layers for the first conv block
    block_1 = Add()([shortcut, x1])
    block_1 = Activation('relu')(block_1)

    x = block_1

    # Shortcut for the second residual block
    for _ in range(1):
        shortcut = x
        # 2nd conv block
        x1 = Conv1D(filters=32, kernel_size=1, strides=1, padding='SAME')(x)
        x1 = BatchNormalization(axis=1)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=32, kernel_size=3, strides=1, padding='SAME')(x1)
        x1 = BatchNormalization(axis=1)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=128, kernel_size=1, strides=1, padding='SAME')(x1)
        x1 = BatchNormalization(axis=1)(x1)

        # Merge the layers for the second conv block
        block_2 = Add()([shortcut, x1])
        block_2 = Activation('relu')(block_2)
        x = block_2

    # 3rd
    shortcut = Conv1D(filters=128, kernel_size=1, strides=2, padding='SAME')(x)
    shortcut = BatchNormalization(axis=1)(shortcut)
    # Residual
    x1 = Conv1D(filters=64, kernel_size=1, strides=2, padding='SAME')(x)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='SAME')(x1)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=128, kernel_size=1, strides=1, padding='SAME')(x1)
    x1 = BatchNormalization(axis=1)(x1)

    # Sum
    block_1 = Add()([shortcut, x1])
    x = Activation('relu')(block_1)

    for _ in range(1):
        shortcut = MaxPool1D(strides=2)(x)
        # 2nd conv block
        x1 = Conv1D(filters=64, kernel_size=1, strides=2, padding='SAME')(x)
        x1 = BatchNormalization(axis=1)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='SAME')(x1)
        x1 = BatchNormalization(axis=1)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=128, kernel_size=1, strides=1, padding='SAME')(x1)
        x1 = BatchNormalization(axis=1)(x1)

        # Merge the layers for the second conv block
        block_2 = Add()([shortcut, x1])
        block_2 = Activation('relu')(block_2)
        x = block_2

    # 3rd
    shortcut = Conv1D(filters=256, kernel_size=1, strides=2, padding='SAME')(x)
    shortcut = BatchNormalization(axis=1)(shortcut)
    # Residual
    x1 = Conv1D(filters=64, kernel_size=1, strides=2, padding='SAME')(x)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='SAME')(x1)
    x1 = BatchNormalization(axis=1)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='SAME')(x1)
    x1 = BatchNormalization(axis=1)(x1)

    # Sum
    block_1 = Add()([shortcut, x1])
    x = Activation('relu')(block_1)

    for _ in range(1):
        shortcut = x
        x1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='SAME')(x)
        x1 = BatchNormalization(axis=1)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='SAME')(x1)
        x1 = BatchNormalization(axis=1)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='SAME')(x1)
        x1 = BatchNormalization(axis=1)(x1)

        # Merge the layers for the second conv block
        block_2 = Add()([shortcut, x1])
        block_2 = Activation('relu')(block_2)
        x = block_2

    x = AveragePooling1D(pool_size=25)(x)

    # Rest of the layers
    flat = Flatten()(x)
    drop1 = Dropout(dropout1)(flat)
    dense1 = Dense(64, activation='relu')(drop1)
    output = Dense(1, activation='sigmoid')(dense1)
    model = Model(input_data_reshape, output, name='resnet')

    # Compile
    model.compile(optimizer=Adam(learning_rate, decay=2e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasBatchClassifier(build_fn=create_model, epochs=30, batch_size=32, verbose=2)

params_space = {
    'learning_rate': hp.loguniform('learning_rate', -10, -4),
    'dropout1': hp.quniform('dropout1', 0.25, .75, 0.25),
}

