#!/usr/bin/env python
# Ziyi Yang
# Stacked_LSTM model with multi-source input features for the DeepBGCpred model.
# Using sliding window strategy in the prediction stage.

from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from keras.layers.core import *

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)


class KerasRNN(BaseEstimator, ClassifierMixin):
    """
    Stacked_LSTM model with multi-source input features for the DeepBGCpred model.
    """

    def __init__(
        self,
        batch_size=1,
        hidden_size=128,
        loss="binary_crossentropy",
        stateful=True,
        activation="sigmoid",
        return_sequences=True,
    ):
        self.model = None
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.loss = loss
        self.stateful = stateful
        self.activation = activation
        self.return_sequences = return_sequences

    def _build_model(
        self,
        input_size,
        stacked_sizes=None,
        fully_connected_sizes=None,
        optimizer_name=None,
        learning_rate=None,
        decay=None,
        custom_batch_size=None,
    ):
        """
        Build Keras Sequential model architecture with given parameters
        :param input_size: Dimensionality of input vector (number of features)
        :param stacked_sizes: Add given number of additional Bi-LSTM layers after first Bi-LSTM layer, provided as list of sizes
        :param fully_connected_sizes: Add a given number of additional fully connected layers after the Bi-LSTM layers, provided as list of sizes
        :param optimizer_name: Name of Keras optimizer, default 'adam'
        :param learning_rate: Keras learning rate
        :param decay: Optimizer decay
        :param custom_batch_size: Use different batch size than self.batch_size
        :return: Keras Sequential model
        """
        from keras.layers.core import Dense
        from keras.layers.recurrent import LSTM
        from keras.layers.wrappers import TimeDistributed, Bidirectional
        from keras import optimizers
        from keras.models import Sequential

        from keras import Input, Model
        from keras.layers import (
            concatenate,
            Dropout,
            Embedding,
            Convolution1D,
            MaxPooling1D,
            Flatten,
            AveragePooling1D,
        )

        if stacked_sizes is None:
            stacked_sizes = []
        if fully_connected_sizes is None:
            fully_connected_sizes = []

        if len(input_size) == 1:
            """The DeepBGC model proposed by Merck"""
            model = Sequential()
            model.add(
                Bidirectional(
                    layer=LSTM(
                        units=self.hidden_size,
                        return_sequences=True,
                        dropout=0.2,
                        recurrent_dropout=0.2,
                        stateful=self.stateful,
                    ),
                    batch_input_shape=(
                        custom_batch_size or self.batch_size,
                        None,
                        input_size[0],
                    ),
                )
            )

            model.add(
                LSTM(self.hidden_size, return_sequences=True, stateful=self.stateful)
            )

            model.add(TimeDistributed(Dense(1, activation="sigmoid")))
        else:
            """The Deep-BGCpred model"""
            input1 = Input(
                shape=(
                    None,
                    input_size[0],
                ),
                name="input1",
            )
            input2 = Input(
                shape=(
                    None,
                    input_size[1],
                ),
                name="input2",
            )
            input3 = Input(
                shape=(
                    None,
                    input_size[2],
                ),
                name="input3",
            )

            layer2_dense = TimeDistributed(Dropout(0.2))(input2)
            layer2_lstm = Bidirectional(
                LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
            )(layer2_dense)
            layer2_dense = TimeDistributed(Dropout(0.2))(layer2_lstm)

            # Embedding layers
            layer3_embedding = Embedding(
                input_dim=alphabet_size + 1, output_dim=30, name="layer3_embedding"
            )(input3)

            # Convolution layers
            layer3_dropout1 = TimeDistributed(Dropout(0.2), name="layer3_dropout1")(
                layer3_embedding
            )
            layer3_conv1d = TimeDistributed(
                Convolution1D(30, 4, padding="same"), name="layer3_convolution1D"
            )(layer3_dropout1)
            layer3_maxpool = TimeDistributed(
                MaxPooling1D(2), name="layer3_MaxPooling1D"
            )(layer3_conv1d)
            layer3_dropout2 = TimeDistributed(Dropout(0.2), name="layer3_dropout2")(
                layer3_maxpool
            )
            layer3_x = TimeDistributed(Flatten(), name="layer3_flatten")(
                layer3_dropout2
            )

            x = concatenate([input1, layer2_dense, layer3_x])
            x = Bidirectional(
                LSTM(
                    self.hidden_size,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                )
            )(x)
            x = LSTM(self.hidden_size, return_sequences=True)(x)
            avg_pool = MaxPooling1D(data_format="channels_first")(x)
            max_pool = AveragePooling1D(data_format="channels_first")(x)
            x = concatenate([avg_pool, max_pool])
            x_output = TimeDistributed(Dense(1, activation="sigmoid"))(x)
            model = Model(inputs=[input1, input2, input3], outputs=x_output)

        for size in stacked_sizes:
            model.add(
                Bidirectional(
                    layer=LSTM(
                        units=size, return_sequences=True, stateful=self.stateful
                    )
                )
            )

        for size in fully_connected_sizes:
            model.add(TimeDistributed(Dense(size, activation="sigmoid")))

        if optimizer_name is None:
            optimizer_name = "adam"

        optimizer_args = {}
        if learning_rate is not None:
            optimizer_args["lr"] = learning_rate
        if decay is not None:
            optimizer_args["decay"] = decay

        if optimizer_name == "adam":
            optimizer = optimizers.Adam(**optimizer_args)
        elif optimizer_args:
            raise ValueError(
                "Optimizer {} not implemented for custom params yet".format(
                    optimizer_name
                )
            )
        else:
            optimizer = optimizer_name

        logging.debug("Using optimizer %s %s", optimizer_name, optimizer_args)
        model.compile(
            loss=self.loss,
            optimizer=optimizer,
            sample_weight_mode="temporal",
            metrics=["accuracy", precision, recall, auc_roc],
        )

        return model

    def fit(
        self,
        X_list,
        y_list,
        timesteps=128,
        validation_size=0.33,
        num_epochs=10,
        verbose=1,
        debug_progress_path=None,
        fully_connected_sizes=None,
        shuffle=True,
        stacked_sizes=None,
        early_stopping=None,
        positive_weight=None,
        weighted=False,
        optimizer=None,
        learning_rate=None,
        decay=None,
        validation_X_list=None,
        validation_y_list=None,
        input_size=[102, 64, 64],
    ):
        """
        Train Keras Sequential model using provided list of positive / negative samples.
        Training is done in given number of epochs with additional stopping criteria.
        In each epoch, we go over all samples in X_list, which are shuffled randomly and merged together into artificial genomes.

        :param X_list: List of DataFrames (samples) where each DataFrame contains protein domains represented by numeric vectors
        :param y_list: List of output values, one value for each sample where 0 = negative sample (non-BGC), 1 = positive sample (BGC)
        :param timesteps: Number of timesteps (protein domains) in one batch
        :param validation_size: Fraction of samples to use for testing
        :param num_epochs: Number of epochs. If early stopping is defined, this serves as a limit of maximum number of epochs.
        :param verbose: Verbosity (0 = silent, 1 = verbose, 2 = very verbose)
        :param debug_progress_path: Log Tensorboard information in given folder
        :param fully_connected_sizes: Add a given number of additional fully connected layers after the Bi-LSTM layers, provided as list of sizes
        :param shuffle: Whether to shuffle samples within each epoch. If not used, make sure that positive and negative samples are already shuffled in the list.
        :param stacked_sizes: Add given number of additional Bi-LSTM layers after first Bi-LSTM layer, provided as list of sizes
        :param early_stopping: Keras early stopping parameters dictionary (monitor, min_delta, patience, mode)
        :param positive_weight: Weight of positive samples (single number). Can be used to counter imbalance in training data.
        :param weighted: Calculate positive weight automatically as num negatives / num positive samples in input training data (y_list).
        :param optimizer: Name of Keras optimizer, default 'adam'.
        :param learning_rate: Keras learning rate
        :param decay: Keras optimizer decay.
        :param validation_X_list: List of DataFrames (samples) used to observe validation performance
        :param validation_y_list: List of output values for validation samples, one value for each sample where 0 = negative sample (non-BGC), 1 = positive sample (BGC)
        :return: self
        """

        import keras

        if not isinstance(X_list, list):
            raise AttributeError("Expected X_list to be list, got " + str(type(X_list)))

        if not isinstance(y_list, list):
            raise AttributeError("Expected y_list to be list, got " + str(type(X_list)))

        if weighted:
            if positive_weight:
                raise ValueError(
                    "Positive weight cannot be specified together with weighted=true"
                )
            num_neg = _count_samples(y_list, 0)
            num_pos = _count_samples(y_list, 1)
            positive_weight = num_neg / num_pos
            logging.info("Negative: %s, Positive: %s", num_neg, num_pos)
            logging.info(
                "Weighing positives based on ratio, weight: %s", positive_weight
            )

        train_model = self._build_model(
            input_size,
            stacked_sizes,
            fully_connected_sizes=fully_connected_sizes,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            decay=decay,
        )
        self.model = self._build_model(
            input_size,
            stacked_sizes,
            fully_connected_sizes=fully_connected_sizes,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            decay=decay,
            custom_batch_size=1,
        )

        # print(train_model.summary())

        X_train, y_train = X_list, y_list
        validation_data, validation_num_batches = None, None

        if validation_X_list:
            if positive_weight:
                logging.warning(
                    'Warning: Not using positive_weight "%s" on external validation set!',
                    positive_weight,
                )
            if validation_size:
                logging.warning(
                    "Warning: LSTM validation size %s specified but ignored, "
                    "because external validation set is also present",
                    validation_size,
                )

            logging.info(
                "Validating on external validation set of %s samples",
                len(validation_X_list),
            )
            if shuffle:
                logging.warning("New in DeepBGC 0.1.17: Shuffling validation samples!")
                validation_X_list, validation_y_list = _shuffle_lists(
                    validation_X_list, validation_y_list
                )
            validation_data = _repeat_all_to_fill_batch_size(
                validation_X_list, validation_y_list, self.batch_size
            )
            validation_num_batches = None
        elif validation_size:
            logging.info("Validating on {:.1f}% of input set", validation_size * 100)
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_list, y_list, test_size=validation_size
            )

            """
             _build_generator_multi_three for Deep-BGCpred
             _build_generator for DeepBGC
            """
            if len(input_size) != 1:
                (
                    get_validation_gen,
                    validation_num_batches,
                ) = _build_generator_multi_three(
                    X_validation,
                    y_validation,
                    batch_size=self.batch_size,
                    timesteps=timesteps,
                    input_size=input_size,
                    shuffle=shuffle,
                    positive_weight=positive_weight,
                )
            else:
                get_validation_gen, validation_num_batches = _build_generator(
                    X_validation,
                    y_validation,
                    batch_size=self.batch_size,
                    timesteps=timesteps,
                    input_size=input_size[0],
                    shuffle=shuffle,
                    positive_weight=positive_weight,
                )
            validation_data = get_validation_gen()

        if len(input_size) != 1:
            get_train_gen, train_num_batches = _build_generator_multi_three(
                X_train,
                y_train,
                batch_size=self.batch_size,
                timesteps=timesteps,
                input_size=input_size,
                shuffle=shuffle,
                positive_weight=positive_weight,
            )
        else:
            get_train_gen, train_num_batches = _build_generator(
                X_train,
                y_train,
                batch_size=self.batch_size,
                timesteps=timesteps,
                input_size=input_size[0],
                shuffle=shuffle,
                positive_weight=positive_weight,
            )
        train_gen = get_train_gen()

        callbacks = []
        if debug_progress_path:
            tb = keras.callbacks.TensorBoard(
                log_dir=debug_progress_path,
                histogram_freq=0,
                batch_size=self.batch_size,
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_layer_names=None,
                embeddings_metadata=None,
            )
            callbacks.append(tb)

        if early_stopping:
            logging.info("Using early stopping: %s", early_stopping)
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    min_delta=early_stopping.get("min_delta"),
                    monitor=early_stopping.get("monitor"),
                    patience=early_stopping.get("patience"),
                    mode=early_stopping.get("mode"),
                    verbose=1,
                )
            )

        history = train_model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_num_batches,
            shuffle=False,
            epochs=num_epochs,
            validation_data=validation_data,
            validation_steps=validation_num_batches,
            callbacks=callbacks,
            verbose=verbose,
        )

        trained_weights = train_model.get_weights()
        self.model.set_weights(trained_weights)

        return history

    def predict(
        self,
        X,
        input_size=[102, 64, 64],
        sliding_window=True,
        sw_width=256,
        sw_steps=20,
    ):
        """
        Predict given sample DataFrame/numpy matrix of numeric protein vectors
        :param X: DataFrame/numpy matrix of protein vectors
        :param input_size: the dimension of the input features
        :param sliding_window: Whether adopt the sliding window strategy: True or False
        :param sw_width: the width of sliding window
        :param sw_steps: the step length of sliding window
        :return: BGC prediction score for each protein vector
        """

        if len(X.shape) != 2:
            raise AttributeError(
                "Can only be called on a single 2-dimensional feature matrix"
            )

        if self.model is None:
            raise AttributeError("Cannot predict using untrained model")

        if sliding_window == False:
            """Without sliding window to detect the BGC regions"""
            if len(input_size) == 1:
                # For DeepBGC model
                batch_matrix = X.values.reshape(1, X.shape[0], X.shape[1])
            else:
                # For Deep-BGCpred model without sliding window strategy.
                X_arr1 = X.iloc[:, 0 : input_size[0]]
                X_arr2 = X.iloc[:, input_size[0] : input_size[0] + input_size[1]]
                X_arr3 = X.iloc[
                    :,
                    input_size[0]
                    + input_size[1] : input_size[0]
                    + input_size[1]
                    + input_size[2],
                ]
                batch_matrix1 = X_arr1.values.reshape(
                    1, X_arr1.shape[0], X_arr1.shape[1]
                )
                batch_matrix2 = X_arr2.values.reshape(
                    1, X_arr2.shape[0], X_arr2.shape[1]
                )
                batch_matrix3 = X_arr3.values.reshape(
                    1, X_arr3.shape[0], X_arr3.shape[1]
                )
                batch_matrix = [batch_matrix1, batch_matrix2, batch_matrix3]

            # Reset hidden state of the model to ensure independent prediction from previous samples
            self.model.reset_states()
            probs = self.model.predict(batch_matrix, batch_size=1)
        else:
            """With sliding window to detect the BGC regions"""
            sw_width = sw_width
            sw_steps = sw_steps
            in_start = 0
            X_winds = []
            tmp = 0
            for _ in range(len(X)):
                in_end = in_start + sw_width
                out_end = in_end

                if out_end <= len(X):
                    X_seq = X[in_start:in_end]
                    X_winds.append(X_seq)
                elif tmp == 0:
                    X_seq = X[in_start : len(X)]
                    X_winds.append(X_seq)
                    tmp += 1
                in_start += sw_steps

            probs_shape = (X.shape[0], len(X_winds))
            probs_array = np.zeros(probs_shape)
            in_start1 = 0

            for i in range(len(X_winds)):

                if X.shape[1] == 102:
                    batch_matrix = X_winds[i].values.reshape(
                        1, X_winds[i].shape[0], X_winds[i].shape[1]
                    )
                else:
                    X_arr1 = X.iloc[:, 0 : input_size[0]]
                    X_arr2 = X.iloc[:, input_size[0] : input_size[0] + input_size[1]]
                    X_arr3 = X.iloc[
                        :,
                        input_size[0]
                        + input_size[1] : input_size[0]
                        + input_size[1]
                        + input_size[2],
                    ]
                    batch_matrix1 = X_arr1.values.reshape(
                        1, X_arr1.shape[0], X_arr1.shape[1]
                    )
                    batch_matrix2 = X_arr2.values.reshape(
                        1, X_arr2.shape[0], X_arr2.shape[1]
                    )
                    batch_matrix3 = X_arr3.values.reshape(
                        1, X_arr3.shape[0], X_arr3.shape[1]
                    )
                    batch_matrix = [batch_matrix1, batch_matrix2, batch_matrix3]

                self.model.reset_states()
                prob = self.model.predict(batch_matrix, batch_size=1)
                prob_list = prob.flatten()

                in_end1 = in_start1 + sw_width
                out_end1 = in_end1
                if out_end1 <= len(X):
                    probs_array[in_start1:in_end1, i] = prob_list
                    in_start1 += sw_steps
                else:
                    probs_array[in_start1 : len(X), i] = prob_list

            exist = probs_array != 0
            num = probs_array.sum(axis=1)
            den = exist.sum(axis=1)
            probs = num / den

            probs = np.array(probs).reshape(1, len(probs), 1)

        return pd.Series(probs[0, :, 0], X.index)

    def __getstate__(self):
        """
        Get representation of object that can be pickled
        :return: objects to be pickled
        """
        attrs = self.__dict__.copy()
        del attrs["model"]

        if self.model is None:
            return attrs, None, None
        return attrs, self.model.to_json(), self.model.get_weights()

    def __setstate__(self, state):
        """
        Load object from pickled representation
        :param state: attributes of model generated by __getstate__
        """
        from keras.models import model_from_json

        attrs, architecture, weights = state

        self.__dict__.update(attrs)

        if architecture is None:
            self.model = None
        else:
            self.model = model_from_json(architecture)
            self.model.set_weights(weights)


def rotate(l, n):
    m = n % len(l)
    return l[m:] + l[:m]


def _shuffle_lists(*lists):
    shuffled = []
    length = len(lists[0])
    idx = list(range(length))
    np.random.shuffle(idx)
    for l in lists:
        if len(l) != length:
            raise ValueError(
                "Lists have to be the same length, got: {} and {}".format(
                    len(l), length
                )
            )
        shuffled.append([l[i] for i in idx])
    return shuffled


def _noop():
    return None


def _yield_single_pair(a, b):
    yield a, b


def _rotate_to_fill_batch_size(X_list, y_list, batch_size, input_size):
    """
    Fill matrix of batch_size rows with samples from X_list in a way that all samples are (approximately) evenly present.
    Create batch_size rows, each row as long as the longest sample in X_list (max_len).
    For row on index i, include concatenated sequence of X_list starting from sample i (sequence is trimmed to max_len).
    :param X_list: list of samples
    :param y_list: list of sample responses
    :param batch_size: how many rows to create
    :param input_size: number of columns in sample
    :return: Filled matrix of batch_size rows with samples from X_list in a way that all samples are (approximately) evenly present.
    """

    if len(X_list) > batch_size:
        raise AttributeError("Cannot repeat more samples than batch_size")

    max_len = max([X.shape[0] for X in X_list])

    fill_shape = (batch_size, max_len, input_size)
    fill_num_values = fill_shape[0] * fill_shape[1] * fill_shape[2]
    logging.info(
        "Filling to batch size shape %s (%sM values)...",
        fill_shape,
        int(fill_num_values / 1000000),
    )

    X_filled = np.zeros(shape=fill_shape)
    y_filled = np.zeros(shape=(fill_shape[0], fill_shape[1], 1))

    for i in range(0, batch_size):
        X_filled[i] = np.concatenate(rotate(X_list, i))[:max_len]
        y_filled[i][:, 0] = np.concatenate(rotate(y_list, i))[:max_len]

    logging.info("Filling done")
    return X_filled, y_filled


def _repeat_all_to_fill_batch_size(X_sequences, y_sequences, batch_size):
    """
    Merge the sequences and repeat batch_size times to fill a matrix with (batch_size, total_sequences_rows, input_size) shape.
    :param X_sequences: list of DataFrames (sequences)
    :param y_sequences: list of Series of output sequence values
    :param batch_size: how many rows to create
    :return: Filled matrix of batch_size rows with samples from X_list in a way that all samples are (approximately) evenly present.
    """

    X_concat = pd.concat(X_sequences)
    y_concat = pd.concat(y_sequences)
    fill_shape = (batch_size,) + X_concat.shape
    fill_num_values = fill_shape[0] * fill_shape[1] * fill_shape[2]
    logging.info(
        "Filling to batch size shape %s (%sM values)...",
        fill_shape,
        int(fill_num_values / 1000000),
    )

    X_filled = np.zeros(shape=fill_shape)
    y_filled = np.zeros(shape=(fill_shape[0], fill_shape[1], 1))

    for i in range(0, batch_size):
        X_filled[i] = X_concat
        y_filled[i][:, 0] = y_concat

    logging.info("Filling done")
    return X_filled, y_filled


def _build_generator(
    X_list, y_list, batch_size, timesteps, input_size, shuffle, positive_weight
):
    """
    Build looping generator of training batches. Will return the generator and the number of batches in each epoch.
    In each epoch, all samples are randomly split into batch_size "chunks", each "chunk" in batch can be trained in parallel.
    Samples in each chunk are shuffled and merged into one whole sequence.
    The whole sequences are separated into batches of given fixed given number of timesteps (protein vectors).
    So the number of batches is defined so that we go over the whole sequence (length of the longest "chunk" sequence divided by the number of timesteps).

    :param X_list: List of samples. Each sample is a matrix/DataFrame of protein domain vectors.
    :param y_list: List of sample outputs.
    :param batch_size: Number of parallel "chunks" in a training batch
    :param timesteps: Number of timesteps (protein domain vectors) in a training batch
    :param input_size: Size of the protein domain vector
    :param shuffle: Whether to shuffle samples within each epoch. If not used, make sure that positive and negative samples are already shuffled in the list.
    :param positive_weight: Weight of positive samples (single number). If provided, a triple of (X_batch, y_batch, weights_batch) are provided
    :return: Tuple of (batch generator, number of batches in each epoch).
    Each batch will contain the X input (batch_size, timesteps, input_size) and y output (batch_size, timesteps, 1)
    """
    if not X_list:
        return _noop, None
    from keras.preprocessing.sequence import pad_sequences

    seq_length = sum([len(X) for X in X_list])
    X_arr = np.array([X.values for X in X_list])
    y_arr = np.array([y.values for y in y_list])
    num_batches = int(np.ceil(np.ceil(seq_length / batch_size) / timesteps))
    maxlen = num_batches * timesteps
    logging.info(
        "Initializing generator of %s batches from sequence length %s",
        num_batches,
        seq_length,
    )

    def generator():
        while True:
            # shuffle the samples
            if shuffle:
                shuffled = np.random.permutation(len(X_list))
            # split samples into batch_size chunks
            X_batches = np.array_split(
                X_arr[shuffled] if shuffle else X_arr, batch_size
            )
            y_batches = np.array_split(
                y_arr[shuffled] if shuffle else y_arr, batch_size
            )

            # merge the samples in each chunk into one sequence
            X_batches = [
                np.concatenate(b) if b.size else np.empty(0) for b in X_batches
            ]
            y_batches = [
                np.concatenate(b) if b.size else np.empty(0) for b in y_batches
            ]

            # pad the sequences with zeros to the length of the longest chunk sequence
            X_batches = pad_sequences(
                X_batches,
                maxlen=maxlen,
                dtype=np.float,
                padding="post",
                truncating="post",
            )
            y_batches = pad_sequences(
                y_batches,
                maxlen=maxlen,
                dtype=np.float,
                padding="post",
                truncating="post",
            )

            # Reshape array so that it can be indexed as [batch number][chunk][timestep][input feature]
            # This will produce an array of dimension (num_batches, batch_size, timesteps, input_size)
            # And output array of dimension (num_batches, batch_size, timesteps, 1)
            X_batches = np.swapaxes(
                X_batches.reshape(batch_size, num_batches, timesteps, input_size), 0, 1
            )
            y_batches = np.swapaxes(
                y_batches.reshape(batch_size, num_batches, timesteps, 1), 0, 1
            )

            if positive_weight:
                # Provide array of weights for each input vector based on the positive weight
                weight_batches = np.ones(y_batches.shape)
                weight_batches[y_batches == 1] = positive_weight
                weight_batches = np.swapaxes(
                    weight_batches.reshape(batch_size, num_batches, timesteps), 0, 1
                )
                for X_batch, y_batch, weight_batch in zip(
                    X_batches, y_batches, weight_batches
                ):
                    yield X_batch, y_batch, weight_batch
            else:
                for X_batch, y_batch in zip(X_batches, y_batches):
                    yield X_batch, y_batch

    return generator, num_batches


def _build_generator_multi_three(
    X_list, y_list, batch_size, timesteps, input_size, shuffle, positive_weight
):
    """
    Build looping generator of training batches. Will return the generator and the number of batches in each epoch.
    In each epoch, all samples are randomly split into batch_size "chunks", each "chunk" in batch can be trained in parallel.
    Samples in each chunk are shuffled and merged into one whole sequence.
    The whole sequences are separated into batches of given fixed given number of timesteps (protein vectors).
    So the number of batches is defined so that we go over the whole sequence (length of the longest "chunk" sequence divided by the number of timesteps).

    :param X_list: List of samples. Each sample is a matrix/DataFrame of protein domain vectors.
    :param y_list: List of sample outputs.
    :param batch_size: Number of parallel "chunks" in a training batch
    :param timesteps: Number of timesteps (protein domain vectors) in a training batch
    :param input_size: Size of the protein domain vector
    :param shuffle: Whether to shuffle samples within each epoch. If not used, make sure that positive and negative samples are already shuffled in the list.
    :param positive_weight: Weight of positive samples (single number). If provided, a triple of (X_batch, y_batch, weights_batch) are provided
    :return: Tuple of (batch generator, number of batches in each epoch).
    Each batch will contain the X input (batch_size, timesteps, input_size) and y output (batch_size, timesteps, 1)
    """
    if not X_list:
        return _noop, None
    from keras.preprocessing.sequence import pad_sequences

    seq_length = sum([len(X) for X in X_list])
    X_arr1 = np.array([X.iloc[:, 0 : input_size[0]].values for X in X_list])
    X_arr2 = np.array(
        [
            X.iloc[:, input_size[0] : input_size[0] + input_size[1]].values
            for X in X_list
        ]
    )
    X_arr3 = np.array(
        [
            X.iloc[
                :,
                input_size[0]
                + input_size[1] : input_size[0]
                + input_size[1]
                + input_size[2],
            ].values
            for X in X_list
        ]
    )

    y_arr = np.array([y.values for y in y_list])
    input_size1 = X_arr1[0].shape[1]
    input_size2 = X_arr2[0].shape[1]
    input_size3 = X_arr3[0].shape[1]
    num_batches = int(np.ceil(np.ceil(seq_length / batch_size) / timesteps))
    maxlen = num_batches * timesteps
    logging.info(
        "Initializing generator of %s batches from sequence length %s",
        num_batches,
        seq_length,
    )

    def generator():
        while True:
            # shuffle the samples
            if shuffle:
                shuffled = np.random.permutation(len(X_list))
            # split samples into batch_size chunks
            X_batches1 = np.array_split(
                X_arr1[shuffled] if shuffle else X_arr1, batch_size
            )
            X_batches2 = np.array_split(
                X_arr2[shuffled] if shuffle else X_arr2, batch_size
            )
            X_batches3 = np.array_split(
                X_arr3[shuffled] if shuffle else X_arr3, batch_size
            )
            y_batches = np.array_split(
                y_arr[shuffled] if shuffle else y_arr, batch_size
            )

            # merge the samples in each chunk into one sequence
            X_batches1 = [
                np.concatenate(b) if b.size else np.empty(0) for b in X_batches1
            ]
            X_batches2 = [
                np.concatenate(b) if b.size else np.empty(0) for b in X_batches2
            ]
            X_batches3 = [
                np.concatenate(b) if b.size else np.empty(0) for b in X_batches3
            ]
            y_batches = [
                np.concatenate(b) if b.size else np.empty(0) for b in y_batches
            ]

            # pad the sequences with zeros to the length of the longest chunk sequence
            X_batches1 = pad_sequences(
                X_batches1,
                maxlen=maxlen,
                dtype=np.float,
                padding="post",
                truncating="post",
            )
            X_batches2 = pad_sequences(
                X_batches2,
                maxlen=maxlen,
                dtype=np.float,
                padding="post",
                truncating="post",
            )
            X_batches3 = pad_sequences(
                X_batches3,
                maxlen=maxlen,
                dtype=np.float,
                padding="post",
                truncating="post",
            )
            y_batches = pad_sequences(
                y_batches,
                maxlen=maxlen,
                dtype=np.float,
                padding="post",
                truncating="post",
            )

            # Reshape array so that it can be indexed as [batch number][chunk][timestep][input feature]
            # This will produce an array of dimension (num_batches, batch_size, timesteps, input_size)
            # And output array of dimension (num_batches, batch_size, timesteps, 1)
            X_batches1 = np.swapaxes(
                X_batches1.reshape(batch_size, num_batches, timesteps, input_size1),
                0,
                1,
            )
            X_batches2 = np.swapaxes(
                X_batches2.reshape(batch_size, num_batches, timesteps, input_size2),
                0,
                1,
            )
            X_batches3 = np.swapaxes(
                X_batches3.reshape(batch_size, num_batches, timesteps, input_size3),
                0,
                1,
            )
            y_batches = np.swapaxes(
                y_batches.reshape(batch_size, num_batches, timesteps, 1), 0, 1
            )

            if positive_weight:
                # Provide array of weights for each input vector based on the positive weight
                weight_batches = np.ones(y_batches.shape)
                weight_batches[y_batches == 1] = positive_weight
                weight_batches = np.swapaxes(
                    weight_batches.reshape(batch_size, num_batches, timesteps), 0, 1
                )
                for X_batch1, X_batch2, X_batch3, y_batch, weight_batch in zip(
                    X_batches1, X_batches2, X_batches3, y_batches, weight_batches
                ):
                    yield [X_batch1, X_batch2, X_batch3], y_batch, weight_batch
            else:
                for X_batch1, X_batch2, X_batch3, y_batch in zip(
                    X_batches1, X_batches2, X_batches3, y_batches
                ):
                    yield [X_batch1, X_batch2, X_batch3], y_batch

    return generator, num_batches


def _count_samples(y_list, klass):
    return np.sum([np.mean(y == klass) for y in y_list])


def _split_matrix_into_batches(X, batch_size):
    if len(X.shape) != 2:
        raise AttributeError(
            "Can only be called on a single 2-dimensional feature matrix"
        )
    return X.reshape(batch_size, X.shape[0], X.shape[1])


def _pad_matrix_to_be_divisible(X, divisible_by):
    from keras.preprocessing.sequence import pad_sequences

    remainder = X.shape[0] % divisible_by
    if not remainder:
        return X
    maxlen = X.shape[0] + divisible_by - remainder
    return pad_sequences(
        [X], maxlen=maxlen, dtype=np.float, padding="post", truncating="post"
    )[0]


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    import keras.backend as K

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    import keras.backend as K

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def auc_roc(y_true, y_pred):
    """
    Defines AUC ROC metric callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
    """
    import tensorflow as tf

    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if "auc_roc" in i.name.split("/")[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
