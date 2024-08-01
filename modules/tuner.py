
import os
import tensorflow_transform as tft
import tensorflow as tf
import kerastuner as kt
from tfx.v1.components import TunerFnResult
from tfx.components.trainer.fn_args_utils import FnArgs
from trainer import NUMERICAL_FEATURES, transformed_name, input_fn
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
import keras_tuner as kt

def model_builder(hyperparameters):
    input_features = []

    for key in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )

    concatenate = tf.keras.layers.concatenate(input_features)
    
    deep = tf.keras.layers.Dense(hyperparameters.Choice('unit_1', [128, 256]), activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(hyperparameters.Choice('dropout_1', [0.2, 0.4]))(deep)

    deep = tf.keras.layers.Dense(hyperparameters.Choice('unit_2', [64, 128]), activation="relu")(deep)
    deep = tf.keras.layers.Dropout(hyperparameters.Choice('dropout_2', [0.2, 0.4]))(deep)

    deep = tf.keras.layers.Dense(hyperparameters.Choice('unit_3', [32, 64]), activation="relu")(deep)
    deep = tf.keras.layers.Dropout(hyperparameters.Choice('dropout_3', [0.2, 0.4]))(deep)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters.Choice('learning_rate', [0.0001, 0.001])),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=10, num_epochs=1)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=10, num_epochs=1)

    '''tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=9,
        factor=3,
        directory=fn_args.working_dir,
        project_name='kt_hyperband')'''
    
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=15,
        directory=fn_args.working_dir,
        project_name='kt_hyperband'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            "epochs": 10
        }
    )
