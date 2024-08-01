
"""
Author: abiyamf
Date: 1/8/2023
This is the components.py module.
Usage:
- ECreate a pipeline with TFX components.
"""

# Import library
import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

# Fungsi untuk melakukan inisialisasi components


def init_components(config):
    """Returns tfx components for the pipeline.

    Args:
        data_dir (str): Directory containing the dataset.
        transform_module (str): Path to the transform module.
        tuner_module (str): Path to the tuner module.
        training_module (str): Path to the training module.
        training_steps (int): Number of training steps.
        eval_steps (int): Number of evaluation steps.
        serving_model_dir (str): Directory to save the serving

    Returns:
        components: Tuple of TFX components.
    """

    # Membagi dataset dengan perbandingan 8:2
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    # Komponen example gen
    example_gen = CsvExampleGen(
        input_base=config["DATA_ROOT"],
        output_config=output
    )

    # Komponen statistics gen
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Komponen schema gen
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    # Komponen example validator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Komponen transform. Menggunakan module transform.py
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(config["transform_module"])
    )

    # Komponen tuner. Menggunakan module tuner.py
    tuner = Tuner(
        module_file=os.path.abspath(config["tuner_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config["training_steps"]),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config["eval_steps"]),
    )

    # Komponen trainer. Menggunakan module trainer.py
    trainer = Trainer(
        module_file=os.path.abspath(config["training_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config["training_steps"]),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config["eval_steps"])
    )

    # Komponen model resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name="Precision"),
            tfma.MetricConfig(class_name="Recall"),
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.8}),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={'value': 0.0001})
                              )
                              )
        ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='HiringDecision')],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=metrics_specs
    )

    # Komponen evaluator
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Komponen pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config["serving_model_dir"]
            )
        ),
    )

    # Mengembalikan semua komponen
    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    # Mengembalikan komponen
    return components
