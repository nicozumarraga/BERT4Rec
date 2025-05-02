from dataclasses import dataclass
import os
import time

import tyro
import pandas as pd
import torch
import random
import numpy as np

from data_preprocessing import DataPreprocessing
from data_processing import DataParameters, DataProcessing
from training import Bert4RecTrainingParams, train


EXPERIMENT_FILE_PLACEHOLDER = "<EXPERIMENT>"


@dataclass
class Args:
    experiment: str
    seed: int = 42
    start_at: int = 0
    results_dir: str = "./results"
    results_file: str = f"{EXPERIMENT_FILE_PLACEHOLDER}_results.csv"
    data_path: str = "./data"
    max_epochs: int = 100


# TODO: encapsulate DataParameters and Bert4RecTrainingParams into one.
def run_experiment(
    args: Args, configurations: list[tuple[DataParameters, Bert4RecTrainingParams]]
):
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, args.results_file)
    results_file_exists = os.path.exists(results_path)
    if args.start_at > 0 and not results_file_exists:
        raise ValueError(
            f"Cannot start at iteration {args.start_at} if there is no existing results file"
        )

    data_preprocessor = DataPreprocessing(args.data_path)

    start_seconds = time.time()

    for idx, (data_parameters, configuration) in enumerate(configurations):
        if idx < args.start_at:
            continue

        data_processor = DataProcessing(
            preprocessor=data_preprocessor, params=data_parameters
        )

        configuration.epochs = args.max_epochs

        # infer some settings from processed data
        configuration.num_pos = data_processor.get_max_sequence_length()
        configuration.vocab_size = data_processor.get_token_count()

        # train the model, hopefully until it early stops
        test_results, val_results, used_epochs = train(data_processor, configuration)

        # save the results with all settings to be processed later
        results = [
            {f"test_{k}": v for k, v in test_results.items()}
            | {f"val_{k}": v for k, v in val_results.items()}
            | {"used_epochs": used_epochs, "experiment": args.experiment}
            | configuration.__dict__
            | data_parameters.__dict__
        ]

        pd.DataFrame(results).to_csv(
            results_path, mode="w" if idx == 0 else "a", index=False, header=(idx == 0)
        )

        time_per_i = (time.time() - start_seconds) / (idx + 1 - args.start_at)
        left_iterations = len(configurations) - idx - 1
        print(
            f"Finished experiment iteration. Experiment will take {time_per_i * left_iterations: 0.2f} seconds to complete."
        )


def layer_size_experiment(args: Args):
    run_experiment(
        args,
        [
            (DataParameters(), Bert4RecTrainingParams(hidden_layer_size=x))
            for x in (64, 128, 256)
        ],
    )


def layer_count_experiment(args: Args):
    run_experiment(
        args,
        [
            (DataParameters(), Bert4RecTrainingParams(num_hidden_layers=x))
            for x in (2, 4, 8)
        ],
    )


def masking_ratio_experiment(args: Args):
    run_experiment(
        args,
        [
            (DataParameters(mask_probability=x), Bert4RecTrainingParams())
            for x in (0.15, 0.2, 0.4)
        ],
    )


# TODO: experiment with positional encoding strategies like tuncating longer sequences
# to a max length of 500 or by just using relative position id's, that start from 0 for each sequence.


def main(args: Args):
    args.results_file = args.results_file.replace(
        EXPERIMENT_FILE_PLACEHOLDER, args.experiment
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.experiment == "num_layers":
        layer_count_experiment(args)
    elif args.experiment == "layer_size":
        layer_size_experiment(args)
    elif args.experiment == "masking_ratio":
        masking_ratio_experiment(args)
    else:
        raise NotImplementedError(f"Could not find experiment {args.experiment}")


if __name__ == "__main__":
    main(tyro.cli(Args))
