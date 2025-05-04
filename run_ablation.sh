#!/bin/sh

python run_experiments.py --experiment num_layers & python run_experiments.py --experiment layer_size & python run_experiments.py --experiment masking_ratio & python run_experiments.py --experiment max_sequence_length & python run_experiments.py --experiment sequence_length

