"""
Processes data for model training building upon the data_preprocessing.py
Separation of concerns is used since this class can be subject to tuning,
the other simply prepares the data.
"""

import random
from math import ceil
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_preprocessing import DataPreprocessing


@dataclass
class DataParameters:
    padding_token: int = 0
    masking_token: int = 1
    pad_length: int = 40
    pad_side: str = "left"
    trunc_side: str = "left"  # keep latest interactions
    train_split: float = 0.7
    test_split: float = 0.15
    val_split: float = 0.15
    min_sequence_lenght: int = 3
    mask_probability: float = 0.15
    max_sequence_length: Optional[int] = 200


class BERT4RecDataset(Dataset):
    def __init__(self, df, params: DataParameters, is_train=True):
        self.df = df
        self.params = params
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = row["user_id"]
        sequence = row["input_seq"]
        position_ids = row["position_ids"]
        includes_target = row["target"]

        # Attention mask so that padding tokens are ignored
        attention_mask = [
            0 if item == self.params.padding_token else 1 for item in sequence
        ]

        labels = sequence.copy()

        # No target, so MLM
        if not includes_target:
            input_ids = self._apply_masking(sequence, attention_mask)

            # Set tokens that are not a label (not masked) to padding so they are ignored by the loss function
            for i in range(len(labels)):
                if input_ids[i] != self.params.masking_token:
                    labels[i] = self.params.padding_token

        # Sequence includes a target: the last element
        else:
            input_ids = sequence.copy()
            last_pos = len(sequence) - 1
            input_ids[last_pos] = self.params.masking_token
            for i in range(len(labels) - 1):
                labels[i] = self.params.padding_token

        return {
            "user_id": user_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _apply_masking(self, sequence, attention_mask):
        masked_sequence = np.array(sequence.copy())

        valid_positions = [i for i, mask in enumerate(attention_mask) if mask == 1]
        n_to_mask = max(1, int(len(valid_positions) * self.params.mask_probability))

        if n_to_mask > 0 and valid_positions:
            mask_positions = random.sample(valid_positions, n_to_mask)
            masked_sequence[mask_positions] = self.params.masking_token

        return masked_sequence


class DataProcessing:
    def __init__(
        self, preprocessor: DataPreprocessing, params: DataParameters = DataParameters()
    ):
        self.preprocessor = preprocessor
        self.params = params
        self.ratings = preprocessor.ratings

        self.train_df, self.val_df, self.test_df = self.split_user_sequences()

        self.train_df = self.pad_user_sequences(self.train_df)
        self.val_df = self.pad_user_sequences(self.val_df)
        self.test_df = self.pad_user_sequences(self.test_df)

    def get_max_sequence_length(self):
        if self.params.max_sequence_length is not None:
            return self.params.max_sequence_length

        max_length = 0

        for row in self.ratings["dense_movie_id"]:
            row_length = len(eval(row))
            if row_length > max_length:
                max_length = row_length

        return max_length

    def get_token_count(self):
        unique_movie_ids = set()

        for row in self.ratings["dense_movie_id"]:
            row = eval(row)
            unique_movie_ids.update(row)

        return max(unique_movie_ids) + 1

    def split_user_sequences(self):
        train_sequences = []
        val_sequences = []
        test_sequences = []

        for _, user_interactions in self.ratings.iterrows():
            user_id, interactions_str = user_interactions

            # Parse interactions if needed
            if isinstance(interactions_str, str):
                interactions_str = interactions_str.strip("[]")
                interactions = [
                    int(item.strip())
                    for item in interactions_str.split(",")
                    if item.strip()
                ]
            else:
                interactions = interactions_str  # If it's already a list

            if len(interactions) < self.params.min_sequence_lenght:
                continue

            if self.params.max_sequence_length is not None:
                interactions = interactions[-self.params.max_sequence_length :]

            train_end = int(self.params.train_split * len(interactions))
            val_end = train_end + ceil(self.params.val_split * len(interactions))

            if val_end == train_end:
                print("Skipping because we could split no validation set")
                continue

            if val_end >= len(interactions):
                print("Skipping because we could split no test set")
                continue

            # Train: MLM, no target. Use sliding window to generate many subsequences per user.
            train_seq = interactions[:train_end]
            min_len = min(self.params.min_sequence_lenght, len(train_seq))

            # Generate sliding window subsequences
            for end_idx in range(min_len, len(train_seq) + 1):
                # Take at most pad_length items to avoid excessive padding
                start_idx = max(0, end_idx - self.params.pad_length)
                subseq = train_seq[start_idx:end_idx]

                train_sequences.append(
                    {
                        "user_id": user_id,
                        "input_seq": subseq,
                        "position_ids": list(range(start_idx, end_idx)),
                        "target": False,
                    }
                )

            # Validation: Predict the next item after train sequence plus validation sequence
            val_sequences.append(
                {
                    "user_id": user_id,
                    "input_seq": interactions[:val_end],
                    "position_ids": list(range(0, val_end)),
                    "target": True,
                }
            )

            # Test: Predict the last item using the entire sequence
            test_sequences.append(
                {
                    "user_id": user_id,
                    "input_seq": interactions,
                    "position_ids": list(range(0, len(interactions))),
                    "target": True,
                }
            )

        return (
            pd.DataFrame(train_sequences),
            pd.DataFrame(val_sequences),
            pd.DataFrame(test_sequences),
        )

    def pad_user_sequences(self, df):
        """
        Pad or truncate sequences to standardized length in a vectorized manner.
        Needs to be performed for the three train, val and test sets.
        """

        def _process_ratings(row, column_name: str, padding_token):
            interactions = row[column_name]
            n_inter = len(interactions)

            if n_inter == self.params.pad_length:
                return interactions  # No padding needed
            elif n_inter < self.params.pad_length:
                padding_length = self.params.pad_length - n_inter
                padding = [padding_token] * padding_length

                if self.params.pad_side == "left":
                    return padding + interactions
                else:
                    return interactions + padding
            else:
                if self.params.trunc_side == "left":
                    return interactions[-self.params.pad_length :]
                else:
                    return interactions[: self.params.pad_length]

        df["input_seq"] = df.apply(
            lambda row: _process_ratings(row, "input_seq", self.params.padding_token),
            axis=1,
        )
        df["position_ids"] = df.apply(
            lambda row: _process_ratings(row, "position_ids", 0), axis=1
        )
        return df

    def get_dataloaders(self, batch_size=32):

        train_dataset = BERT4RecDataset(self.train_df, self.params)
        val_dataset = BERT4RecDataset(self.val_df, self.params)
        test_dataset = BERT4RecDataset(self.test_df, self.params)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
