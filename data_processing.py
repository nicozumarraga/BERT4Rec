"""
Processes data for model training building upon the data_preprocessing.py
Separation of concerns is used since this class can be subject to tuning,
the other simply prepares the data.
"""

from dataclasses import dataclass
from data_preprocessing import DataPreprocessing
import pandas as pd
from torch.utils.data import DataLoader
from math import ceil

from dataset import BERT4RecDataset


@dataclass
class DataParameters:
    padding_token: int = 0
    masking_token: int = -1
    pad_length: int = 20
    pad_side: str = "left"
    trunc_side: str = "left"  # keep latest interactions
    train_split: float = 0.7
    test_split: float = 0.15
    val_split: float = 0.15
    min_sequence_lenght: int = 3
    mask_probability: float = 0.15


# TODO: again test this code


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

            train_end = int(self.params.train_split * len(interactions))
            val_end = train_end + ceil(self.params.val_split * len(interactions))

            if val_end == train_end:
                print("Skipping because we could split no validation set")
                continue

            if val_end >= len(interactions):
                print("Skipping because we could split no test set")
                continue

            # TODO: possibly generate subsequences for training

            # Train: MLM, no target
            train_sequences.append(
                {
                    "user_id": user_id,
                    "input_seq": interactions[:train_end],
                    "target": False,
                }
            )

            # Validation: Predict the next item after train sequence plus validation sequence
            val_sequences.append(
                {
                    "user_id": user_id,
                    "input_seq": interactions[:val_end],
                    "target": True,
                }
            )

            # Test: Predict the last item using the entire sequence
            test_sequences.append(
                {
                    "user_id": user_id,
                    "input_seq": interactions,
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

        def _process_ratings(row):
            interactions = row["input_seq"]
            n_inter = len(interactions)

            if n_inter == self.params.pad_length:
                return interactions  # No padding needed
            elif n_inter < self.params.pad_length:
                padding_length = self.params.pad_length - n_inter
                padding = [self.params.padding_token] * padding_length

                if self.params.pad_side == "left":
                    return padding + interactions
                else:
                    return interactions + padding
            else:
                if self.params.trunc_side == "left":
                    return interactions[-self.params.pad_length :]
                else:
                    return interactions[: self.params.pad_length]

        df["input_seq"] = df.apply(_process_ratings, axis=1)
        return df

    def get_dataloaders(self, batch_size=32):

        train_dataset = BERT4RecDataset(self.train_df, self.params)
        val_dataset = BERT4RecDataset(self.val_df, self.params)
        test_dataset = BERT4RecDataset(self.test_df, self.params)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
