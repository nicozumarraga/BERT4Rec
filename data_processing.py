"""
Processes data for model training building upon the data_preprocessing.py
Separation of concerns is used since this class can be subject to tuning,
the other simply prepares the data.
"""
from dataclasses import dataclass
from data_preprocessing import Dataset
import pandas as pd
from enum import IntEnum

@dataclass
class DataParameters():
    PAD: int = 0
    MASK: int = 1
    pad_length: int = 20
    pad_side: str = "left"
    trunc_side: str = "left" # keep latest interactions



class DataProcessing:
    def __init__(self):
        dataset = Dataset()
        self.params = DataParameters()
        self.ratings = dataset.ratings

    def split_user_sequences(self,
                            train: float = 0.7,
                            validation: float = 0.15,
                            test: float = 0.15
                            ):
        """
        Assumptions made (## TODO - double check):
        we want to split sequences chronologically.
        we do not want to have overlapping interactions per set (unclear from
        the slides image but seems like the more logical approach)

        """
        train_data = []
        val_data = []
        test_data = []

        for _, user_interactions in self.ratings.iterrows():
            user_id, interactions_str = user_interactions

            if isinstance(interactions_str, str):
                interactions_str = interactions_str.strip('[]')
                interactions = [int(item.strip()) for item in interactions_str.split(',') if item.strip()]
            else:
                interactions = interactions_str  # If it's already a list

            n_train = int(train * len(interactions))
            n_val = int(validation * len(interactions))

            train_data.append({'user_id': user_id, 'ratings': interactions[:n_train]})
            val_data.append({'user_id': user_id, 'ratings': interactions[n_train:(n_train+n_val)]})
            test_data.append({'user_id': user_id, 'ratings': interactions[(n_train+n_val):]})

        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)

        return train_df, val_df, test_df

    def pad_user_sequences(self, df):
        """
        Pad or truncate sequences to standardized length in a vectorized manner.
        Needs to be performed for the three train, val and test sets.
        """

        def _process_ratings(row):
            interactions = row['ratings']

            if isinstance(interactions, str):
                interactions = interactions.strip('[]')
                interactions = [int(item.strip()) for item in interactions.split(',') if item.strip()]

            n_inter = len(interactions)

            if n_inter == self.params.pad_length:
                return interactions  # No padding needed
            elif n_inter < self.params.pad_length:
                padding_length = self.params.pad_length - n_inter
                padding = [self.params.PAD] * padding_length

                if self.params.pad_side == "left":
                    return padding + interactions
                else:
                    return interactions + padding
            else:
                if self.params.trunc_side == "left":
                    return interactions[-self.params.pad_length:]
                else:
                    return interactions[:self.params.pad_length]

        df['ratings'] = df.apply(_process_ratings, axis=1)
        return df
