import torch
import random
import numpy as np
from torch.utils.data import Dataset

from data_processing import DataParameters


# TODO: test this code
class BERT4RecDataset(Dataset):
    def __init__(self, df, params: "DataParameters", is_train=True):
        self.df = df
        self.params = params
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = row["user_id"]
        sequence = row["input_seq"]
        includes_target = row["target"]

        # Attention mask so that padding tokens are ignored
        attention_mask = [
            0 if item == self.params.padding_token else 1 for item in sequence
        ]

        labels = sequence.copy()

        # No target, so MLM
        if not includes_target:
            input_ids = self._apply_masking(sequence, attention_mask)

            # Set tokens that are not a label (not masked) to -100
            for i in range(len(labels)):
                if input_ids[i] != self.params.masking_token:
                    labels[i] = -100

        # Sequence includes a target: the last element
        else:
            input_ids = sequence.copy()
            last_pos = len(sequence) - 1
            input_ids[last_pos] = self.params.masking_token
            for i in range(len(labels) - 1):
                labels[i] = -100

        return {
            "user_id": user_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _apply_masking(self, sequence, attention_mask):
        masked_sequence = np.array(sequence.copy())

        valid_positions = [i for i, mask in enumerate(attention_mask) if mask == 1]
        n_to_mask = max(1, int(len(valid_positions) * self.params.mask_probability))

        if n_to_mask > 0 and valid_positions:
            mask_positions = random.sample(valid_positions, n_to_mask)
            masked_sequence[mask_positions] = self.params.MASK

        return masked_sequence
