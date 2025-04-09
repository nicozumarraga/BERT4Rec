import os
import pandas as pd
import numpy as np
import torch
import random

class Dataset:
    def __init__(self, path):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.path = path
        self.ratings_path = os.path.join(path, "ratings.dat")
        self.ratings = self.preprocess_ratings(self.load_ratings_as_df(self.ratings_path))

    def load_ratings_as_df(self, filename):
        ratings = pd.read_csv(
            filename,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
        return ratings

    def preprocess_ratings(self, ratings):
        ratings_processed = ratings.copy()
        ratings_processed = self.convert_to_binary(ratings_processed)
        ratings_processed = self.filter_less_than_5(ratings_processed)
        ratings_processed = self.order_by_timestamp(ratings_processed)
        #ratings_processed = self.perform_sequence_truncation(ratings_processed)
        #ratings_processed = self.split_train_test(ratings_processed)
        return ratings_processed

    def convert_to_binary(self, ratings):
        ratings = ratings.loc[ratings['rating'] >= 4]
        ratings.loc[:, 'rating'] = 1
        return ratings

    def order_by_timestamp(self, ratings):
        return ratings.sort_values(by=['user_id', 'timestamp'], ascending=True)

    def filter_less_than_5(self, ratings):
        return ratings.groupby('user_id').filter(lambda x: len(x) >= 5)

    def perform_sequence_truncation(self, ratings, sequence_length: int = 20):
        pass

    def split_train_test(self, ratings, train: float = 0.7, val: float = 0.15, test: float = 0.15):
        pass
