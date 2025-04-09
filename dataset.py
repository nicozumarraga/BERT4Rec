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
        self.ratings = self.preprocess_ratings(self.load_ratings_as_df(path))

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
        ratings_processed = self.order_by_timestamp(ratings_processed)
        ratings_processed = self.filter_less_than_5(ratings_processed)
        ratings_processed = self.perform_sequence_truncation(ratings_processed,
                                                        sequence_length=20)
        ratings_processed = self.split_train_test(ratings_processed,
                                             train=0.7,
                                             val=0.15,
                                             test=0.15)
        return ratings_processed

    def convert_to_binary(self, ratings):
        pass

    def order_by_timestamp(self, ratings):
        pass

    def filter_less_than_5(self, ratings):
        pass

    def perform_sequence_truncation(self, ratings):
        pass

    def split_train_test(self, ratings):
        pass
