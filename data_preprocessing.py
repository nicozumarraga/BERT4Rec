import os
import pandas as pd
import numpy as np
import torch
import random

class Dataset:
    def __init__(self, path = "data/"):
        self.path = path
        self.preprocessed_ratings_path = os.path.join(self.path, "user_ratings.csv")

        if os.path.exists(self.preprocessed_ratings_path):
            self.ratings = pd.read_csv(self.preprocessed_ratings_path)
        else:
            self.ratings_path = os.path.join(self.path, "ratings.dat")
            self.ratings = self.preprocess_ratings(self.load_ratings_as_df(self.ratings_path))
            self.ratings.to_csv(self.preprocessed_ratings_path, index=False)

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
        ratings_processed = self._filter_less_than_4(ratings_processed)
        ratings_processed.loc[:, 'rating'] = 1 # convert to binary signal
        ratings_processed = self._map_ratings_to_dense_sequence(ratings_processed)
        ratings_processed = self._filter_less_than_5_interactions(ratings_processed)
        ratings_processed = self._reformat_into_list_and_drop_others(ratings_processed)
        return ratings_processed

    def _filter_less_than_4(self, ratings):
        return ratings.loc[ratings.rating >= 4]

    def _map_ratings_to_dense_sequence(self, ratings):
        """
        Used to map the movie_id to a dense sequence for better training performance.
        We reserve the first two ids for padding and `MASKED` values.
        """
        dense_movie_ids = {k:i+2 for i, k in enumerate(sorted(list(ratings.movie_id.unique())))}
        ratings["dense_movie_id"] = ratings["movie_id"].map(dense_movie_ids)
        return ratings

    def _filter_less_than_5_interactions(self, ratings):
        user_interactions = ratings.groupby('user_id').size()
        out_users = user_interactions[user_interactions < 5].index.tolist()
        ratings = ratings[~ratings['user_id'].isin(out_users)]
        return ratings

    def _reformat_into_list_and_drop_others(self, ratings):
        ratings.sort_values(by=['timestamp'], inplace=True)
        ratings = ratings.groupby('user_id').agg(list).drop(['rating', 'timestamp','movie_id'], axis=1)
        return ratings
