import pandas as pd
from surprise.dataset import Dataset
from surprise.reader import Reader

class MovieRatingsDataset ():

    def __init__(self) -> None:
        self.table, raws = self.ingest_data()
        ratings = raws.pop('ratings')
        self.ratings = ratings
        self.metadata = raws
        self.og_dataset = self.dataset_from_df(ratings)
        self.popular_dataset = self.dataset_from_df(self.popular_trimming(ratings))
        self.unpopular_dataset = self.dataset_from_df(self.unpopular_trimming(ratings))
        self.high_variance_dataset = self.dataset_from_df(self.high_variance_trimming(ratings))

    def format_ratings(self, ratings_df: pd.DataFrame):
        return ratings_df[['userId', 'movieId', 'rating']]

    def ingest_data(self):
        links = pd.read_csv('../data/links.csv')
        movies = pd.read_csv('../data/movies.csv')
        ratings = self.format_ratings(pd.read_csv('../data/ratings.csv'))
        tags = pd.read_csv('../data/tags.csv')
        raws = {
            "links": links,
            "movies": movies,
            "ratings": ratings,
            "tags": tags
        }
        table = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).to_numpy()
        return table, raws


    def popular_trimming(self, ratings_df: pd.DataFrame):
        popular_movies = ratings_df['movieId'].value_counts() > 2
        popular_indexes = []
        for i in popular_movies.index:
            if popular_movies.loc[i]:
                popular_indexes.append(i)
        popular_ratings = ratings_df[ratings_df['movieId'].isin(popular_indexes)]
        return popular_ratings

    def unpopular_trimming(self, ratings_df: pd.DataFrame):
        unpopular_movies = ratings_df['movieId'].value_counts() <= 2
        unpopular_indexes = []
        for i in unpopular_movies.index:
            if unpopular_movies.loc[i]:
                unpopular_indexes.append(i)
        unpopular_ratings = ratings_df[ratings_df['movieId'].isin(unpopular_indexes)]
        return unpopular_ratings

    def high_variance_trimming(self, ratings_df: pd.DataFrame):
        popular_movies = ratings_df['movieId'].value_counts() >= 5
        popular_indexes = []
        for i in popular_movies.index:
            if popular_movies.loc[i]:
                popular_indexes.append(i)
        popular_ratings = ratings_df[ratings_df['movieId'].isin(popular_indexes)]

        movie_rating_vars = popular_ratings.groupby(['movieId']).var()

        high_var_movies = movie_rating_vars['rating'] >= 2
        high_var_indexes = []
        for j in high_var_movies.index:
            if high_var_movies.loc[j]:
                high_var_indexes.append(j)
        high_var_ratings = popular_ratings[popular_ratings['movieId'].isin(high_var_indexes)]
        return high_var_ratings


    def dataset_from_df(self, ratings_df):
        reader = Reader(rating_scale=(0.5, 5))
        dataset = Dataset.load_from_df(ratings_df, reader)
        return dataset

    