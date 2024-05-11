import pandas as pd

def format_ratings(ratings_df: pd.DataFrame):
    return ratings_df[['rating', 'userId', 'movieId']]

def ingest_data():
    links = pd.read_csv('../data/links.csv')
    movies = pd.read_csv('../data/movies.csv')
    ratings = format_ratings(pd.read_csv('../data/ratings.csv'))
    tags = pd.read_csv('../data/tags.csv')
    raws = {
        "links": links,
        "movies": movies,
        "ratings": ratings,
        "tags": tags
    }
    table = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).to_numpy()
    return table, raws


    