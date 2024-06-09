import pathlib
import json
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
class TweetsDataset:
    hashtags = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]
    def __init__(self, folder_path="../data/ECE219_tweet_data/") -> None:
        self.folder_path = folder_path
        self.elapsed_times = None

    def get_filepath_from_hashtag(self, hashtag):
        path = pathlib.Path(self.folder_path)
        filename = f"tweets_{hashtag}.txt"
        filepath = path.joinpath(filename)
        return filepath
    
    def get_elapsed_time(self):
        if self.elapsed_times is None:
            results = {}
            for hashtag in self.hashtags:
                earliest_tweet_time = datetime.datetime.max
                latest_tweet_time = datetime.datetime.min
                filepath = self.get_filepath_from_hashtag(hashtag)
                with open(filepath) as file:
                    for tweet in tqdm(file):
                        parsed = json.loads(tweet)
                        time_posted = datetime.datetime.fromtimestamp(parsed['citation_date'])
                        if time_posted < earliest_tweet_time:
                            earliest_tweet_time = time_posted
                        if time_posted > latest_tweet_time:
                            latest_tweet_time = time_posted
                    elapsed_time = (latest_tweet_time - earliest_tweet_time)
                results[hashtag] = elapsed_time
            self.elapsed_times = results
        
        return self.elapsed_times
    
    def average_tweets_per_hour(self):
        results = {}
        elapsed_times = self.get_elapsed_time()
        for hashtag in self.hashtags:
            filepath = self.get_filepath_from_hashtag(hashtag)
            with open(filepath) as f:
                num_tweets = sum(1 for _ in f)
            elapsed_time = elapsed_times[hashtag].total_seconds()
            elapsed_time_hours = elapsed_time / (3600)
            average_tweets_per_hour = num_tweets / elapsed_time_hours
            results[hashtag] = average_tweets_per_hour
        return results
    
    def average_followers(self):
        results = {}
        for hashtag in self.hashtags:
            filepath = self.get_filepath_from_hashtag(hashtag)
            with open(filepath) as file:
                num_tweets = 0
                num_followers = 0
                for tweet in tqdm(file):
                    num_tweets += 1
                    parsed = json.loads(tweet)
                    followers = parsed['author']['followers']
                    num_followers += followers
            average_followers = num_followers / num_tweets
            results[hashtag] = average_followers
        return results
    
    def average_retweets(self):
        results = {}
        for hashtag in self.hashtags:
            filepath = self.get_filepath_from_hashtag(hashtag)
            with open(filepath) as file:
                num_tweets = 0
                num_retweets = 0
                for tweet in tqdm(file):
                    num_tweets += 1
                    parsed = json.loads(tweet)
                    retweets = parsed['metrics']['citations']['total']
                    num_retweets += retweets
            average_retweets = num_retweets / num_tweets
            results[hashtag] = average_retweets
        return results

    def plot_tweets_time_hist(self, hashtag):
        if hashtag not in self.hashtags:
            raise ValueError("Bad hashtag")
        elapsed_time = self.get_elapsed_time()[hashtag]
        filepath = self.get_filepath_from_hashtag(hashtag)
        tweet_times = []
        with open(filepath) as file:
            for tweet in tqdm(file):
                parsed = json.loads(tweet)
                time_posted = datetime.datetime.fromtimestamp(parsed['citation_date'])
                tweet_times.append(time_posted)
        earliest_tweet_time = min(tweet_times)
        print(earliest_tweet_time)
        tweet_times_conv = [(x - earliest_tweet_time).total_seconds() / 3600 for x in tweet_times]
        num_bins = (elapsed_time.total_seconds() // 3600) + 1
        bins = np.arange(num_bins + 1)
        plt.hist(tweet_times_conv, bins=bins)
        plt.xlabel("Time (hours)")
        plt.ylabel("Tweets")
        plt.title(f"Histogram of Tweets per Hour {hashtag}")
        plt.show()
