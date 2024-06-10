import pathlib
import json
import os
import datetime
import random
from typing import List
import pytz
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project4.utils.DataIO import DataIO

MST_TZ_INFO = pytz.timezone('US/Arizona')
SUPERBOWL_START = datetime.datetime(2015, 2, 1, 4, 30, 0, 0, tzinfo=MST_TZ_INFO)
SUPERBOWL_END = datetime.datetime(2015, 2, 1, 8, 0, 0, 0, tzinfo=MST_TZ_INFO) # estimate
TIME_BUFFER = datetime.timedelta(hours=4)

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

    def get_a_tweet(self):
        with open(self.get_filepath_from_hashtag("#nfl")) as file:
            for tweet in file:
                parsed = json.loads(tweet)
                return parsed
            
    def get_tweet_fields(self, parsed_tweet, fields: List[str]):
        info = {}
        for field in fields:
            curr = parsed_tweet
            pathing = field.split("/")
            for item in pathing:
                curr = curr[item]
            info[field] = curr
        return info
            
    def sample_data(self, total_num, fields, hashtag_weights, time_weights, cache_io: DataIO, cache_name="sampled_tweets"):
        path = f'{cache_io.data_path}/{cache_name}'
        if os.path.exists(path):
            self.data = cache_io.load(cache_name)
        else:
            tweets = []
            for key, value in hashtag_weights.items():
                num_from_hashtag = total_num * value
                filepath = self.get_filepath_from_hashtag(key)
                before = []
                num_before = int(num_from_hashtag * time_weights['before'])
                during = []
                num_during = int(num_from_hashtag * time_weights['during'])
                after = []
                num_after = int(num_from_hashtag * time_weights['after'])
                with open(filepath) as file:
                    tweet_idx = 0
                    for tweet in file:
                        parsed = json.loads(tweet)
                        time = datetime.datetime.fromtimestamp(parsed['citation_date'], pytz.utc)
                        if time < SUPERBOWL_START - TIME_BUFFER:
                            before.append(tweet_idx)
                        elif time > SUPERBOWL_END + TIME_BUFFER:
                            after.append(tweet_idx)
                        else:
                            during.append(tweet_idx)
                        tweet_idx += 1
                sampled_idx = set()
                before_sample_num = num_before if len(before) >= num_before else len(before)
                during_sample_num = num_during if len(during) >= num_during else len(during)
                after_sample_num = num_after if len(after) >= num_after else len(after)
                sampled_idx = sampled_idx.union(set(random.sample(before, before_sample_num)))
                sampled_idx = sampled_idx.union(set(random.sample(during, during_sample_num)))
                sampled_idx = sampled_idx.union(set(random.sample(after, after_sample_num)))
                with open(filepath) as file:
                    tweet_idx = 0
                    for tweet in file:
                        if tweet_idx not in sampled_idx:
                            tweet_idx += 1
                            continue
                        else:
                            parsed = json.loads(tweet)
                            tweets.append(self.get_tweet_fields(parsed, fields))
                            tweet_idx += 1
            self.data = tweets
            cache_io.save(self.data, cache_name)
        return self.data
    
    def remove_indices(self, text, indices):
                    chars = list(text)
                    for index in indices:
                        last_index = index[1] if index[1] < len(chars) else len(chars)
                        for i in range(index[0], last_index+1):
                            chars[i-1] = '' #indices are 1 indexed
                    return "".join(chars)

    def process_sample(self, cache_io: DataIO, cache_name="processed_tweets"):
        path = f'{cache_io.data_path}/{cache_name}'
        if os.path.exists(path):
            self.processed_tweets = cache_io.load(cache_name)
        else:
            tweets = []
        
            for tweet in self.data:
                text = tweet['tweet/text']
                options = []
                if 'user_mentions' in tweet['tweet/entities']:
                    tweet['num_mentioned_users'] = len(tweet['tweet/entities']['user_mentions'])
                    options.append('user_mentions')
                if 'media' in tweet["tweet/entities"]:
                    tweet['num_media'] = len(tweet['tweet/entities']['media'])
                    options.append('media')
                if 'urls' in tweet["tweet/entities"]:
                    options.append('urls')
                if 'hashtags' in tweet["tweet/entities"]:
                    tweet['num_hashtags'] = len(tweet['tweet/entities']['hashtags'])
                    hashtags = []
                    for i in tweet['tweet/entities']['hashtags']:
                        hashtags.append(i['text'])
                    tweet['hashtags'] = " ".join(hashtags)
                    options.append('hashtags')

                for option in options:
                    for_removal = []
                    entities = tweet['tweet/entities'][option]
                    for ent in entities:
                        for_removal.append(ent['indices'])
                
                tweet['processed_text'] = self.remove_indices(text, for_removal)
                del tweet['tweet/entities']
                del tweet['tweet/text']
                tweets.append(tweet)

            self.processed_tweets = tweets
            cache_io.save(self.processed_tweets, cache_name)

        return self.processed_tweets
    
    def convert_to_df(self, cache_io: DataIO, cache_name="processed_tweets_df"):
        path = f'{cache_io.data_path}/{cache_name}'
        if os.path.exists(path):
            self.processed_tweets_df = cache_io.load(cache_name)
        else:
            self.processed_tweets_df = pd.DataFrame(self.processed_tweets)
            cache_io.save(self.processed_tweets_df, cache_name)
        return self.processed_tweets_df