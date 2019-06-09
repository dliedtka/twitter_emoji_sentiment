import re
import string
import emoji
import pickle
import os


emoji_list = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../usable/emoji_list.pkl", "rb"))[:400]


def preprocess(tweets, lowercase=True): 
    '''
    preprocess tweets to all lowercase (optional), tokenize usernames and URLs, remove repeated letters; 
    return new array of tweets
    convert to lowercase on by default, but their may be value in all caps for sentiment
    tweets over 140 character and featuring non-ascii were filtered out during collection this time
    '''

    # no duplicate tweets
    tweets_no_duplicates = list(dict.fromkeys(tweets))

    # new array to return
    new_tweets = []

    # iterate through all tweets
    counter = 0
    for tweet in tweets_no_duplicates:
        counter += 1
        print ("preprocessing: " + str(counter))
        new_tweet = tweet
        
        # remove retweets (still include tweet, just remove RT)
        if tweet[:3] == "RT ":
            new_tweet = tweet[3:]

        # convert to all lowercase (optional)
        if lowercase:
            new_tweet = new_tweet.lower()

        # tokenize usernames (@____) to USERNAME
        new_tweet = re.sub(r'@\S+', "<user>", new_tweet)

        # tokenize URLs to URL
        # any URLs that start with http:// or https://
        new_tweet = re.sub(r'https?://\S+', "<url>", new_tweet)
        # any URLs missed above in the format www.____.___
        new_tweet = re.sub(r'www\.\S+\.\S+', "<url>", new_tweet)

        # remove repeated letters (3+ to 2), just iterate through 26 letters
        for c in string.ascii_lowercase:
            new_tweet = re.sub(r'' + c*3 + '+', "" + c*2, new_tweet)

        # separate emojis from words with spaces, replace hashtags with HASHTAG token
        final_tweet = ""
        for c in new_tweet:
            if c in emoji_list:
                final_tweet += " " + c + " "
            elif c == "#":
                final_tweet += " <hashtag> "
            else:
                final_tweet += c

        new_tweets.append(final_tweet)

    # ensure no tweets became duplicates (ie same tweet, different usernames)
    final_tweets = list(dict.fromkeys(new_tweets))
    return final_tweets
